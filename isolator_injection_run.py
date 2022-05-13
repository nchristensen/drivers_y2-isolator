"""Simulation driver for the Y2 Isolator case."""

__copyright__ = """
Copyright (C) 2020 University of Illinois Board of Trustees
"""

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""
import logging
import sys
import yaml
import numpy as np
import pyopencl as cl
import numpy.linalg as la  # noqa
import pyopencl.array as cla  # noqa
from functools import partial
from pytools.obj_array import make_obj_array

from arraycontext import thaw, freeze
from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
from grudge.eager import EagerDGDiscretization
from grudge.shortcuts import make_visualizer
from grudge.dof_desc import DTAG_BOUNDARY
#from grudge.op import nodal_max, nodal_min
from logpyle import IntervalTimer, set_dt
from mirgecom.logging_quantities import (
    initialize_logmgr,
    logmgr_add_cl_device_info,
    logmgr_set_time,
    set_sim_state
)

from mirgecom.navierstokes import ns_operator
from mirgecom.artificial_viscosity import \
    av_laplacian_operator, smoothness_indicator
from mirgecom.simutil import (
    check_step,
    write_visfile,
    check_naninf_local,
    check_range_local,
    get_sim_timestep
)
from mirgecom.restart import write_restart_file
from mirgecom.io import make_init_message
from mirgecom.mpi import mpi_entry_point
import pyopencl.tools as cl_tools
from mirgecom.integrators import (rk4_step, lsrk54_step, lsrk144_step,
                                  euler_step)

from mirgecom.steppers import advance_state
from mirgecom.boundary import (
    PrescribedFluidBoundary,
    IsothermalWallBoundary,
)
from mirgecom.eos import IdealSingleGas, PyrometheusMixture
from mirgecom.transport import SimpleTransport
from mirgecom.gas_model import GasModel, make_fluid_state
from mirgecom.fluid import make_conserved


class SingleLevelFilter(logging.Filter):
    """Filter I/O."""
    def __init__(self, passlevel, reject):
        """Initialize the I/O filter."""
        self.passlevel = passlevel
        self.reject = reject

    def filter(self, record):
        """Filter it."""
        if self.reject:
            return (record.levelno != self.passlevel)
        else:
            return (record.levelno == self.passlevel)


class MyRuntimeError(RuntimeError):
    """Simple exception to kill the simulation."""

    pass


class SparkSource:
    r"""Deposit energy from an ignition source."

    Internal energy is deposited as a gaussian of the form:

    .. math::

        e &= e + e_{a}\exp^{(1-r^{2})}\\

    .. automethod:: __init__
    .. automethod:: __call__
    """

    def __init__(self, *, dim, center=None, width=1.0,
                 amplitude=0., amplitude_func=None):
        r"""Initialize the spark parameters.

        Parameters
        ----------
        center: numpy.ndarray
            center of source
        amplitude: float
            source strength modifier
        amplitude_fun: function
            variation of amplitude with time
        """
        if center is None:
            center = np.zeros(shape=(dim,))
        self._center = center
        self._dim = dim
        self._amplitude = amplitude
        self._width = width
        self._amplitude_func = amplitude_func

    def __call__(self, x_vec, cv, time, **kwargs):
        """
        Create the energy deposition at time *t* and location *x_vec*.

        the source at time *t* is created by evaluting the gaussian
        with time-dependent amplitude at *t*.

        Parameters
        ----------
        cv: :class:`mirgecom.fluid.ConservedVars`
            Fluid conserved quantities
        time: float
            Current time at which the solution is desired
        x_vec: numpy.ndarray
            Nodal coordinates
        """
        t = time
        if self._amplitude_func is not None:
            amplitude = self._amplitude*self._amplitude_func(t)
        else:
            amplitude = self._amplitude

        #print(f"{time=} {amplitude=}")

        loc = self._center

        # coordinates relative to lump center
        rel_center = make_obj_array(
            [x_vec[i] - loc[i] for i in range(self._dim)]
        )
        actx = x_vec[0].array_context
        r = actx.np.sqrt(np.dot(rel_center, rel_center))
        expterm = amplitude * actx.np.exp(-(r**2)/(2*self._width*self._width))

        mass = 0*cv.mass
        momentum = 0*cv.momentum
        species_mass = 0*cv.species_mass

        energy = cv.energy + cv.mass*expterm

        return make_conserved(dim=self._dim, mass=mass, energy=energy,
                              momentum=momentum, species_mass=species_mass)


#MJA TODO this is removed in the slimmer PR
def sponge_source(cv, cv_ref, sigma):
    return sigma*(cv_ref - cv)


class InitSponge:
    r"""Initialize sponge.

    .. automethod:: __init__
    .. automethod:: __call__
    """

    def __init__(self, *, x0, thickness, amplitude):
        r"""Initialize the sponge parameters.

        Parameters
        ----------
        x0: float
            sponge starting x location
        thickness: float
            sponge extent
        amplitude: float
            sponge strength modifier
        """
        self._x0 = x0
        self._thickness = thickness
        self._amplitude = amplitude

    def __call__(self, x_vec, *, time=0.0):
        """Create the sponge intensity at locations *x_vec*.

        Parameters
        ----------
        x_vec: numpy.ndarray
            Coordinates at which solution is desired
        time: float
            Time at which solution is desired. The strength is (optionally)
            dependent on time
        """
        xpos = x_vec[0]
        actx = xpos.array_context
        zeros = 0*xpos
        x0 = zeros + self._x0

        return self._amplitude * actx.np.where(
            actx.np.greater(xpos, x0),
            (zeros + ((xpos - self._x0)/self._thickness) *
            ((xpos - self._x0)/self._thickness)),
            zeros + 0.0
        )


@mpi_entry_point
def main(ctx_factory=cl.create_some_context,
         restart_filename=None, target_filename=None,
         use_profiling=False, use_logmgr=True, user_input_file=None,
         use_overintegration=False, actx_class=False, casename=None,
         lazy=False):

    """Drive the isolator case."""
    if actx_class is None:
        raise RuntimeError("Array context class missing.")

    # control log messages
    logger = logging.getLogger(__name__)
    logger.propagate = False

    if (logger.hasHandlers()):
        logger.handlers.clear()

    # send info level messages to stdout
    h1 = logging.StreamHandler(sys.stdout)
    f1 = SingleLevelFilter(logging.INFO, False)
    h1.addFilter(f1)
    logger.addHandler(h1)

    # send everything else to stderr
    h2 = logging.StreamHandler(sys.stderr)
    f2 = SingleLevelFilter(logging.INFO, True)
    h2.addFilter(f2)
    logger.addHandler(h2)

    cl_ctx = ctx_factory()

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nparts = comm.Get_size()

    from mirgecom.simutil import global_reduce as _global_reduce
    global_reduce = partial(_global_reduce, comm=comm)

    if casename is None:
        casename = "mirgecom"

    # logging and profiling
    log_path = "log_data/"
    logname = log_path + casename + ".sqlite"

    if rank == 0:
        import os
        log_dir = os.path.dirname(logname)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)

    logmgr = initialize_logmgr(use_logmgr,
        filename=logname, mode="wo", mpi_comm=comm)

    if use_profiling:
        queue = cl.CommandQueue(cl_ctx,
            properties=cl.command_queue_properties.PROFILING_ENABLE)
    else:
        queue = cl.CommandQueue(cl_ctx)

    # main array context for the simulation
    if lazy:
        actx = actx_class(comm, queue,
                allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue)),
                mpi_base_tag=12000)
    else:
        actx = actx_class(comm, queue,
                allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue)),
                force_device_scalars=True)

    # default i/o junk frequencies
    nviz = 500
    nhealth = 1
    nrestart = 5000
    nstatus = 1

    # default timestepping control
    integrator = "rk4"
    current_dt = 1e-8
    t_final = 1e-7
    current_t = 0
    current_step = 0
    current_cfl = 1.0
    constant_cfl = False

    # default health status bounds
    health_pres_min = 1.0e-1
    health_pres_max = 2.0e6
    health_temp_min = 1.0
    health_temp_max = 5000
    health_mass_frac_min = -10
    health_mass_frac_max = 10

    # discretization and model control
    order = 1
    alpha_sc = 0.3
    s0_sc = -5.0
    kappa_sc = 0.5
    dim = 2

    # material properties
    mu = 1.0e-5
    spec_diff = 1e-4
    mu_override = False  # optionally read in from input
    nspecies = 0

    # rhs control
    use_ignition = False
    use_sponge = True
    use_av = True
    use_combustion = False

    # initialize the ignition spark
    spark_center = np.zeros(shape=(dim,))
    spark_center[0] = 0.677
    spark_center[1] = -0.021
    if dim == 3:
        spark_center[2] = 0.035/2.
    spark_diameter = 0.0025

    spark_strength = 30000000./current_dt
    #spark_strength = 5e-3

    spark_init_time = 999999999.

    spark_duration = 1.e-8

    if user_input_file:
        input_data = None
        if rank == 0:
            with open(user_input_file) as f:
                input_data = yaml.load(f, Loader=yaml.FullLoader)
        input_data = comm.bcast(input_data, root=0)
        try:
            nviz = int(input_data["nviz"])
        except KeyError:
            pass
        try:
            nrestart = int(input_data["nrestart"])
        except KeyError:
            pass
        try:
            nhealth = int(input_data["nhealth"])
        except KeyError:
            pass
        try:
            nstatus = int(input_data["nstatus"])
        except KeyError:
            pass
        try:
            current_dt = float(input_data["current_dt"])
        except KeyError:
            pass
        try:
            t_final = float(input_data["t_final"])
        except KeyError:
            pass
        try:
            alpha_sc = float(input_data["alpha_sc"])
        except KeyError:
            pass
        try:
            kappa_sc = float(input_data["kappa_sc"])
        except KeyError:
            pass
        try:
            s0_sc = float(input_data["s0_sc"])
        except KeyError:
            pass
        try:
            mu_input = float(input_data["mu"])
            mu_override = True
        except KeyError:
            pass
        try:
            spec_diff = float(input_data["spec_diff"])
        except KeyError:
            pass
        try:
            nspecies = int(input_data["nspecies"])
        except KeyError:
            pass
        try:
            order = int(input_data["order"])
        except KeyError:
            pass
        try:
            dim = int(input_data["dimen"])
        except KeyError:
            pass
        try:
            integrator = input_data["integrator"]
        except KeyError:
            pass
        try:
            health_pres_min = float(input_data["health_pres_min"])
        except KeyError:
            pass
        try:
            health_pres_max = float(input_data["health_pres_max"])
        except KeyError:
            pass
        try:
            health_temp_min = float(input_data["health_temp_min"])
        except KeyError:
            pass
        try:
            health_temp_max = float(input_data["health_temp_max"])
        except KeyError:
            pass
        try:
            health_mass_frac_min = float(input_data["health_mass_frac_min"])
        except KeyError:
            pass
        try:
            health_mass_frac_max = float(input_data["health_mass_frac_max"])
        except KeyError:
            pass
        try:
            use_ignition = bool(input_data["use_ignition"])
        except KeyError:
            pass
        try:
            spark_init_time = float(input_data["ignition_init_time"])
        except KeyError:
            pass
        try:
            use_sponge = bool(input_data["use_sponge"])
        except KeyError:
            pass
        try:
            use_av = bool(input_data["use_av"])
        except KeyError:
            pass
        try:
            use_combustion = bool(input_data["use_combustion"])
        except KeyError:
            pass

    # param sanity check
    allowed_integrators = ["rk4", "euler", "lsrk54", "lsrk144"]
    if integrator not in allowed_integrators:
        error_message = "Invalid time integrator: {}".format(integrator)
        raise RuntimeError(error_message)

    s0_sc = np.log10(1.0e-4 / np.power(order, 4))
    if rank == 0:
        print(f"Shock capturing parameters: alpha {alpha_sc}, "
              f"s0 {s0_sc}, kappa {kappa_sc}")

    if rank == 0:
        print("\n#### Simluation control data: ####")
        print(f"\tnviz = {nviz}")
        print(f"\tnrestart = {nrestart}")
        print(f"\tnhealth = {nhealth}")
        print(f"\tnstatus = {nstatus}")
        print(f"\tcurrent_dt = {current_dt}")
        print(f"\tt_final = {t_final}")
        print(f"\torder = {order}")
        print(f"\tdimen = {dim}")
        print(f"\tTime integration {integrator}")
        print("#### Simluation control data: ####\n")

    if rank == 0 and use_ignition:
        print("\n#### Ignition control parameters ####")
        print(f"spark center ({spark_center[0]},{spark_center[1]})")
        print(f"spark FWHM {spark_diameter}")
        print(f"spark strength {spark_strength}")
        print(f"ignition time {spark_init_time}")
        print(f"ignition duration {spark_duration}")
        print("#### Ignition control parameters ####\n")

    timestepper = rk4_step
    if integrator == "euler":
        timestepper = euler_step
    if integrator == "lsrk54":
        timestepper = lsrk54_step
    if integrator == "lsrk144":
        timestepper = lsrk144_step

    # }}}
    # working gas: O2/N2 #
    #   O2 mass fraction 0.273
    #   gamma = 1.4
    #   cp = 37.135 J/mol-K,
    #   rho= 1.977 kg/m^3 @298K
    gamma = 1.4
    mw_o2 = 15.999*2
    mw_n2 = 14.0067*2
    mf_o2 = 0.273
    # viscosity @ 400C, Pa-s
    mu_o2 = 3.76e-5
    mu_n2 = 3.19e-5
    mu_mix = mu_o2*mf_o2 + mu_n2*(1-mu_o2)  # 3.3456e-5
    mw = mw_o2*mf_o2 + mw_n2*(1.0 - mf_o2)
    r = 8314.59/mw
    cp = r*gamma/(gamma - 1)
    Pr = 0.75

    if mu_override:
        mu = mu_input
    else:
        mu = mu_mix

    kappa = cp*mu/Pr
    init_temperature = 300.0

    # Turn off combustion unless EOS supports it
    if nspecies < 3:
        use_combustion = False

    if rank == 0:
        print("\n#### Simluation material properties: ####")
        print(f"\tmu = {mu}")
        print(f"\tkappa = {kappa}")
        print(f"\tPrandtl Number  = {Pr}")
        print(f"\tnspecies = {nspecies}")
        print(f"\tspecies diffusivity = {spec_diff}")
        if nspecies == 0:
            print("\tno passive scalars, uniform ideal gas eos")
        elif nspecies == 2:
            print("\tpassive scalars to track air/fuel mixture, ideal gas eos")
        else:
            print("\tfull multi-species initialization with pyrometheus eos")

    #spec_diffusivity = 0. * np.ones(nspecies)
    spec_diffusivity = spec_diff * np.ones(nspecies)
    transport_model = SimpleTransport(viscosity=mu, thermal_conductivity=kappa,
                                      species_diffusivity=spec_diffusivity)

    # make the eos
    if nspecies < 3:
        eos = IdealSingleGas(gamma=gamma, gas_const=r)
        species_names = ["air", "fuel"]
    else:
        from mirgecom.thermochemistry import get_pyrometheus_wrapper_class
        from mirgecom.mechanisms.uiuc import Thermochemistry
        pyro_mech = get_pyrometheus_wrapper_class(Thermochemistry)(actx.np)
        eos = PyrometheusMixture(pyro_mech, temperature_guess=init_temperature)
        species_names = pyro_mech.species_names

    gas_model = GasModel(eos=eos, transport=transport_model)

    viz_path = "viz_data/"
    vizname = viz_path + casename
    restart_path = "restart_data/"
    restart_pattern = (
        restart_path + "{cname}-{step:06d}-{rank:04d}.pkl"
    )
    if restart_filename:  # read the grid from restart data
        restart_filename = f"{restart_filename}-{rank:04d}.pkl"

        from mirgecom.restart import read_restart_data
        restart_data = read_restart_data(actx, restart_filename)
        current_step = restart_data["step"]
        current_t = restart_data["t"]
        local_mesh = restart_data["local_mesh"]
        local_nelements = local_mesh.nelements
        global_nelements = restart_data["global_nelements"]
        restart_order = int(restart_data["order"])
        # will use this later
        #restart_nspecies = int(restart_data["nspecies"])

        assert restart_data["num_parts"] == nparts
        assert restart_data["nspecies"] == nspecies
    else:
        error_message = "Driver only supports restart. Start with -r <filename>"
        raise RuntimeError(error_message)

    if target_filename:  # read the grid from restart data
        target_filename = f"{target_filename}-{rank:04d}.pkl"

        from mirgecom.restart import read_restart_data
        target_data = read_restart_data(actx, target_filename)
        target_order = int(target_data["order"])
        # will use this later

        assert restart_data["num_parts"] == nparts
        assert restart_data["nspecies"] == nspecies
        assert restart_data["global_nelements"] == target_data["global_nelements"]
    else:
        logger.warning("No target file specied, using restart as target")

    if rank == 0:
        logging.info("Making discretization")

    from grudge.dof_desc import DISCR_TAG_BASE, DISCR_TAG_QUAD
    from meshmode.discretization.poly_element import \
          default_simplex_group_factory, QuadratureSimplexGroupFactory

    discr = EagerDGDiscretization(
        actx, local_mesh,
        discr_tag_to_group_factory={
            DISCR_TAG_BASE: default_simplex_group_factory(
                base_dim=local_mesh.dim, order=order),
            DISCR_TAG_QUAD: QuadratureSimplexGroupFactory(2*order + 1)
        },
        mpi_communicator=comm
    )

    if use_overintegration:
        quadrature_tag = DISCR_TAG_QUAD
    else:
        quadrature_tag = None

    if rank == 0:
        logging.info("Done making discretization")

    vis_timer = None

    if logmgr:
        logmgr_add_cl_device_info(logmgr, queue)

        logmgr.add_watches([
            ("step.max", "step = {value}, "),
            ("t_sim.max", "sim time: {value:1.6e} s, "),
            ("t_step.max", "step walltime: {value:6g} s")
            #("t_log.max", "log walltime: {value:6g} s")
        ])

        try:
            logmgr.add_watches(["memory_usage.max"])
        except KeyError:
            pass

        if use_profiling:
            logmgr.add_watches(["pyopencl_array_time.max"])

        vis_timer = IntervalTimer("t_vis", "Time spent visualizing")
        logmgr.add_quantity(vis_timer)

    if rank == 0:
        logging.info("Before restart/init")

    def get_fluid_state(cv, temperature_seed):
        return make_fluid_state(cv=cv, gas_model=gas_model,
                                temperature_seed=temperature_seed)

    create_fluid_state = actx.compile(get_fluid_state)

    if restart_filename:
        if rank == 0:
            logging.info("Restarting soln.")
        restart_cv = restart_data["cv"]
        temperature_seed = restart_data["temperature_seed"]
        if restart_order != order:
            restart_discr = EagerDGDiscretization(
                actx,
                local_mesh,
                order=restart_order,
                mpi_communicator=comm)
            from meshmode.discretization.connection import make_same_mesh_connection
            connection = make_same_mesh_connection(
                actx,
                discr.discr_from_dd("vol"),
                restart_discr.discr_from_dd("vol")
            )
            restart_cv = connection(restart_data["cv"])
            temperature_seed = connection(restart_data["temperature_seed"])

        if logmgr:
            logmgr_set_time(logmgr, current_step, current_t)
    else:
        error_message = "Driver only supports restart. Start with -r <filename>"
        raise RuntimeError(error_message)

    if target_filename:
        if rank == 0:
            logging.info("Reading target soln.")
        target_cv = target_data["cv"]
        if target_order != order:
            target_discr = EagerDGDiscretization(
                actx,
                local_mesh,
                order=target_order,
                mpi_communicator=comm)
            from meshmode.discretization.connection import make_same_mesh_connection
            connection = make_same_mesh_connection(
                actx,
                discr.discr_from_dd("vol"),
                target_discr.discr_from_dd("vol")
            )
            target_cv = connection(target_data["cv"])

        if logmgr:
            logmgr_set_time(logmgr, current_step, current_t)
    else:
        target_cv = restart_cv

    current_state = create_fluid_state(restart_cv, temperature_seed)
    target_state = create_fluid_state(target_cv, temperature_seed)
    temperature_seed = current_state.temperature

    # if you divide by 2.355, 50% of the spark is within this diameter
    spark_diameter /= 2.355
    # if you divide by 6, 99% of the energy is deposited in this time
    spark_duration /= 6.0697

    # gaussian application in time
    def spark_time_func(t):
        expterm = actx.np.exp((-(t - spark_init_time)**2) /
                              (2*spark_duration*spark_duration))
        return expterm

    #spark_strength = 0.0
    ignition_source = SparkSource(dim=dim, center=spark_center,
                                  amplitude=spark_strength,
                                  amplitude_func=spark_time_func,
                                  width=spark_diameter)

    # initialize the sponge field
    sponge_thickness = 0.09
    sponge_amp = 1.0/current_dt/1000
    sponge_x0 = 0.9

    sponge_init = InitSponge(x0=sponge_x0, thickness=sponge_thickness,
                             amplitude=sponge_amp)
    x_vec = thaw(discr.nodes(), actx)

    #TODO MJA, inline sponge sigma with a compiled helper function
    sponge_sigma = sponge_init(x_vec=x_vec)

    '''
    def _sponge_sigma(x_vec):
        return sponge_init(x_vec=x_vec)

    get_sponge_sigma = actx.compile(_sponge_sigma)
    sponge_sigma = get_sponge_sigma(x_vec)

    def _sponge_source(cv):
        """Create sponge source."""
        return sponge_sigma*(current_state.cv - cv)
    '''


    #TODO MJA, rewrite boundary routine so avoid project at every invocation
    # set the boundary conditions
    def _ref_state_func(discr, btag, gas_model, ref_state, **kwargs):
        from mirgecom.gas_model import project_fluid_state
        from grudge.dof_desc import DOFDesc, as_dofdesc
        dd_base_vol = DOFDesc("vol")
        return project_fluid_state(discr, dd_base_vol,
                                   as_dofdesc(btag).with_discr_tag(quadrature_tag),
                                   ref_state, gas_model)

    _ref_boundary_state_func = partial(_ref_state_func, ref_state=target_state)

    '''
    from mirgecom.gas_model import project_fluid_state
    from grudge.dof_desc import DOFDesc, as_dofdesc
    dd_base_vol = DOFDesc("vol")

    flow_boundary_btag = DTAG_BOUNDARY("flow")
    flow_ref_state = project_fluid_state(
        discr, dd_base_vol,
        as_dofdesc(flow_boundary_btag).with_discr_tag(quadrature_tag),
        target_state, gas_model
        )

    # Set boundary conditions
    def _target_flow_state_func(**kwargs):
        return flow_ref_state

    target_state_boundary = PrescribedFluidBoundary(boundary_state_func=_target_flow_state_func)
    '''

    target_state_boundary = PrescribedFluidBoundary(boundary_state_func=_ref_boundary_state_func)
    wall = IsothermalWallBoundary()

    boundaries = {
        DTAG_BOUNDARY("inflow"): target_state_boundary,
        DTAG_BOUNDARY("outflow"): target_state_boundary,
        DTAG_BOUNDARY("injection"): target_state_boundary,
        DTAG_BOUNDARY("wall"): wall
    }

    # MJA TODO, lump boundaries together 
    '''
    boundaries = {
        DTAG_BOUNDARY("flow"): target_state_boundary,
        DTAG_BOUNDARY("wall"): wall
    }
    '''
    #from mirgecom.simutil import boundary_report
    #boundary_report(discr, boundaries, f"{casename}_boundaries_np{nparts}.yaml")

    visualizer = make_visualizer(discr)

    #    initname = initializer.__class__.__name__
    eosname = eos.__class__.__name__
    init_message = make_init_message(dim=dim, order=order, nelements=local_nelements,
                                     global_nelements=global_nelements,
                                     dt=current_dt, t_final=t_final, nstatus=nstatus,
                                     nviz=nviz, cfl=current_cfl,
                                     constant_cfl=constant_cfl, initname=casename,
                                     eosname=eosname, casename=casename)
    if rank == 0:
        logger.info(init_message)

    # some utility functions
    def vol_min_loc(x):
        from grudge.op import nodal_min_loc
        return actx.to_numpy(nodal_min_loc(discr, "vol", x))[()]

    def vol_max_loc(x):
        from grudge.op import nodal_max_loc
        return actx.to_numpy(nodal_max_loc(discr, "vol", x))[()]

    def vol_min(x):
        from grudge.op import nodal_min
        return actx.to_numpy(nodal_min(discr, "vol", x))[()]

    def vol_max(x):
        from grudge.op import nodal_max
        return actx.to_numpy(nodal_max(discr, "vol", x))[()]

    def my_write_status(cv, dv, dt, cfl):
        status_msg = f"-------- dt = {dt:1.3e}, cfl = {cfl:1.4f}"
        temperature = thaw(freeze(dv.temperature, actx), actx)
        pressure = thaw(freeze(dv.pressure, actx), actx)
        p_min = vol_min(pressure)
        p_max = vol_max(pressure)
        t_min = vol_min(temperature)
        t_max = vol_max(temperature)

        from pytools.obj_array import obj_array_vectorize
        y_min = obj_array_vectorize(lambda x: vol_min(x),
                                      cv.species_mass_fractions)
        y_max = obj_array_vectorize(lambda x: vol_max(x),
                                      cv.species_mass_fractions)

        dv_status_msg = (
            f"\n-------- P (min, max) (Pa) = ({p_min:1.9e}, {p_max:1.9e})")
        dv_status_msg += (
            f"\n-------- T (min, max) (K)  = ({t_min:7g}, {t_max:7g})")
        for i in range(nspecies):
            dv_status_msg += (
                f"\n-------- y_{species_names[i]} (min, max) = "
                f"({y_min[i]:1.3e}, {y_max[i]:1.3e})")
        status_msg += dv_status_msg
        status_msg += "\n"

        if rank == 0:
            logger.info(status_msg)

    def my_write_viz(step, t, cv, dv, ts_field, alpha_field):
        tagged_cells = smoothness_indicator(discr, cv.mass, s0=s0_sc,
                                            kappa=kappa_sc)

        mach = (actx.np.sqrt(np.dot(cv.velocity, cv.velocity)) /
                            dv.speed_of_sound)

        viz_fields = [("cv", cv),
                      ("dv", dv),
                      ("mach", mach),
                      ("velocity", cv.velocity),
                      ("sponge_sigma", sponge_sigma),
                      ("alpha", alpha_field),
                      ("tagged_cells", tagged_cells),
                      ("dt" if constant_cfl else "cfl", ts_field)]
        # species mass fractions
        viz_fields.extend(
            ("Y_"+species_names[i], cv.species_mass_fractions[i])
            for i in range(nspecies))
        write_visfile(discr, viz_fields, visualizer, vizname=vizname,
                      step=step, t=t, overwrite=True)

    def my_write_restart(step, t, cv, temperature_seed):
        restart_fname = restart_pattern.format(cname=casename, step=step, rank=rank)
        if restart_fname != restart_filename:
            restart_data = {
                "local_mesh": local_mesh,
                "cv": cv,
                "temperature_seed": temperature_seed,
                "nspecies": nspecies,
                "t": t,
                "step": step,
                "order": order,
                "global_nelements": global_nelements,
                "num_parts": nparts
            }
            write_restart_file(actx, restart_data, restart_fname, comm)

    def my_health_check(cv, dv):
        health_error = False
        if check_naninf_local(discr, "vol", dv.pressure):
            health_error = True
            logger.info(f"{rank=}: NANs/Infs in pressure data.")

        if global_reduce(check_range_local(discr, "vol", dv.pressure,
                                     health_pres_min, health_pres_max),
                                     op="lor"):
            health_error = True
            p_min = vol_min(dv.pressure)
            p_max = vol_max(dv.pressure)
            logger.info(f"Pressure range violation: "
                        f"Simulation Range ({p_min=}, {p_max=}) "
                        f"Specified Limits ({health_pres_min=}, {health_pres_max=})")

        if global_reduce(check_range_local(discr, "vol", dv.temperature,
                                     health_temp_min, health_temp_max),
                                     op="lor"):
            health_error = True
            t_min = vol_min(dv.temperature)
            t_max = vol_max(dv.temperature)
            logger.info(f"Temperature range violation: "
                        f"Simulation Range ({t_min=}, {t_max=}) "
                        f"Specified Limits ({health_temp_min=}, {health_temp_max=})")

        for i in range(nspecies):
            if global_reduce(check_range_local(discr, "vol",
                                               cv.species_mass_fractions[i],
                                         health_mass_frac_min, health_mass_frac_max),
                                         op="lor"):
                health_error = True
                y_min = vol_min(cv.species_mass_fractions[i])
                y_max = vol_max(cv.species_mass_fractions[i])
                logger.info(f"Species mass fraction range violation. "
                            f"{species_names[i]}: ({y_min=}, {y_max=})")

        return health_error

    def my_get_viscous_timestep(discr, state, alpha):
        """Routine returns the the node-local maximum stable viscous timestep.

        Parameters
        ----------
        discr: grudge.eager.EagerDGDiscretization
            the discretization to use
        state: :class:`~mirgecom.gas_model.FluidState`
            Full fluid state including conserved and thermal state
        alpha: :class:`~meshmode.DOFArray`
            Arfifical viscosity

        Returns
        -------
        :class:`~meshmode.dof_array.DOFArray`
            The maximum stable timestep at each node.
        """
        from grudge.dt_utils import characteristic_lengthscales

        length_scales = characteristic_lengthscales(state.array_context, discr)

        nu = 0
        d_alpha_max = 0

        if state.is_viscous:
            nu = state.viscosity/state.mass_density
            # this appears to break lazy for whatever reason
            from mirgecom.viscous import get_local_max_species_diffusivity
            d_alpha_max = \
                get_local_max_species_diffusivity(
                    state.array_context,
                    state.species_diffusivity
                )

        return(
            length_scales / (state.wavespeed
            + ((nu + d_alpha_max + alpha) / length_scales))
        )

    def my_get_viscous_cfl(discr, dt, state, alpha):
        """Calculate and return node-local CFL based on current state and timestep.

        Parameters
        ----------
        discr: :class:`grudge.eager.EagerDGDiscretization`
            the discretization to use
        dt: float or :class:`~meshmode.dof_array.DOFArray`
            A constant scalar dt or node-local dt
        state: :class:`~mirgecom.gas_model.FluidState`
            Full fluid state including conserved and thermal state
        alpha: :class:`~meshmode.DOFArray`
            Arfifical viscosity

        Returns
        -------
        :class:`~meshmode.dof_array.DOFArray`
            The CFL at each node.
        """
        return dt / my_get_viscous_timestep(discr, state=state, alpha=alpha)

    def my_get_timestep(t, dt, state, alpha):
        t_remaining = max(0, t_final - t)
        if constant_cfl:
            ts_field = current_cfl * my_get_viscous_timestep(discr, state=state,
                                                             alpha=alpha)
            from grudge.op import nodal_min
            dt = actx.to_numpy(nodal_min(discr, "vol", ts_field))
            cfl = current_cfl
        else:
            ts_field = my_get_viscous_cfl(discr, dt=dt, state=state, alpha=alpha)
            from grudge.op import nodal_max
            cfl = actx.to_numpy(nodal_max(discr, "vol", ts_field))

        return ts_field, cfl, min(t_remaining, dt)

    def my_get_alpha(discr, state, alpha):
        """ Scale alpha by the element characteristic length """
        from grudge.dt_utils import characteristic_lengthscales
        array_context = state.array_context
        length_scales = characteristic_lengthscales(array_context, discr)

        #from mirgecom.fluid import compute_wavespeed
        #wavespeed = compute_wavespeed(eos, state)

        vmag = array_context.np.sqrt(np.dot(state.velocity, state.velocity))
        #alpha_field = alpha*wavespeed*length_scales
        alpha_field = alpha*vmag*length_scales
        #alpha_field = wavespeed*0 + alpha*current_step
        #alpha_field = state.mass

        return alpha_field

    #TODO MJA, calculate the length scale outside of the main stepping loop
    # also remove discr from aux function
    '''
    from grudge.dt_utils import characteristic_lengthscales
    length_scales = characteristic_lengthscales(actx, discr)

    def my_get_alpha(state, alpha):
        """Scale alpha by the element characteristic length and velocity"""
        return alpha*state.speed*length_scales

    # what is this for? MJA TODO
    def get_sc_scale():
        return alpha_sc * length_scales

    get_sc_scale_compiled = actx.compile(get_sc_scale)

    sc_scale = get_sc_scale_compiled()
    '''

    def my_pre_step(step, t, dt, state):
        cv, tseed = state
        fluid_state = create_fluid_state(cv=cv, temperature_seed=tseed)
        dv = fluid_state.dv

        try:
            if logmgr:
                logmgr.tick_before()

            alpha_field = my_get_alpha(discr, fluid_state, alpha_sc)
            # MJA TODO, this goes with new my_get_alpha
            '''
            alpha_field = my_get_alpha(fluid_state, alpha_sc)
            '''
            ts_field, cfl, dt = my_get_timestep(t, dt, fluid_state, alpha_field)

            do_viz = check_step(step=step, interval=nviz)
            do_restart = check_step(step=step, interval=nrestart)
            do_health = check_step(step=step, interval=nhealth)
            do_status = check_step(step=step, interval=nstatus)

            if do_health:
                health_errors = global_reduce(my_health_check(cv, dv), op="lor")
                if health_errors:
                    if rank == 0:
                        logger.warning("Fluid solution failed health check.")
                    raise MyRuntimeError("Failed simulation health check.")

            if do_status:
                my_write_status(dt=dt, cfl=cfl, dv=dv, cv=cv)

            if do_restart:
                my_write_restart(step=step, t=t, cv=cv, temperature_seed=tseed)

            if do_viz:
                my_write_viz(step=step, t=t, cv=cv, dv=dv,
                             ts_field=ts_field, alpha_field=alpha_field)

        except MyRuntimeError:
            if rank == 0:
                logger.error("Errors detected; attempting graceful exit.")
            my_write_viz(step=step, t=t, cv=cv, dv=dv, ts_field=ts_field,
                         alpha_field=alpha_field)
            my_write_restart(step=step, t=t, cv=cv, temperature_seed=tseed)
            raise

        return state, dt

    def my_post_step(step, t, dt, state):
        cv, tseed = state
        # MJA TODO, we don't need to make a fluid state here, just return state, dt
        fluid_state = create_fluid_state(cv=cv, temperature_seed=tseed)
        # Logmgr needs to know about EOS, dt, dim?
        # imo this is a design/scope flaw
        if logmgr:
            set_dt(logmgr, dt)
            set_sim_state(logmgr, dim, cv, gas_model.eos)
            logmgr.tick_after()
        return make_obj_array([fluid_state.cv, fluid_state.temperature]), dt

    def my_rhs_without_combustion(t, state):
        cv, tseed = state
        fluid_state = make_fluid_state(cv=cv, gas_model=gas_model,
                                       temperature_seed=tseed)
        alpha_field = my_get_alpha(discr, fluid_state, alpha_sc)
        cv_rhs = (
            ns_operator(discr, state=fluid_state, time=t, boundaries=boundaries,
                        gas_model=gas_model, quadrature_tag=quadrature_tag)
            + av_laplacian_operator(discr, fluid_state=fluid_state,
                                    boundaries=boundaries,
                                    boundary_kwargs={"time": t,
                                                     "gas_model": gas_model},
                                    alpha=alpha_field, s0=s0_sc, kappa=kappa_sc,
                                    quadrature_tag=quadrature_tag)
            + sponge_source(cv=fluid_state.cv, cv_ref=target_cv, sigma=sponge_sigma)
        )
        return make_obj_array([cv_rhs, 0*tseed])

    # TODO MJA, merge these rhs implementations back together using the flags for turning different parts on/off
    # provide grad cv to ns_operator and av_operator so we don't have to recompute it

    '''
    from mirgecom.gas_model import make_operator_fluid_states
    from mirgecom.navierstokes import grad_cv_operator

    def my_rhs(t, state):
        cv, tseed = state
        fluid_state = make_fluid_state(cv=cv, gas_model=gas_model,
                                       temperature_seed=tseed)
        # Temperature seed RHS (keep tseed updated)
                tseed_rhs = fluid_state.temperature - tseed

        # Steps common to NS and AV (and wall model needs grad(temperature))
        operator_fluid_states = make_operator_fluid_states(
            discr, fluid_state, gas_model, boundaries, quadrature_tag)

        grad_fluid_cv = grad_cv_operator(
            discr, gas_model, boundaries, fluid_state,
            quadrature_tag=quadrature_tag,
            operator_states_quad=operator_fluid_states)

        # Fluid CV RHS contributions
        ns_rhs = \
        ns_operator(discr, state=fluid_state, time=t, boundaries=boundaries,
            gas_model=gas_model, quadrature_tag=quadrature_tag,
            operator_states_quad=operator_fluid_states,
            grad_cv=grad_fluid_cv)

        chem_rhs = 0*cv
        if use_combustion:  # conditionals evaluated only once at compile time
            chem_rhs =  \
                eos.get_species_source_terms(cv, temperature=fluid_state.temperature)

        av_rhs = 0*cv
        if use_av:
            alpha_field = sc_scale * fluid_state.speed
            av_rhs = \
            av_laplacian_operator(discr, fluid_state=fluid_state,
                  boundaries=boundaries, gas_model=gas_model,
                  time=t, alpha=alpha_field, s0=s0_sc,
                  kappa=kappa_sc, quadrature_tag=quadrature_tag,
                        operator_states_quad=operator_fluid_states)

        sponge_rhs = 0*cv
        if use_sponge:
            sponge_rhs = _sponge_source(cv=cv)

        ignition_rhs = 0*cv
        if use_ignition:
            ignition_rhs = ignition_source(x_vec=x_vec, cv=cv, time=t)

        cv_rhs = ns_rhs + chem_rhs + av_rhs + sponge_rhs + ignition_rhs
        return make_obj_array([cv_rhs, tseed_rhs])
    '''

    def my_rhs_with_combustion(t, state):
        cv, tseed = state
        fluid_state = make_fluid_state(cv=cv, gas_model=gas_model,
                                       temperature_seed=tseed)
        alpha_field = my_get_alpha(discr, fluid_state, alpha_sc)
        cv_rhs = (
            ns_operator(discr, state=fluid_state, time=t, boundaries=boundaries,
                        gas_model=gas_model, quadrature_tag=quadrature_tag)
            + eos.get_species_source_terms(cv,
                                           temperature=fluid_state.temperature)
            + av_laplacian_operator(discr, fluid_state=fluid_state,
                                    boundaries=boundaries,
                                    boundary_kwargs={"time": t,
                                                     "gas_model": gas_model},
                                    alpha=alpha_field, s0=s0_sc, kappa=kappa_sc,
                                    quadrature_tag=quadrature_tag)
            + sponge_source(cv=cv, cv_ref=restart_cv, sigma=sponge_sigma)
        )
        return make_obj_array([cv_rhs, 0*tseed])

    def my_rhs_with_ignition(t, state):
        cv, tseed = state
        fluid_state = make_fluid_state(cv=cv, gas_model=gas_model,
                                       temperature_seed=tseed)
        alpha_field = my_get_alpha(discr, fluid_state, alpha_sc)
        cv_rhs = (
            ns_operator(discr, state=fluid_state, time=t, boundaries=boundaries,
                        gas_model=gas_model, quadrature_tag=quadrature_tag)
            + eos.get_species_source_terms(cv,
                                           temperature=fluid_state.temperature)
            + av_laplacian_operator(discr, fluid_state=fluid_state,
                                    boundaries=boundaries,
                                    boundary_kwargs={"time": t,
                                                     "gas_model": gas_model},
                                    alpha=alpha_field, s0=s0_sc, kappa=kappa_sc,
                                    quadrature_tag=quadrature_tag)
            + ignition_source(x_vec=x_vec, cv=cv, time=t)
            + sponge_source(cv=cv, cv_ref=restart_cv, sigma=sponge_sigma)
        )
        return make_obj_array([cv_rhs, 0*tseed])

    current_dt = get_sim_timestep(discr, current_state, current_t, current_dt,
                                  current_cfl, t_final, constant_cfl)

    my_rhs = my_rhs_without_combustion
    if nspecies > 2:
        if use_ignition:
            my_rhs = my_rhs_with_ignition
        else:
            my_rhs = my_rhs_with_combustion

    current_step, current_t, stepper_state = \
        advance_state(rhs=my_rhs, timestepper=timestepper,
                      pre_step_callback=my_pre_step,
                      post_step_callback=my_post_step,
                      istep=current_step, dt=current_dt,
                      t=current_t, t_final=t_final,
                      state=make_obj_array([current_state.cv, temperature_seed]))
    current_cv, tseed = stepper_state
    # TODO MJA, use compiled version, also rename so we can tell the difference
    '''
    current_state = create_fluid_state(current_cv, tseed)
    '''
    current_state = make_fluid_state(current_cv, gas_model, tseed)

    # Dump the final data
    if rank == 0:
        logger.info("Checkpointing final state ...")
    final_dv = current_state.dv
    # TODO MJA, goes with new my_get_alpha
    '''
    alpha_field = my_get_alpha(current_state, alpha_sc)
    '''
    alpha_field = my_get_alpha(discr, current_state, alpha_sc)
    ts_field, cfl, dt = my_get_timestep(t=current_t, dt=current_dt,
                                        state=current_state, alpha=alpha_field)
    my_write_status(dt=dt, cfl=cfl, cv=current_state.cv, dv=final_dv)

    my_write_viz(step=current_step, t=current_t, cv=current_state.cv, dv=final_dv,
                 ts_field=ts_field, alpha_field=alpha_field)
    my_write_restart(step=current_step, t=current_t, cv=current_state.cv,
                     temperature_seed=tseed)

    if logmgr:
        logmgr.close()
    elif use_profiling:
        print(actx.tabulate_profiling_data())

    finish_tol = 1e-16
    assert np.abs(current_t - t_final) < finish_tol


if __name__ == "__main__":

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO)

    import argparse
    parser = argparse.ArgumentParser(
        description="MIRGE-Com Isentropic Nozzle Driver")
    parser.add_argument("-r", "--restart_file", type=ascii, dest="restart_file",
                        nargs="?", action="store", help="simulation restart file")
    parser.add_argument("-t", "--target_file", type=ascii, dest="target_file",
                        nargs="?", action="store", help="simulation target file")
    parser.add_argument("-i", "--input_file", type=ascii, dest="input_file",
                        nargs="?", action="store", help="simulation config file")
    parser.add_argument("-c", "--casename", type=ascii, dest="casename", nargs="?",
                        action="store", help="simulation case name")
    parser.add_argument("--profile", action="store_true", default=False,
                        help="enable kernel profiling [OFF]")
    parser.add_argument("--log", action="store_true", default=False,
                        help="enable logging profiling [ON]")
    parser.add_argument("--lazy", action="store_true", default=False,
                        help="enable lazy evaluation [OFF]")
    parser.add_argument("--overintegration", action="store_true",
        help="use overintegration in the RHS computations")

    args = parser.parse_args()

    # for writing output
    casename = "isolator"
    if args.casename:
        print(f"Custom casename {args.casename}")
        casename = args.casename.replace("'", "")
    else:
        print(f"Default casename {casename}")

    lazy = args.lazy
    if args.profile:
        if lazy:
            raise ValueError("Can't use lazy and profiling together.")

    from grudge.array_context import get_reasonable_array_context_class
    actx_class = get_reasonable_array_context_class(lazy=lazy, distributed=True)

    restart_filename = None
    if args.restart_file:
        restart_filename = (args.restart_file).replace("'", "")
        print(f"Restarting from file: {restart_filename}")

    target_filename = None
    if args.target_file:
        target_filename = (args.target_file).replace("'", "")
        print(f"Target file specified: {target_filename}")

    input_file = None
    if args.input_file:
        input_file = args.input_file.replace("'", "")
        print(f"Ignoring user input from file: {input_file}")
    else:
        print("No user input file, using default values")

    print(f"Running {sys.argv[0]}\n")
    main(restart_filename=restart_filename, target_filename=target_filename,
         user_input_file=input_file,
         use_profiling=args.profile, use_logmgr=args.log,
         use_overintegration=args.overintegration, lazy=lazy,
         actx_class=actx_class, casename=casename)

# vim: foldmethod=marker
