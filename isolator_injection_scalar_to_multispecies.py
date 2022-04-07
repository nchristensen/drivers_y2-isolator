"""mirgecom driver for the Y0 demonstration.

Note: this example requires a *scaled* version of the Y0
grid. A working grid example is located here:
github.com:/illinois-ceesd/data@y0scaled
"""

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
import math
from functools import partial

from arraycontext import thaw
from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
from grudge.eager import EagerDGDiscretization
from grudge.shortcuts import make_visualizer

from mirgecom.simutil import (
    generate_and_distribute_mesh,
    write_visfile,
)
from mirgecom.restart import write_restart_file
from mirgecom.mpi import mpi_entry_point
import pyopencl.tools as cl_tools

from mirgecom.fluid import make_conserved
import cantera
from mirgecom.eos import IdealSingleGas, PyrometheusMixture
from mirgecom.transport import SimpleTransport
from mirgecom.gas_model import GasModel, make_fluid_state


class SingleLevelFilter(logging.Filter):
    def __init__(self, passlevel, reject):
        self.passlevel = passlevel
        self.reject = reject

    def filter(self, record):
        if self.reject:
            return (record.levelno != self.passlevel)
        else:
            return (record.levelno == self.passlevel)


class MyRuntimeError(RuntimeError):
    """Simple exception to kill the simulation."""

    pass


@mpi_entry_point
def main(ctx_factory=cl.create_some_context, restart_filename=None,
         user_input_file=None,
         use_overintegration=False, actx_class=None, casename=None,
         lazy=False):

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

    #from mirgecom.simutil import global_reduce as _global_reduce
    #global_reduce = partial(_global_reduce, comm=comm)

    if casename is None:
        casename = "mirgecom"

    queue = cl.CommandQueue(cl_ctx)

    # main array context for the simulation
    if lazy:
        actx = actx_class(comm, queue, mpi_base_tag=12000)
    else:
        actx = actx_class(comm, queue,
                allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue)),
                force_device_scalars=True)

    # discretization and model control
    order = 1
    dim = 2

    # material properties
    nspecies_scalar = 2
    nspecies = 0

    if user_input_file:
        input_data = None
        if rank == 0:
            with open(user_input_file) as f:
                input_data = yaml.load(f, Loader=yaml.FullLoader)
        input_data = comm.bcast(input_data, root=0)
        try:
            order = int(input_data["order"])
        except KeyError:
            pass
        try:
            dim = int(input_data["dimen"])
        except KeyError:
            pass
        try:
            nspecies = int(input_data["nspecies"])
        except KeyError:
            pass

    if rank == 0:
        print("\n#### Simluation control data: ####")
        print(f"\torder = {order}")
        print(f"\tdimen = {dim}")
        print("#### Simluation control data: ####")

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
    mf_c2h4 = 0.5
    mf_h2 = 0.5
    # visocsity @ 400C, Pa-s
    mu_o2 = 3.76e-5
    mu_n2 = 3.19e-5
    mu_mix = mu_o2*mf_o2 + mu_n2*(1-mu_o2)  # 3.3456e-5
    mu = mu_mix
    mw = mw_o2*mf_o2 + mw_n2*(1.0 - mf_o2)
    r = 8314.59/mw
    cp = r*gamma/(gamma - 1)
    Pr = 0.75
    kappa = cp*mu_mix/Pr
    init_temperature = 300.0

    if rank == 0:
        print("\n#### Simluation material properties: ####")
        print(f"\tmu = {mu}")
        print(f"\tkappa = {kappa}")
        print(f"\tPrandtl Number  = {Pr}")
        print(f"\tnspecies = {nspecies}")
        print("#### Simluation material properties: ####")

    spec_diffusivity = 1.e-4 * np.ones(nspecies)
    transport_model = SimpleTransport(viscosity=mu, thermal_conductivity=kappa,
                                      species_diffusivity=spec_diffusivity)

    # initialize eos and species mass fractions
    y = np.zeros(nspecies)
    from mirgecom.mechanisms import get_mechanism_cti
    mech_cti = get_mechanism_cti("uiuc")

    cantera_soln = cantera.Solution(phase_id="gas", source=mech_cti)
    cantera_nspecies = cantera_soln.n_species
    if nspecies != cantera_nspecies:
        if rank == 0:
            print(f"specified {nspecies=}, but cantera mechanism"
                  f" needs nspecies={cantera_nspecies}")
        raise RuntimeError()

    i_c2h4 = cantera_soln.species_index("C2H4")
    i_h2 = cantera_soln.species_index("H2")
    i_ox = cantera_soln.species_index("O2")
    i_di = cantera_soln.species_index("N2")
    # Set the species mass fractions to the free-stream flow
    y[i_ox] = mf_o2
    y[i_di] = 1. - mf_o2

    cantera_soln.TPY = init_temperature, 101325, y

    # make the eos
    eos_scalar = IdealSingleGas(gamma=gamma, gas_const=r)
    from mirgecom.thermochemistry import make_pyrometheus_mechanism_class
    pyro_mech = make_pyrometheus_mechanism_class(cantera_soln)(actx.np)
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
        assert restart_data["nspecies"] == nspecies_scalar
    else:
        error_message = "Driver only supports restart. Start with -r <filename>"
        raise RuntimeError(error_message)

    if rank == 0:
        logging.info("Making discretization")

    discr = EagerDGDiscretization(actx, local_mesh, order, mpi_communicator=comm)

    if rank == 0:
        logging.info("Done making discretization")

    if rank == 0:
        logging.info("Initializing solution")

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

    else:
        error_message = "Driver only supports restart. Start with -r <filename>"
        raise RuntimeError(error_message)

    mass = restart_cv.mass
    velocity = restart_cv.momentum/mass
    species_mass_frac = restart_cv.species_mass/mass
    species_mass_frac_multi = 0.*mass*y

    pressure = eos_scalar.pressure(restart_cv)
    temperature = eos_scalar.temperature(restart_cv, temperature_seed)

    # air is species 0 in scalar sim
    species_mass_frac_multi[i_ox] = mf_o2*species_mass_frac[0]
    species_mass_frac_multi[i_di] = (1. - mf_o2)*species_mass_frac[0]

    # fuel is speices 1 in scalar sim
    species_mass_frac_multi[i_c2h4] = mf_c2h4*species_mass_frac[1]
    species_mass_frac_multi[i_h2] = mf_h2*species_mass_frac[1]

    internal_energy = eos.get_internal_energy(temperature=temperature,
        species_mass_fractions=species_mass_frac_multi)

    modified_mass = eos.get_density(pressure, temperature, species_mass_frac_multi)

    total_energy = modified_mass*(internal_energy + np.dot(velocity, velocity)/(2.0))

    modified_cv = make_conserved(dim,
                                 mass=modified_mass,
                                 momentum=modified_mass*velocity,
                                 energy=total_energy,
                                 species_mass=modified_mass*species_mass_frac_multi)

    current_state = make_fluid_state(modified_cv, gas_model, temperature)

    visualizer = make_visualizer(discr)

    def my_write_viz(step, t, cv, dv):

        mach = (actx.np.sqrt(np.dot(cv.velocity, cv.velocity)) /
                            dv.speed_of_sound)
        viz_fields = [("cv", cv),
                      ("dv", dv),
                      ("mach", mach),
                      ("rank", rank),
                      ("velocity", cv.velocity)]
        # species mass fractions
        viz_fields.extend(
            ("Y_"+species_names[i], cv.species_mass_fractions[i])
            for i in range(nspecies))
        write_visfile(discr, viz_fields, visualizer, vizname=vizname,
                      step=step, t=t, overwrite=True)

    def my_write_restart(step, t, cv, temperature_seed):
        restart_fname = restart_pattern.format(cname=casename, step=step, rank=rank)
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

    # write visualization and restart data
    my_write_viz(step=current_step, t=current_t, 
                 cv=current_state.cv, dv=current_state.dv)
    my_write_restart(step=current_step, t=current_t, cv=current_state.cv,
                     temperature_seed=current_state.dv.temperature)


if __name__ == "__main__":

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO)

    import argparse
    parser = argparse.ArgumentParser(
        description="MIRGE-Com Isentropic Nozzle Driver")
    parser.add_argument("-r", "--restart_file", type=ascii, dest="restart_file",
                        nargs="?", action="store", help="simulation restart file")
    parser.add_argument("-i", "--input_file", type=ascii, dest="input_file",
                        nargs="?", action="store", help="simulation config file")
    parser.add_argument("-c", "--casename", type=ascii, dest="casename", nargs="?",
                        action="store", help="simulation case name")
    parser.add_argument("--lazy", action="store_true", default=False,
                        help="enable lazy evaluation [OFF]")

    args = parser.parse_args()
    lazy = args.lazy

    # for writing output
    casename = "isolator_transfer"
    if args.casename:
        print(f"Custom casename {args.casename}")
        casename = args.casename.replace("'", "")
    else:
        print(f"Default casename {casename}")

    from grudge.array_context import get_reasonable_array_context_class
    actx_class = get_reasonable_array_context_class(lazy=lazy, distributed=True)

    restart_filename = None
    if args.restart_file:
        restart_filename = (args.restart_file).replace("'", "")
        print(f"Restarting from file: {restart_filename}")

    input_file = None
    if args.input_file:
        input_file = args.input_file.replace("'", "")
        print(f"Reading user input from file: {input_file}")
    else:
        print("No user input file, using default values")

    print(f"Running {sys.argv[0]}\n")
    main(restart_filename=restart_filename, user_input_file=input_file,
         actx_class=actx_class, casename=casename,
         lazy=lazy)

# vim: foldmethod=marker
