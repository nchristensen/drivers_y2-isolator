"""process a mirgecom restart to find global minimum and maximum values
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
from pytools.obj_array import make_obj_array
from functools import partial

from grudge.array_context import PyOpenCLArrayContext
from arraycontext import thaw, freeze, flatten, unflatten, to_numpy, from_numpy
from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
from grudge.eager import EagerDGDiscretization
from grudge.shortcuts import make_visualizer
from grudge.dof_desc import DTAG_BOUNDARY
from grudge.op import nodal_max, nodal_min
from logpyle import IntervalTimer, set_dt
from mirgecom.euler import extract_vars_for_logging, units_for_logging
from mirgecom.logging_quantities import (
    initialize_logmgr,
    logmgr_add_many_discretization_quantities,
    logmgr_add_cl_device_info,
    logmgr_set_time,
    LogUserQuantity,
    set_sim_state
)

from mirgecom.navierstokes import ns_operator
from mirgecom.artificial_viscosity import \
    av_laplacian_operator, smoothness_indicator
from mirgecom.simutil import (
    check_step,
    generate_and_distribute_mesh,
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

from mirgecom.fluid import make_conserved
from mirgecom.steppers import advance_state
from mirgecom.boundary import (
    PrescribedFluidBoundary,
    IsothermalNoSlipBoundary,
)
#from mirgecom.initializers import (Uniform, PlanarDiscontinuity)
from mirgecom.eos import IdealSingleGas
from mirgecom.transport import SimpleTransport
from mirgecom.gas_model import GasModel, make_fluid_state


class InitACTII:
    r"""Solution initializer for flow in the ACT-II facility

    This initializer creates a physics-consistent flow solution
    given the top and bottom geometry profiles and an EOS using isentropic
    flow relations.

    The flow is initialized from the inlet stagnations pressure, P0, and
    stagnation temperature T0.

    geometry locations are linearly interpolated between given data points

    .. automethod:: __init__
    .. automethod:: __call__
    """

    def __init__(
            self, *, dim=2, nspecies=0, geom_top, geom_bottom,
            P0, T0, temp_wall, temp_sigma, vel_sigma,
            mass_frac=None,
            inj_pres, inj_temp, inj_vel, inj_mass_frac=None,
            inj_temp_sigma, inj_vel_sigma,
            inj_ytop, inj_ybottom
    ):
        r"""Initialize mixture parameters.

        Parameters
        ----------
        dim: int
            specifies the number of dimensions for the solution
        P0: float
            stagnation pressure
        T0: float
            stagnation temperature
        temp_wall: float
            wall temperature
        temp_sigma: float
            near-wall temperature relaxation parameter
        vel_sigma: float
            near-wall velocity relaxation parameter
        geom_top: numpy.ndarray
            coordinates for the top wall
        geom_bottom: numpy.ndarray
            coordinates for the bottom wall
        """

        # check number of points in the geometry
        #top_size = geom_top.size
        #bottom_size = geom_bottom.size

        if mass_frac is None:
            if nspecies > 0:
                mass_frac = np.zeros(shape=(nspecies,))

        if inj_mass_frac is None:
            if nspecies > 0:
                inj_mass_frac = np.zeros(shape=(nspecies,))

        if inj_vel is None:
            inj_vel = np.zeros(shape=(dim,))

        self._dim = dim
        self._nspecies = nspecies
        self._P0 = P0
        self._T0 = T0
        self._geom_top = geom_top
        self._geom_bottom = geom_bottom
        self._temp_wall = temp_wall
        self._temp_sigma = temp_sigma
        self._vel_sigma = vel_sigma
        # TODO, calculate these from the geometry files
        self._throat_height = 3.61909e-3
        self._x_throat = 0.283718298
        self._mass_frac = mass_frac

        self._inj_P0 = inj_pres
        self._inj_T0 = inj_temp
        self._inj_vel = inj_vel

        self._temp_sigma_injection = inj_temp_sigma
        self._vel_sigma_injection = inj_vel_sigma
        self._inj_mass_frac = inj_mass_frac
        self._inj_ytop = inj_ytop
        self._inj_ybottom = inj_ybottom

    def __call__(self, discr, x_vec, eos, *, time=0.0):
        """Create the solution state at locations *x_vec*.

        Parameters
        ----------
        x_vec: numpy.ndarray
            Coordinates at which solution is desired
        eos:
            Mixture-compatible equation-of-state object must provide
            these functions:
            `eos.get_density`
            `eos.get_internal_energy`
        time: float
            Time at which solution is desired. The location is (optionally)
            dependent on time
        """
        if x_vec.shape != (self._dim,):
            raise ValueError(f"Position vector has unexpected dimensionality,"
                             f" expected {self._dim}.")

        xpos = x_vec[0]
        ypos = x_vec[1]
        if self._dim == 3:
            zpos = x_vec[2]
        ytop = 0*x_vec[0]
        actx = xpos.array_context
        zeros = 0*xpos
        ones = zeros + 1.0

        xpos_flat = to_numpy(flatten(xpos, actx), actx)
        ypos_flat = to_numpy(flatten(ypos, actx), actx)
        gamma = eos.gamma()
        gas_const = eos.gas_const()

        ytop_flat = 0*xpos_flat
        ybottom_flat = 0*xpos_flat
        theta_top_flat = 0*xpos_flat
        theta_bottom_flat = 0*xpos_flat
        mach_flat = 0*xpos_flat
        theta_flat = 0*xpos_flat
        throat_height = 1

        theta_geom_top = get_theta_from_data(self._geom_top)
        theta_geom_bottom = get_theta_from_data(self._geom_bottom)

        for inode in range(xpos_flat.size):
            ytop_flat[inode] = get_y_from_x(xpos_flat[inode], self._geom_top)
            ybottom_flat[inode] = get_y_from_x(xpos_flat[inode], self._geom_bottom)
            theta_top_flat[inode] = get_y_from_x(xpos_flat[inode], theta_geom_top)
            theta_bottom_flat[inode] = get_y_from_x(xpos_flat[inode],
                                                    theta_geom_bottom)
            if ytop_flat[inode] - ybottom_flat[inode] < throat_height:
                throat_height = ytop_flat[inode] - ybottom_flat[inode]
                throat_loc = xpos_flat[inode]
            # temporary fix for parallel, needs to be reduced across partitions
            throat_height = self._throat_height
            throat_loc = self._x_throat

        #print(f"throat height {throat_height}")
        for inode in range(xpos_flat.size):
            area_ratio = (ytop_flat[inode] - ybottom_flat[inode])/throat_height
            theta_flat[inode] = (theta_bottom_flat[inode] +
                          (theta_top_flat[inode]-theta_bottom_flat[inode]) /
                          (ytop_flat[inode]-ybottom_flat[inode]) *
                          (ypos_flat[inode] - ybottom_flat[inode]))
            if xpos_flat[inode] < throat_loc:
                mach_flat[inode] = getMachFromAreaRatio(area_ratio=area_ratio,
                                                        gamma=gamma, mach_guess=0.01)
            elif xpos_flat[inode] > throat_loc:
                mach_flat[inode] = getMachFromAreaRatio(area_ratio=area_ratio,
                                                        gamma=gamma, mach_guess=1.01)
            else:
                mach_flat[inode] = 1.0

        ytop = unflatten(xpos, from_numpy(ytop_flat, actx), actx)
        ybottom = unflatten(xpos, from_numpy(ybottom_flat, actx), actx)
        mach = unflatten(xpos, from_numpy(mach_flat, actx), actx)
        theta = unflatten(xpos, from_numpy(theta_flat, actx), actx)

        pressure = getIsentropicPressure(
            mach=mach,
            P0=self._P0,
            gamma=gamma
        )
        temperature = getIsentropicTemperature(
            mach=mach,
            T0=self._T0,
            gamma=gamma
        )

        # save the unsmoothed temerature, so we can use it with the injector init
        unsmoothed_temperature = temperature

        # modify the temperature in the near wall region to match the
        # isothermal boundaries
        sigma = self._temp_sigma
        wall_temperature = self._temp_wall
        smoothing_top = actx.np.tanh(sigma*(actx.np.abs(ypos-ytop)))
        smoothing_bottom = actx.np.tanh(sigma*(actx.np.abs(ypos-ybottom)))

        smooth_temperature = (wall_temperature +
            (temperature - wall_temperature)*smoothing_top*smoothing_bottom)

        # make a little region along the top of the cavity where we don't want
        # the temperature smoothed
        xc_left = zeros + 0.65163 + 0.0004
        xc_right = zeros + 0.72163 - 0.0004
        yc_top = zeros - 0.006
        yc_bottom = zeros - 0.01

        left_edge = actx.np.greater(xpos, xc_left)
        right_edge = actx.np.less(xpos, xc_right)
        top_edge = actx.np.less(ypos, yc_top)
        bottom_edge = actx.np.greater(ypos, yc_bottom)
        inside_block = left_edge*right_edge*top_edge*bottom_edge
        temperature = actx.np.where(inside_block, temperature, smooth_temperature)

        # smooth on fore and aft boundaries if 3D
        smoothing_fore = ones
        smoothing_aft = ones
        z0 = 0.
        z1 = 0.035
        if self._dim == 3:
            smoothing_fore = actx.np.tanh(sigma*(actx.np.abs(zpos-z0)))
            smoothing_aft = actx.np.tanh(sigma*(actx.np.abs(zpos-z1)))
        temperature = (wall_temperature +
            (temperature - wall_temperature)*smoothing_fore*smoothing_aft)

        y = make_obj_array([self._mass_frac[i] * ones
                            for i in range(self._nspecies)])

        #mass = eos.get_density(pressure, temperature, y)
        mass = pressure/temperature/gas_const
        velocity = np.zeros(self._dim, dtype=object)
        mom = mass*velocity
        #energy = mass*eos.get_internal_energy(temperature, y)
        energy = pressure/(gamma - 1)
        cv = make_conserved(dim=self._dim, mass=mass, momentum=mom, energy=energy,
                            species_mass=mass*y)
        velocity[0] = mach*eos.sound_speed(cv, temperature)

        # modify the velocity in the near-wall region to have a tanh profile
        # this approximates the BL velocity profile
        sigma = self._vel_sigma
        smoothing_top = actx.np.tanh(sigma*(actx.np.abs(ypos-ytop)))
        smoothing_bottom = actx.np.tanh(sigma*(actx.np.abs(ypos-ybottom)))
        smoothing_fore = ones
        smoothing_aft = ones
        if self._dim == 3:
            smoothing_fore = actx.np.tanh(sigma*(actx.np.abs(zpos-z0)))
            smoothing_aft = actx.np.tanh(sigma*(actx.np.abs(zpos-z1)))
        velocity[0] = (velocity[0]*smoothing_top*smoothing_bottom *
                       smoothing_fore*smoothing_aft)

        # split into x and y components
        velocity[1] = velocity[0]*actx.np.sin(theta)
        velocity[0] = velocity[0]*actx.np.cos(theta)

        # zero out the velocity in the cavity region, let the flow develop naturally
        # initially in pressure/temperature equilibrium with the exterior flow
        zeros = 0*xpos
        xc_left = zeros + 0.65163 - 0.000001
        xc_right = zeros + 0.72163 + 0.000001
        yc_top = zeros - 0.0083245
        yc_bottom = zeros - 0.0283245
        xc_bottom = zeros + 0.70163
        wall_theta = np.sqrt(2)/2.

        left_edge = actx.np.greater(xpos, xc_left)
        right_edge = actx.np.less(xpos, xc_right)
        top_edge = actx.np.less(ypos, yc_top)
        inside_cavity = left_edge*right_edge*top_edge

        # smooth the temperature at the cavity walls
        sigma = self._temp_sigma
        smoothing_front = actx.np.tanh(sigma*(actx.np.abs(xpos-xc_left)))
        smoothing_bottom = actx.np.tanh(sigma*(actx.np.abs(ypos-yc_bottom)))
        wall_dist = (wall_theta*(ypos - yc_bottom) -
                     wall_theta*(xpos - xc_bottom))
        smoothing_slant = actx.np.tanh(sigma*(actx.np.abs(wall_dist)))
        cavity_temperature = (wall_temperature +
            (temperature - wall_temperature) *
             smoothing_front*smoothing_bottom*smoothing_slant)
        temperature = actx.np.where(inside_cavity, cavity_temperature, temperature)

        #mass = eos.get_density(pressure, temperature, y)
        mass = pressure/temperature/gas_const
        velocity = np.zeros(self._dim, dtype=object)
        mom = mass*velocity
        #energy = mass*eos.get_internal_energy(temperature, y)
        energy = pressure/(gamma - 1)
        cv = make_conserved(dim=self._dim, mass=mass, momentum=mom, energy=energy,
                            species_mass=mass*y)
        velocity[0] = mach*eos.sound_speed(cv, temperature)

        # zero of the velocity
        velocity[0] = actx.np.where(inside_cavity, zeros, velocity[0])

        # fuel stream initialization
        # initially in pressure/temperature equilibrium with the cavity
        #inj_left = 0.71
        #inj_left = 0.704
        # even with the top corner
        inj_left = 0.7074
        #inj_left = 0.65
        inj_right = 0.73
        inj_top = -0.0226
        inj_bottom = -0.025
        inj_fore = 0.035/2. + 1.59e-3
        inj_aft = 0.035/2. - 1.59e-3
        xc_left = zeros + inj_left
        xc_right = zeros + inj_right
        yc_top = zeros + inj_top
        yc_bottom = zeros + inj_bottom
        zc_fore = zeros + inj_fore
        zc_aft = zeros + inj_aft

        yc_center = zeros - 0.0283245 + 4e-3 + 1.59e-3/2.
        zc_center = zeros + 0.035/2.
        inj_radius = 1.59e-3/2.

        if self._dim == 3:
            radius = actx.np.sqrt((ypos - yc_center)**2 + (zpos - zc_center)**2)

        left_edge = actx.np.greater(xpos, xc_left)
        right_edge = actx.np.less(xpos, xc_right)
        bottom_edge = actx.np.greater(ypos, yc_bottom)
        top_edge = actx.np.less(ypos, yc_top)
        fore_edge = ones
        fore_edge = ones
        if self._dim == 3:
            aft_edge = actx.np.greater(zpos, zc_aft)
            fore_edge = actx.np.less(zpos, zc_fore)
        inside_injector = (left_edge*right_edge*top_edge*bottom_edge *
                           aft_edge*fore_edge)

        inj_y = make_obj_array([self._inj_mass_frac[i] * ones
                            for i in range(self._nspecies)])

        inj_velocity = mach*np.zeros(self._dim, dtype=object)
        inj_velocity[0] = self._inj_vel[0]

        inj_mach = mach*0. + 1.0

        # smooth out the injection profile
        # relax to the cavity temperature/pressure/velocity
        inj_x0 = 0.712
        # the entrace to the injector
        #inj_fuel_x0 = 0.7085
        # back inside the injector
        inj_fuel_x0 = 0.717
        # out in the cavity
        #inj_fuel_x0 = 0.7
        inj_fuel_y0 = -0.0243245 - 3.e-3
        inj_fuel_y1 = -0.0227345 + 3.e-3
        inj_fuel_z0 = 0.035/2. - 3.e-3
        inj_fuel_z1 = 0.035/2. + 3.e-3
        inj_sigma = 1500
        gamma_guess_inj = gamma

        # seperate the fuel from the flow, allow the fuel to spill out into the
        # cavity ahead of hte injection flow, see if this helps startup
        # left extent
        inj_tanh = inj_sigma*(inj_fuel_x0 - xpos)
        inj_weight = 0.5*(1.0 - actx.np.tanh(inj_tanh))
        for i in range(self._nspecies):
            inj_y[i] = y[i] + (inj_y[i] - y[i])*inj_weight

        # bottom extent
        inj_tanh = inj_sigma*(inj_fuel_y0 - ypos)
        inj_weight = 0.5*(1.0 - actx.np.tanh(inj_tanh))
        for i in range(self._nspecies):
            inj_y[i] = y[i] + (inj_y[i] - y[i])*inj_weight

        # top extent
        inj_tanh = inj_sigma*(ypos - inj_fuel_y1)
        inj_weight = 0.5*(1.0 - actx.np.tanh(inj_tanh))
        for i in range(self._nspecies):
            inj_y[i] = y[i] + (inj_y[i] - y[i])*inj_weight

        if self._dim == 3:
            # aft extent
            inj_tanh = inj_sigma*(inj_fuel_z0 - zpos)
            inj_weight = 0.5*(1.0 - actx.np.tanh(inj_tanh))
            for i in range(self._nspecies):
                inj_y[i] = y[i] + (inj_y[i] - y[i])*inj_weight

            # fore extent
            inj_tanh = inj_sigma*(zpos - inj_fuel_z1)
            inj_weight = 0.5*(1.0 - actx.np.tanh(inj_tanh))
            for i in range(self._nspecies):
                inj_y[i] = y[i] + (inj_y[i] - y[i])*inj_weight

        # transition the mach number from 0 (cavitiy) to 1 (injection)
        inj_tanh = inj_sigma*(inj_x0 - xpos)
        inj_weight = 0.5*(1.0 - actx.np.tanh(inj_tanh))
        inj_mach = inj_weight
        # assume a smooth transition in gamma, could calculate it
        #inj_gamma = gamma_guess + (gamma_guess_inj - gamma_guess)*inj_weight
        inj_gamma = gamma

        inj_pressure = getIsentropicPressure(
            mach=inj_mach,
            P0=self._inj_P0,
            gamma=inj_gamma
        )
        inj_temperature = getIsentropicTemperature(
            mach=inj_mach,
            T0=self._inj_T0,
            gamma=inj_gamma
        )

        #inj_mass = eos.get_density(inj_pressure, inj_temperature, inj_y)
        inj_mass = inj_pressure/inj_temperature/gas_const
        inj_velocity = mach*np.zeros(self._dim, dtype=object)
        inj_mom = inj_mass*inj_velocity
        #inj_energy = inj_mass*eos.get_internal_energy(inj_temperature, inj_y)
        inj_energy = inj_pressure/(gamma - 1)
        #print(f"energy {energy}")

        # the velocity magnitude
        inj_cv = make_conserved(dim=self._dim, mass=inj_mass, momentum=inj_mom,
                                energy=inj_energy, species_mass=inj_mass*inj_y)

        #sos = eos.sound_speed(cv)
        #print(f"sos {sos}")
        inj_velocity[0] = -inj_mach*eos.sound_speed(inj_cv, inj_temperature)

        # relax the pressure at the cavity/injector interface
        inj_pressure = pressure + (inj_pressure - pressure)*inj_weight
        inj_temperature = (unsmoothed_temperature +
            (inj_temperature - unsmoothed_temperature)*inj_weight)

        # we need to calculate the velocity from a prescribed mass flow rate
        # this will need to take into account the velocity relaxation at the
        # injector walls
        #inj_velocity[0] = velocity[0] + (self._inj_vel[0] - velocity[0])*inj_weight

        # modify the temperature in the near wall region to match the
        # isothermal boundaries
        sigma = self._temp_sigma_injection
        wall_temperature = self._temp_wall
        smoothing_top = actx.np.tanh(sigma*(actx.np.abs(ypos-self._inj_ytop)))
        smoothing_bottom = actx.np.tanh(sigma*(actx.np.abs(ypos-self._inj_ybottom)))

        if self._dim == 2:
            inj_temperature = (wall_temperature +
                (inj_temperature - wall_temperature)*smoothing_top*smoothing_bottom)
        else:
            smoothing_radius = actx.np.tanh(sigma*(actx.np.abs(radius - inj_radius)))
            inj_temperature = (wall_temperature +
                (inj_temperature - wall_temperature)*smoothing_radius)

        # compute the density and then energy from the pressure/temperature state
        #inj_mass = eos.get_density(inj_pressure, inj_temperature, inj_y)
        inj_mass = inj_pressure/inj_temperature/gas_const
        #inj_energy = inj_mass*eos.get_internal_energy(inj_temperature, inj_y)
        inj_energy = inj_pressure/(gamma - 1)

        # modify the velocity in the near-wall region to have a tanh profile
        # this approximates the BL velocity profile
        sigma = self._vel_sigma_injection
        smoothing_top = actx.np.tanh(sigma*(actx.np.abs(ypos-self._inj_ytop)))
        smoothing_bottom = actx.np.tanh(sigma*(actx.np.abs(ypos-self._inj_ybottom)))
        if self._dim == 2:
            inj_velocity[0] = inj_velocity[0]*smoothing_top*smoothing_bottom
        else:
            smoothing_radius = actx.np.tanh(sigma*(actx.np.abs(radius - inj_radius)))
            inj_velocity[0] = inj_velocity[0]*smoothing_radius

        # use the species field with fuel added everywhere
        for i in range(self._nspecies):
            #y[i] = actx.np.where(inside_injector, inj_y[i], y[i])
            y[i] = inj_y[i]

        # recompute the mass and energy (outside the injector) to account for
        # the change in mass fraction
        #mass = eos.get_density(pressure, temperature, y)
        mass = pressure/temperature/gas_const
        #energy = mass*eos.get_internal_energy(temperature, y)
        energy = pressure/(gamma - 1)

        mass = actx.np.where(inside_injector, inj_mass, mass)
        velocity[0] = actx.np.where(inside_injector, inj_velocity[0], velocity[0])
        energy = actx.np.where(inside_injector, inj_energy, energy)
        mom = mass*velocity
        energy = (energy + np.dot(mom, mom)/(2.0*mass))
        return make_conserved(
            dim=self._dim,
            mass=mass,
            momentum=mom,
            energy=energy,
            species_mass=mass*y
        )


@mpi_entry_point
def main(ctx_factory=cl.create_some_context, restart_filename=None,
         actx_class=PyOpenCLArrayContext, casename=None):
    """Drive the Y0 example."""
    cl_ctx = ctx_factory()

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nparts = comm.Get_size()

    if casename is None:
        casename = "mirgecom"

    queue = cl.CommandQueue(cl_ctx)

    actx = actx_class(
        queue,
        allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue)))

    # working gas: O2/N2 #
    #   O2 mass fraction 0.273
    #   gamma = 1.4
    #   cp = 37.135 J/mol-K,
    #   rho= 1.977 kg/m^3 @298K
    gamma = 1.4
    mw_o2 = 15.999*2
    mw_n2 = 14.0067*2
    mf_o2 = 0.273
    # visocsity @ 400C, Pa-s
    mu_o2 = 3.76e-5
    mu_n2 = 3.19e-5
    mu_mix = mu_o2*mf_o2 + mu_n2*(1-mu_o2)  # 3.3456e-5
    mw = mw_o2*mf_o2 + mw_n2*(1.0 - mf_o2)
    r = 8314.59/mw
    cp = r*gamma/(gamma - 1)
    Pr = 0.75
    nspecies = 2

    mu = mu_mix
    kappa = cp*mu/Pr

    if rank == 0:
        print("\n#### Simluation material properties: ####")
        print(f"\tmu = {mu}")
        print(f"\tkappa = {kappa}")
        print(f"\tPrandtl Number  = {Pr}")

    spec_diffusivity = 0. * np.ones(nspecies)
    transport_model = SimpleTransport(viscosity=mu, thermal_conductivity=kappa,
                                      species_diffusivity=spec_diffusivity)

    eos = IdealSingleGas(gamma=gamma, gas_const=r)
    gas_model = GasModel(eos=eos, transport=transport_model)

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

        assert restart_data["nparts"] == nparts
    else:
        error_message = "Restart file not specified"
        raise RuntimeError(error_message)

    order = restart_order

    if restart_filename:
        if rank == 0:
            logging.info("Restarting soln.")
        current_cv = restart_data["cv"]

    discr = EagerDGDiscretization(
        actx, local_mesh, order=order, mpi_communicator=comm
    )

    fluid_state = make_fluid_state(cv=current_cv, gas_model=gas_model)
    cv = fluid_state.cv
    dv = fluid_state.dv

    nodes = thaw(discr.nodes(), actx)

    xpos = nodes[0]
    ypos = nodes[1]
    zpos = nodes[2]

    xpos_flat = to_numpy(flatten(xpos, actx), actx)
    ypos_flat = to_numpy(flatten(ypos, actx), actx)
    zpos_flat = to_numpy(flatten(zpos, actx), actx)

    print(f"{xpos_flat.size}")

    mass = cv.mass
    mass_flat = to_numpy(flatten(mass, actx), actx)

    print(f"{mass_flat.size}")

    # find minimum location for each piece of cv
    mass_max_val = -1e99
    mass_max_loc = [0, 1, 2]
    mass_min_val = 1e99
    mass_min_loc = [0, 1, 2]
    for inode in range(xpos_flat.size):
        if mass_flat[inode] > mass_max_val:
            mass_max_val = mass_flat[inode]
            mass_max_loc = [xpos_flat[inode],
                            ypos_flat[inode],
                            zpos_flat[inode]]
        if mass_flat[inode] < mass_min_val:
            mass_min_val = mass_flat[inode]
            mass_min_loc = [xpos_flat[inode],
                            ypos_flat[inode],
                            zpos_flat[inode]]

    print(f"Minimum density {mass_min_val} at {mass_min_loc}")
    print(f"Maximum density {mass_max_val} at {mass_max_loc}")

    pressure = dv.pressure
    pressure_flat = to_numpy(flatten(pressure, actx), actx)

    print(f"{pressure_flat.size}")

    # find minimum location for each piece of dv
    pressure_max_val = -1e99
    pressure_max_loc = [0, 1, 2]
    pressure_min_val = 1e99
    pressure_min_loc = [0, 1, 2]
    for inode in range(xpos_flat.size):
        if pressure_flat[inode] > pressure_max_val:
            pressure_max_val = pressure_flat[inode]
            pressure_max_loc = [xpos_flat[inode],
                            ypos_flat[inode],
                            zpos_flat[inode]]
        if pressure_flat[inode] < pressure_min_val:
            pressure_min_val = pressure_flat[inode]
            pressure_min_loc = [xpos_flat[inode],
                            ypos_flat[inode],
                            zpos_flat[inode]]

    print(f"Minimum pressure {pressure_min_val} at {pressure_min_loc}")
    print(f"Maximum pressure {pressure_max_val} at {pressure_max_loc}")

    temperature = dv.temperature
    temperature_flat = to_numpy(flatten(temperature, actx), actx)

    print(f"{temperature_flat.size}")

    # find minimum location for each piece of dv
    temperature_max_val = -1e99
    temperature_max_loc = [0, 1, 2]
    temperature_min_val = 1e99
    temperature_min_loc = [0, 1, 2]
    for inode in range(xpos_flat.size):
        if temperature_flat[inode] > temperature_max_val:
            temperature_max_val = temperature_flat[inode]
            temperature_max_loc = [xpos_flat[inode],
                            ypos_flat[inode],
                            zpos_flat[inode]]
        if temperature_flat[inode] < temperature_min_val:
            temperature_min_val = temperature_flat[inode]
            temperature_min_loc = [xpos_flat[inode],
                            ypos_flat[inode],
                            zpos_flat[inode]]

    print(f"Minimum temperature {temperature_min_val} at {temperature_min_loc}")
    print(f"Maximum temperature {temperature_max_val} at {temperature_max_loc}")


if __name__ == "__main__":
    import sys

    logging.basicConfig(format="%(message)s", level=logging.INFO)

    import argparse
    parser = argparse.ArgumentParser(
        description="MIRGE-Com restart file analyzer")
    parser.add_argument("-r", "--restart_file", type=ascii, dest="restart_file",
                        nargs="?", action="store", help="simulation restart file")
    parser.add_argument("-c", "--casename", type=ascii, dest="casename", nargs="?",
                        action="store", help="simulation case name")

    args = parser.parse_args()

    casename = "isolator"
    if args.casename:
        print(f"Custom casename {args.casename}")
        casename = args.casename.replace("'", "")
    else:
        print(f"Default casename {casename}")

    actx_class = PyOpenCLArrayContext

    restart_filename = None
    if args.restart_file:
        restart_filename = (args.restart_file).replace("'", "")
        print(f"Working with restart file: {restart_filename}")

    print(f"Running {sys.argv[0]}\n")
    main(restart_filename=restart_filename, actx_class=actx_class, casename=casename)

