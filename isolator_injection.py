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
from pytools.obj_array import make_obj_array
from functools import partial

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


class SingleLevelFilter(logging.Filter):
    def __init__(self, passlevel, reject):
        self.passlevel = passlevel
        self.reject = reject

    def filter(self, record):
        if self.reject:
            return (record.levelno != self.passlevel)
        else:
            return (record.levelno == self.passlevel)


h1 = logging.StreamHandler(sys.stdout)
f1 = SingleLevelFilter(logging.INFO, False)
h1.addFilter(f1)
root_logger = logging.getLogger()
root_logger.addHandler(h1)
h2 = logging.StreamHandler(sys.stderr)
f2 = SingleLevelFilter(logging.INFO, True)
h2.addFilter(f2)
root_logger.addHandler(h2)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
#logger.debug("A DEBUG message")
#logger.info("An INFO message")
#logger.warning("A WARNING message")
#logger.error("An ERROR message")
#logger.critical("A CRITICAL message")


class MyRuntimeError(RuntimeError):
    """Simple exception to kill the simulation."""

    pass


def get_mesh(dim, read_mesh=True):
    """Get the mesh."""
    from meshmode.mesh.io import read_gmsh
    mesh_filename = "data/isolator.msh"
    #mesh = read_gmsh(mesh_filename, force_ambient_dim=dim)
    mesh = partial(read_gmsh, filename=mesh_filename, force_ambient_dim=dim)
    #mesh = read_gmsh(mesh_filename)

    return mesh


def sponge(cv, cv_ref, sigma):
    return sigma*(cv_ref - cv)


class InitSponge:
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


def getIsentropicPressure(mach, P0, gamma):
    pressure = (1. + (gamma - 1.)*0.5*mach**2)
    pressure = P0*pressure**(-gamma / (gamma - 1.))
    return pressure


def getIsentropicTemperature(mach, T0, gamma):
    temperature = (1. + (gamma - 1.)*0.5*mach**2)
    temperature = T0/temperature
    return temperature


def getMachFromAreaRatio(area_ratio, gamma, mach_guess=0.01):
    error = 1.0e-8
    nextError = 1.0e8
    g = gamma
    M0 = mach_guess
    while nextError > error:
        R = (((2/(g + 1) + ((g - 1)/(g + 1)*M0*M0))**(((g + 1)/(2*g - 2))))/M0
            - area_ratio)
        dRdM = (2*((2/(g + 1) + ((g - 1)/(g + 1)*M0*M0))**(((g + 1)/(2*g - 2))))
               / (2*g - 2)*(g - 1)/(2/(g + 1) + ((g - 1)/(g + 1)*M0*M0)) -
               ((2/(g + 1) + ((g - 1)/(g + 1)*M0*M0))**(((g + 1)/(2*g - 2))))
               * M0**(-2))
        M1 = M0 - R/dRdM
        nextError = abs(R)
        M0 = M1

    return M1


def get_y_from_x(x, data):
    """
    Return the linearly interpolated the value of y
    from the value in data(x,y) at x
    """

    if x <= data[0][0]:
        y = data[0][1]
    elif x >= data[-1][0]:
        y = data[-1][1]
    else:
        ileft = 0
        iright = data.shape[0]-1

        # find the bracketing points, simple subdivision search
        while iright - ileft > 1:
            ind = int(ileft+(iright - ileft)/2)
            if x < data[ind][0]:
                iright = ind
            else:
                ileft = ind

        leftx = data[ileft][0]
        rightx = data[iright][0]
        lefty = data[ileft][1]
        righty = data[iright][1]

        dx = rightx - leftx
        dy = righty - lefty
        y = lefty + (x - leftx)*dy/dx
    return y


def get_theta_from_data(data):
    """
    Calculate theta = arctan(dy/dx)
    Where data[][0] = x and data[][1] = y
    """

    theta = data.copy()
    for index in range(1, theta.shape[0]-1):
        #print(f"index {index}")
        theta[index][1] = np.arctan((data[index+1][1]-data[index-1][1]) /
                          (data[index+1][0]-data[index-1][0]))
    theta[0][1] = np.arctan(data[1][1]-data[0][1])/(data[1][0]-data[0][0])
    theta[-1][1] = np.arctan(data[-1][1]-data[-2][1])/(data[-1][0]-data[-2][0])
    return(theta)


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


class UniformModifiedInjector:
    r"""Solution initializer for a uniform flow with boundary layer smoothing.

    Similar to the Uniform initializer, except the velocity profile is modified
    so that the velocity goes to zero at a radius r from a given point xc, yc, zc

    The smoothing comes from a hyperbolic tangent with weight sigma

    .. automethod:: __init__
    .. automethod:: __call__
    """

    def __init__(
            self, *, dim=1, nspecies=0, pressure=1.0, temperature=2.5,
            velocity=None, mass_fracs=None,
            temp_wall, temp_sigma, vel_sigma,
            inj_radius=1., yc=0., zc=0.
    ):
        r"""Initialize uniform flow parameters.

        Parameters
        ----------
        dim: int
            specify the number of dimensions for the flow
        nspecies: int
            specify the number of species in the flow
        temperature: float
            specifies the temperature
        pressure: float
            specifies the pressure
        velocity: numpy.ndarray
            specifies the flow velocity
        temp_wall: float
            wall temperature
        temp_sigma: float
            near-wall temperature relaxation parameter
        vel_sigma: float
            near-wall velocity relaxation parameter
            # update this to be current for whats being implemented here
            # really should just modify the boundary to take a function for smoothing
        """
        if velocity is not None:
            numvel = len(velocity)
            myvel = velocity
            if numvel > dim:
                dim = numvel
            elif numvel < dim:
                myvel = np.zeros(shape=(dim,))
                for i in range(numvel):
                    myvel[i] = velocity[i]
            self._velocity = myvel
        else:
            self._velocity = np.zeros(shape=(dim,))

        if mass_fracs is not None:
            self._nspecies = len(mass_fracs)
            self._mass_fracs = mass_fracs
        else:
            self._nspecies = nspecies
            self._mass_fracs = np.zeros(shape=(nspecies,))

        if self._velocity.shape != (dim,):
            raise ValueError(f"Expected {dim}-dimensional inputs.")

        self._pressure = pressure
        self._temperature = temperature
        self._dim = dim
        self._temp_wall = temp_wall
        self._temp_sigma = temp_sigma
        self._vel_sigma = vel_sigma
        self._yc = yc
        self._zc = zc
        self._inj_radius = inj_radius

    def __call__(self, x_vec, *, eos, **kwargs):
        """
        Create a uniform flow solution at locations *x_vec*.

        Parameters
        ----------
        x_vec: numpy.ndarray
            Nodal coordinates
        eos: :class:`mirgecom.eos.IdealSingleGas`
            Equation of state class with method to supply gas *gamma*.
        """

        ypos = x_vec[1]
        if self._dim == 3:
            zpos = x_vec[2]
        actx = ypos.array_context
        yc = 0.0*x_vec[1] + self._yc
        zc = 0.0*x_vec[2] + self._zc
        inj_radius = 0.0*x_vec[0] + self._inj_radius
        # injector aligned with the x-axis
        radius = actx.np.sqrt((ypos - yc)**2 + (zpos - zc)**2)

        ones = (1.0 + x_vec[0]) - x_vec[0]

        pressure = self._pressure * ones
        temperature = self._temperature * ones

        # modify the temperature in the near wall region to match
        # the isothermal boundaries
        sigma = self._temp_sigma
        wall_temperature = self._temp_wall
        smoothing_radius = actx.np.tanh(sigma*(actx.np.abs(radius - inj_radius)))
        temperature = (wall_temperature +
            (temperature - wall_temperature)*smoothing_radius)

        velocity = make_obj_array([self._velocity[i] * ones
                                   for i in range(self._dim)])
        y = make_obj_array([self._mass_fracs[i] * ones
                            for i in range(self._nspecies)])
        # just comment this out and use single species
        """
        if self._nspecies:
            mass = eos.get_density(pressure, temperature, y)
        else:
            mass = pressure/temperature/eos.gas_const()
        """
        mass = pressure/temperature/eos.gas_const()
        specmass = mass * y

        sigma = self._vel_sigma
        # modify the velocity profile from uniform
        smoothing_radius = actx.np.tanh(sigma*(actx.np.abs(radius - inj_radius)))
        velocity[0] = velocity[0]*smoothing_radius

        mom = mass*velocity
        """
        if self._nspecies:
            internal_energy = eos.get_internal_energy(temperature=temperature,
                                                      species_mass=specmass)
        else:
            internal_energy = pressure/(eos.gamma() - 1)
        """
        internal_energy = pressure/(eos.gamma() - 1)
        kinetic_energy = 0.5 * np.dot(mom, mom)/mass
        energy = internal_energy + kinetic_energy

        return make_conserved(dim=self._dim, mass=mass, energy=energy,
                              momentum=mom, species_mass=specmass)


class UniformModified:
    r"""Solution initializer for a uniform flow with boundary layer smoothing.

    Similar to the Uniform initializer, except the velocity profile is modified
    so that the velocity goes to zero at y(min, max)

    The smoothing comes from a hyperbolic tangent with weight sigma

    .. automethod:: __init__
    .. automethod:: __call__
    """

    def __init__(
            self, *, dim=1, nspecies=0, pressure=1.0, temperature=2.5,
            velocity=None, mass_fracs=None,
            temp_wall, temp_sigma, vel_sigma,
            ymin=0., ymax=1.0
    ):
        r"""Initialize uniform flow parameters.

        Parameters
        ----------
        dim: int
            specify the number of dimensions for the flow
        nspecies: int
            specify the number of species in the flow
        temperature: float
            specifies the temperature
        pressure: float
            specifies the pressure
        velocity: numpy.ndarray
            specifies the flow velocity
        temp_wall: float
            wall temperature
        temp_sigma: float
            near-wall temperature relaxation parameter
        vel_sigma: float
            near-wall velocity relaxation parameter
        ymin: flaot
            minimum y-coordinate for smoothing
        ymax: float
            maximum y-coordinate for smoothing
        """
        if velocity is not None:
            numvel = len(velocity)
            myvel = velocity
            if numvel > dim:
                dim = numvel
            elif numvel < dim:
                myvel = np.zeros(shape=(dim,))
                for i in range(numvel):
                    myvel[i] = velocity[i]
            self._velocity = myvel
        else:
            self._velocity = np.zeros(shape=(dim,))

        if mass_fracs is not None:
            self._nspecies = len(mass_fracs)
            self._mass_fracs = mass_fracs
        else:
            self._nspecies = nspecies
            self._mass_fracs = np.zeros(shape=(nspecies,))

        if self._velocity.shape != (dim,):
            raise ValueError(f"Expected {dim}-dimensional inputs.")

        self._pressure = pressure
        self._temperature = temperature
        self._dim = dim
        self._temp_wall = temp_wall
        self._temp_sigma = temp_sigma
        self._vel_sigma = vel_sigma
        self._ymin = ymin
        self._ymax = ymax

    def __call__(self, x_vec, *, eos, **kwargs):
        """
        Create a uniform flow solution at locations *x_vec*.

        Parameters
        ----------
        x_vec: numpy.ndarray
            Nodal coordinates
        eos: :class:`mirgecom.eos.IdealSingleGas`
            Equation of state class with method to supply gas *gamma*.
        """

        ypos = x_vec[1]
        if self._dim == 3:
            zpos = x_vec[2]
        actx = ypos.array_context
        ymax = 0.0*x_vec[1] + self._ymax
        ymin = 0.0*x_vec[1] + self._ymin
        if self._dim == 3:
            zmax = 0.0*x_vec[2] + 0.035
            zmin = 0.0*x_vec[2]

        ones = (1.0 + x_vec[0]) - x_vec[0]

        pressure = self._pressure * ones
        temperature = self._temperature * ones

        # modify the temperature in the near wall region to match
        # the isothermal boundaries
        sigma = self._temp_sigma
        wall_temperature = self._temp_wall
        smoothing_min = actx.np.tanh(sigma*(actx.np.abs(ypos-ymin)))
        smoothing_max = actx.np.tanh(sigma*(actx.np.abs(ypos-ymax)))
        smoothing_aft = ones
        smoothing_fore = ones
        if self._dim == 3:
            smoothing_aft = actx.np.tanh(sigma*(actx.np.abs(zpos-zmin)))
            smoothing_fore = actx.np.tanh(sigma*(actx.np.abs(zpos-zmax)))
        temperature = (wall_temperature +
                       (temperature - wall_temperature)*smoothing_min*smoothing_max *
                        smoothing_aft*smoothing_fore)

        velocity = make_obj_array([self._velocity[i] * ones
                                   for i in range(self._dim)])
        y = make_obj_array([self._mass_fracs[i] * ones
                            for i in range(self._nspecies)])
        # just comment this out and use single species
        """
        if self._nspecies:
            mass = eos.get_density(pressure, temperature, y)
        else:
            mass = pressure/temperature/eos.gas_const()
        """
        mass = pressure/temperature/eos.gas_const()
        specmass = mass * y

        sigma = self._vel_sigma
        # modify the velocity profile from uniform
        smoothing_max = actx.np.tanh(sigma*(actx.np.abs(ypos-ymax)))
        smoothing_min = actx.np.tanh(sigma*(actx.np.abs(ypos-ymin)))
        if self._dim == 3:
            smoothing_aft = actx.np.tanh(sigma*(actx.np.abs(zpos-zmin)))
            smoothing_fore = actx.np.tanh(sigma*(actx.np.abs(zpos-zmax)))
        velocity[0] = (velocity[0]*smoothing_max*smoothing_min *
                      smoothing_aft*smoothing_fore)

        mom = mass*velocity
        """
        if self._nspecies:
            internal_energy = eos.get_internal_energy(temperature=temperature,
                                                      species_mass=specmass)
        else:
            internal_energy = pressure/(eos.gamma() - 1)
        """
        internal_energy = pressure/(eos.gamma() - 1)
        kinetic_energy = 0.5 * np.dot(mom, mom)/mass
        energy = internal_energy + kinetic_energy

        return make_conserved(dim=self._dim, mass=mass, energy=energy,
                              momentum=mom, species_mass=specmass)


@mpi_entry_point
def main(ctx_factory=cl.create_some_context, restart_filename=None,
         use_profiling=False, use_logmgr=True, user_input_file=None,
         use_overintegration=False, actx_class=None, casename=None,
         lazy=False):
    """Drive the Y0 example."""
    if actx_class is None:
        raise RuntimeError("Array context class missing.")

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
    logmgr = initialize_logmgr(use_logmgr,
        filename=f"{casename}.sqlite", mode="wo", mpi_comm=comm)

    if use_profiling:
        queue = cl.CommandQueue(cl_ctx,
            properties=cl.command_queue_properties.PROFILING_ENABLE)
    else:
        queue = cl.CommandQueue(cl_ctx)

    if lazy:
        actx = actx_class(comm, queue,
                allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue)),
                mpi_base_tag=12000)
    else:
        actx = actx_class(comm, queue,
                allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue)),
                force_device_scalars=True)

    # an array context for things that just can't lazy
    from grudge.array_context import PyOpenCLArrayContext
    init_actx = PyOpenCLArrayContext(queue,
        allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue)))

    # default i/o junk frequencies
    nviz = 500
    nhealth = 1
    nrestart = 5000
    nstatus = 1
    log_dependent = 0

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

    # discretization and model control
    order = 1
    alpha_sc = 0.3
    s0_sc = -5.0
    kappa_sc = 0.5
    dim = 2

    # material properties
    mu = 1.0e-5
    mu_override = False  # optionally read in from input

    # ACTII flow properties
    total_pres_inflow = 2.745e5
    total_temp_inflow = 2076.43

    # injection flow properties
    total_pres_inj = 50400
    total_temp_inj = 300.0

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
            log_dependent = int(input_data["log_dependent"])
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
            total_pres_inj = float(input_data["total_pres_inj"])
        except KeyError:
            pass
        try:
            total_temp_inj = float(input_data["total_temp_inj"])
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
        if log_dependent:
            print("\tDependent variable logging is ON.")
            print("\tWARNING: This may be a performance drag in lazy mode")
        else:
            print("\tDependent variable logging is OFF.")
        print("#### Simluation control data: ####\n")

    if rank == 0:
        print("\n#### Simluation setup data: ####")
        print(f"\ttotal_pres_inj = {total_pres_inj}")
        print(f"\ttotal_temp_inj = {total_temp_inj}")
        print("\n#### Simluation setup data: ####")

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
    # visocsity @ 400C, Pa-s
    mu_o2 = 3.76e-5
    mu_n2 = 3.19e-5
    mu_mix = mu_o2*mf_o2 + mu_n2*(1-mu_o2)  # 3.3456e-5
    mw = mw_o2*mf_o2 + mw_n2*(1.0 - mf_o2)
    r = 8314.59/mw
    cp = r*gamma/(gamma - 1)
    Pr = 0.75
    nspecies = 2

    if mu_override:
        mu = mu_input
    else:
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

    #
    # nozzle inflow #
    #
    # stagnation tempertuare 2076.43 K
    # stagnation pressure 2.745e5 Pa
    #
    # isentropic expansion based on the area ratios between the inlet (r=54e-3m) and
    # the throat (r=3.167e-3)
    #
    vel_inflow = np.zeros(shape=(dim,))
    vel_outflow = np.zeros(shape=(dim,))
    vel_injection = np.zeros(shape=(dim,))


    throat_height = 3.61909e-3
    inlet_height = 54.129e-3
    outlet_height = 28.54986e-3
    inlet_area_ratio = inlet_height/throat_height
    outlet_area_ratio = outlet_height/throat_height

    # 2 tracking scalars, either fuel or not-fuel
    y = np.zeros(nspecies)
    y_fuel = np.zeros(nspecies)
    y[0] = 1
    y[1] = 1
    species_names = ["air", "fuel"]

    inlet_mach = getMachFromAreaRatio(area_ratio=inlet_area_ratio,
                                      gamma=gamma,
                                      mach_guess=0.01)
    pres_inflow = getIsentropicPressure(mach=inlet_mach,
                                        P0=total_pres_inflow,
                                        gamma=gamma)
    temp_inflow = getIsentropicTemperature(mach=inlet_mach,
                                           T0=total_temp_inflow,
                                           gamma=gamma)
    rho_inflow = pres_inflow/temp_inflow/r
    vel_inflow[0] = inlet_mach*math.sqrt(gamma*pres_inflow/rho_inflow)

    if rank == 0:
        print("#### Simluation initialization data: ####")
        print(f"\tinlet Mach number {inlet_mach}")
        print(f"\tinlet temperature {temp_inflow}")
        print(f"\tinlet pressure {pres_inflow}")
        print(f"\tinlet rho {rho_inflow}")
        print(f"\tinlet velocity {vel_inflow[0]}")
        #print(f"final inlet pressure {pres_inflow_final}")

    outlet_mach = getMachFromAreaRatio(area_ratio=outlet_area_ratio,
                                       gamma=gamma,
                                       mach_guess=1.1)
    pres_outflow = getIsentropicPressure(mach=outlet_mach,
                                         P0=total_pres_inflow,
                                         gamma=gamma)
    temp_outflow = getIsentropicTemperature(mach=outlet_mach,
                                            T0=total_temp_inflow,
                                            gamma=gamma)
    rho_outflow = pres_outflow/temp_outflow/r
    vel_outflow[0] = outlet_mach*math.sqrt(gamma*pres_outflow/rho_outflow)

    if rank == 0:
        print(f"\toutlet Mach number {outlet_mach}")
        print(f"\toutlet temperature {temp_outflow}")
        print(f"\toutlet pressure {pres_outflow}")
        print(f"\toutlet rho {rho_outflow}")
        print(f"\toutlet velocity {vel_outflow[0]}")
        print("#### Simluation initialization data: ####\n")

    # injection mach number
    inj_mach = 1.0
    # for now it's all the same material
    gamma_inj = gamma

    pres_injection = getIsentropicPressure(mach=inj_mach,
                                           P0=total_pres_inj,
                                           gamma=gamma_inj)
    temp_injection = getIsentropicTemperature(mach=inj_mach,
                                              T0=total_temp_inj,
                                              gamma=gamma_inj)

    rho_injection = pres_injection/temp_injection/r
    vel_injection[0] = -inj_mach*math.sqrt(gamma*pres_injection/rho_injection)

    if rank == 0:
        print(f"\tinjector temperature {temp_injection}")
        print(f"\tinjector pressure {pres_injection}")
        print(f"\tinjector rho {rho_injection}")
        print(f"\tinjector velocity {vel_injection[0]}")

    if rank == 0:
        print("#### Simluation initialization data: ####\n")

    eos = IdealSingleGas(gamma=gamma, gas_const=r)
    gas_model = GasModel(eos=eos, transport=transport_model)

    # read geometry files
    geometry_bottom = None
    geometry_top = None
    if rank == 0:
        from numpy import loadtxt
        geometry_bottom = loadtxt("nozzleBottom.dat", comments="#", unpack=False)
        geometry_top = loadtxt("nozzleTop.dat", comments="#", unpack=False)
    geometry_bottom = comm.bcast(geometry_bottom, root=0)
    geometry_top = comm.bcast(geometry_top, root=0)

    # parameters to adjust the shape of the initialization
    vel_sigma = 2000
    temp_sigma = 2500
    # adjusted to match the mass flow rate
    vel_sigma_injection = 5000
    temp_sigma_injection = 5000
    temp_wall = 300

    inj_ymin = -0.0243245
    inj_ymax = -0.0227345
    bulk_init = InitACTII(dim=dim,
                          geom_top=geometry_top, geom_bottom=geometry_bottom,
                          P0=total_pres_inflow, T0=total_temp_inflow,
                          temp_wall=temp_wall, temp_sigma=temp_sigma,
                          vel_sigma=vel_sigma, nspecies=nspecies,
                          mass_frac=y,
                          inj_pres=total_pres_inj,
                          inj_temp=total_temp_inj,
                          inj_vel=vel_injection, inj_mass_frac=y_fuel,
                          inj_temp_sigma=temp_sigma_injection,
                          inj_vel_sigma=vel_sigma_injection,
                          inj_ytop=inj_ymax, inj_ybottom=inj_ymin)

    _inflow_init = UniformModified(
        dim=dim, nspecies=nspecies,
        mass_fracs=y,
        temperature=temp_inflow,
        pressure=pres_inflow,
        velocity=vel_inflow,
        temp_wall=temp_wall,
        temp_sigma=temp_sigma,
        vel_sigma=vel_sigma,
        ymin=-0.0270645,
        ymax=0.0270645
    )

    _outflow_init = UniformModified(
        dim=dim, nspecies=nspecies,
        mass_fracs=y,
        temperature=temp_outflow,
        pressure=pres_outflow,
        velocity=vel_outflow,
        temp_wall=temp_wall,
        temp_sigma=temp_sigma,
        vel_sigma=vel_sigma,
        ymin=-0.016874377,
        ymax=0.011675488
    )

    _injection_init = UniformModifiedInjector(
        dim=dim, nspecies=nspecies,
        mass_fracs=y_fuel,
        temperature=temp_injection,
        pressure=pres_injection,
        velocity=vel_injection,
        temp_wall=temp_wall,
        temp_sigma=temp_sigma_injection,
        vel_sigma=vel_sigma_injection,
        yc=-0.0283245 + 4e-3 + 1.59e-3/2.,
        zc=0.035/2.,
        inj_radius=1.59e-3/2.
    )

    def _boundary_state_func(discr, btag, gas_model, actx, init_func, **kwargs):
        bnd_discr = discr.discr_from_dd(btag)
        nodes = thaw(bnd_discr.nodes(), actx)
        return make_fluid_state(init_func(x_vec=nodes, eos=gas_model.eos,
                                          **kwargs), gas_model)

    def _inflow_state_func(discr, btag, gas_model, state_minus, **kwargs):
        return _boundary_state_func(discr, btag, gas_model,
                                    state_minus.array_context,
                                    _inflow_init, **kwargs)

    def _outflow_state_func(discr, btag, gas_model, state_minus, **kwargs):
        return _boundary_state_func(discr, btag, gas_model,
                                    state_minus.array_context,
                                    _outflow_init, **kwargs)

    def _injection_state_func(discr, btag, gas_model, state_minus, **kwargs):
        return _boundary_state_func(discr, btag, gas_model,
                                    state_minus.array_context,
                                    _injection_init, **kwargs)

    inflow = PrescribedFluidBoundary(boundary_state_func=_inflow_state_func)
    outflow = PrescribedFluidBoundary(boundary_state_func=_outflow_state_func)
    injection = PrescribedFluidBoundary(boundary_state_func=_injection_state_func)
    wall = IsothermalNoSlipBoundary()

    boundaries = {
        DTAG_BOUNDARY("inflow"): inflow,
        DTAG_BOUNDARY("outflow"): outflow,
        DTAG_BOUNDARY("wall"): wall,
        DTAG_BOUNDARY("injection"): injection
    }

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

        assert restart_data["nparts"] == nparts
    else:  # generate the grid from scratch
        local_mesh, global_nelements = generate_and_distribute_mesh(
            comm, get_mesh(dim=dim))
        local_nelements = local_mesh.nelements

    if rank == 0:
        logging.info("Making discretization")

    #discr = EagerDGDiscretization(
        #actx, local_mesh, order=order, mpi_communicator=comm
#
    #)

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

    # initialize the sponge field
    sponge_thickness = 0.09
    sponge_amp = 1.0/current_dt/1000
    sponge_x0 = 0.9

    sponge_init = InitSponge(x0=sponge_x0, thickness=sponge_thickness,
                             amplitude=sponge_amp)
    sponge_sigma = sponge_init(x_vec=thaw(discr.nodes(), actx))
    ref_cv = bulk_init(discr=discr, x_vec=thaw(discr.nodes(), init_actx),
                       eos=eos, time=0)
    ref_cv = thaw(freeze(ref_cv, init_actx), actx)

    vis_timer = None
    log_cfl = LogUserQuantity(name="cfl", value=current_cfl)

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

        if log_dependent:
            logmgr_add_many_discretization_quantities(logmgr, discr, dim,
                                                      extract_vars_for_logging,
                                                      units_for_logging)

            logmgr.add_quantity(log_cfl, interval=nstatus)
            logmgr.add_watches([
                ("cfl.max", ", cfl = {value:1.4f}\n"),
                ("min_pressure", "------- P (min, max) (Pa) = ({value:1.9e}, "),
                ("max_pressure", "{value:1.9e})\n"),
                ("min_temperature", "------- T (min, max) (K)  = ({value:7g}, "),
                ("max_temperature", "{value:7g})\n")])

    if rank == 0:
        logging.info("Before restart/init")

    if restart_filename:
        if rank == 0:
            logging.info("Restarting soln.")
        current_cv = restart_data["cv"]
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
            current_cv = connection(restart_data["cv"])
        if logmgr:
            logmgr_set_time(logmgr, current_step, current_t)
    else:
        # Set the current state from time 0
        if rank == 0:
            logging.info("Initializing soln.")
        current_cv = bulk_init(discr=discr, x_vec=thaw(discr.nodes(), init_actx),
                                  eos=eos, time=0)
        current_cv = thaw(freeze(current_cv, init_actx), actx)

    current_state = make_fluid_state(current_cv, gas_model)

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

    def my_write_status(dt, cfl, dv):
        status_msg = f"-------- dt = {dt:1.3e}, cfl = {cfl:1.4f}"
        temp = dv.temperature
        pres = dv.pressure
        actx = temp.array_context
        temp = thaw(freeze(temp, actx), actx)
        pres = thaw(freeze(pres, actx), actx)
        from grudge.op import nodal_min_loc, nodal_max_loc
        pmin = global_reduce(
            actx.to_numpy(nodal_min_loc(discr, "vol", pres)), op="min")
        pmax = global_reduce(
            actx.to_numpy(nodal_max_loc(discr, "vol", pres)), op="max")
        dv_status_msg = (
            f"\n-------- P (min, max) (Pa) = ({pmin:1.9e}, {pmax:1.9e})")
        tmin = global_reduce(
            actx.to_numpy(nodal_min_loc(discr, "vol", temp)), op="min")
        tmax = global_reduce(
            actx.to_numpy(nodal_max_loc(discr, "vol", temp)), op="max")
        dv_status_msg += (
            f"\n-------- T (min, max) (K)  = ({tmin:7g}, {tmax:7g})")
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

    def my_write_restart(step, t, cv):
        restart_fname = restart_pattern.format(cname=casename, step=step, rank=rank)
        if restart_fname != restart_filename:
            restart_data = {
                "local_mesh": local_mesh,
                "cv": cv,
                "t": t,
                "step": step,
                "order": order,
                "global_nelements": global_nelements,
                "num_parts": nparts
            }
            write_restart_file(actx, restart_data, restart_fname, comm)

    def my_health_check(dv):
        health_error = False
        if check_naninf_local(discr, "vol", dv.pressure):
            health_error = True
            logger.info(f"{rank=}: NANs/Infs in pressure data.")

        if global_reduce(check_range_local(discr, "vol", dv.pressure,
                                     health_pres_min, health_pres_max),
                                     op="lor"):
            health_error = True
            p_min = actx.to_numpy(nodal_min(discr, "vol", dv.pressure))
            p_max = actx.to_numpy(nodal_max(discr, "vol", dv.pressure))
            logger.info(f"Pressure range violation ({p_min=}, {p_max=})")

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

        mu = 0
        d_alpha_max = 0

        if state.is_viscous:
            from mirgecom.viscous import get_local_max_species_diffusivity
            mu = state.viscosity
            d_alpha_max = \
                get_local_max_species_diffusivity(
                    state.array_context,
                    state.species_diffusivity
                )

        return(
            length_scales / (state.wavespeed
            + ((mu + d_alpha_max + alpha) / length_scales))
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

    def my_pre_step(step, t, dt, state):
        fluid_state = make_fluid_state(cv=state, gas_model=gas_model)
        cv = fluid_state.cv
        dv = fluid_state.dv

        try:

            if logmgr:
                logmgr.tick_before()

            alpha_field = my_get_alpha(discr, fluid_state, alpha_sc)
            ts_field, cfl, dt = my_get_timestep(t, dt, fluid_state, alpha_field)
            if log_dependent:
                log_cfl.set_quantity(cfl)

            do_viz = check_step(step=step, interval=nviz)
            do_restart = check_step(step=step, interval=nrestart)
            do_health = check_step(step=step, interval=nhealth)
            do_status = check_step(step=step, interval=nstatus)

            if do_health:
                health_errors = global_reduce(my_health_check(dv), op="lor")
                if health_errors:
                    if rank == 0:
                        logger.warning("Fluid solution failed health check.")
                    raise MyRuntimeError("Failed simulation health check.")

            if do_status and (log_dependent == 0):
                my_write_status(dt=dt, cfl=cfl, dv=dv)

            if do_restart:
                my_write_restart(step=step, t=t, cv=cv)

            if do_viz:
                my_write_viz(step=step, t=t, cv=cv, dv=dv,
                             ts_field=ts_field, alpha_field=alpha_field)

        except MyRuntimeError:
            if rank == 0:
                logger.error("Errors detected; attempting graceful exit.")
            my_write_viz(step=step, t=t, cv=cv, dv=dv, ts_field=ts_field,
                         alpha_field=alpha_field)
            my_write_restart(step=step, t=t, cv=cv)
            raise

        dt = get_sim_timestep(discr, fluid_state, t, dt, current_cfl, t_final,
                              constant_cfl)

        return state, dt

    def my_post_step(step, t, dt, state):
        # Logmgr needs to know about EOS, dt, dim?
        # imo this is a design/scope flaw
        if logmgr:
            set_dt(logmgr, dt)
            set_sim_state(logmgr, dim, state, eos)
            logmgr.tick_after()
        return state, dt

    def my_rhs(t, state):
        fluid_state = make_fluid_state(cv=state, gas_model=gas_model)
        alpha_field = my_get_alpha(discr, fluid_state, alpha_sc)
        return (
            ns_operator(discr, state=fluid_state, time=t, boundaries=boundaries,
                        gas_model=gas_model, quadrature_tag=quadrature_tag)
            + av_laplacian_operator(discr, fluid_state=fluid_state,
                                    boundaries=boundaries,
                                    boundary_kwargs={"time": t,
                                                     "gas_model": gas_model},
                                    alpha=alpha_field, s0=s0_sc, kappa=kappa_sc,
                                    quadrature_tag=quadrature_tag)
            + sponge(cv=fluid_state.cv, cv_ref=ref_cv, sigma=sponge_sigma)
        )

    current_dt = get_sim_timestep(discr, current_state, current_t, current_dt,
                                  current_cfl, t_final, constant_cfl)

    current_step, current_t, current_cv = \
        advance_state(rhs=my_rhs, timestepper=timestepper,
                      pre_step_callback=my_pre_step,
                      post_step_callback=my_post_step,
                      istep=current_step, dt=current_dt,
                      t=current_t, t_final=t_final,
                      state=current_state.cv)
    current_state = make_fluid_state(current_cv, gas_model)

    # Dump the final data
    if rank == 0:
        logger.info("Checkpointing final state ...")
    final_dv = current_state.dv
    alpha_field = my_get_alpha(discr, current_state, alpha_sc)
    ts_field, cfl, dt = my_get_timestep(t=current_t, dt=current_dt,
                                        state=current_state, alpha=alpha_field)
    my_write_status(dt=dt, cfl=cfl, dv=final_dv)

    my_write_viz(step=current_step, t=current_t, cv=current_state.cv, dv=final_dv,
                 ts_field=ts_field, alpha_field=alpha_field)
    my_write_restart(step=current_step, t=current_t, cv=current_state.cv)

    if logmgr:
        logmgr.close()
    elif use_profiling:
        print(actx.tabulate_profiling_data())

    finish_tol = 1e-16
    assert np.abs(current_t - t_final) < finish_tol


if __name__ == "__main__":
    import sys

    logging.basicConfig(format="%(message)s", level=logging.INFO)

    import argparse
    parser = argparse.ArgumentParser(
        description="MIRGE-Com Isentropic Nozzle Driver")
    parser.add_argument("-r", "--restart_file", type=ascii, dest="restart_file",
                        nargs="?", action="store", help="simulation restart file")
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

    input_file = None
    if args.input_file:
        input_file = args.input_file.replace("'", "")
        print(f"Ignoring user input from file: {input_file}")
    else:
        print("No user input file, using default values")

    print(f"Running {sys.argv[0]}\n")
    main(restart_filename=restart_filename, user_input_file=input_file,
         use_profiling=args.profile, use_logmgr=args.log,
         use_overintegration=args.overintegration, lazy=lazy,
         actx_class=actx_class, casename=casename)

# vim: foldmethod=marker
