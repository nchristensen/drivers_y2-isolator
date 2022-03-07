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
            P0, T0, temp_wall, temp_sigma, vel_sigma, gamma_guess,
            mass_frac=None,
            inj_pres, inj_temp, inj_vel, inj_mass_frac=None,
            inj_gamma_guess,
            inj_temp_sigma, inj_vel_sigma,
            inj_ytop, inj_ybottom,
            inj_mach
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
        gamma_guess: float
            guesstimate for gamma
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
        self._gamma_guess = gamma_guess
        # TODO, calculate these from the geometry files
        self._throat_height = 3.61909e-3
        self._x_throat = 0.283718298
        self._mass_frac = mass_frac

        self._inj_P0 = inj_pres
        self._inj_T0 = inj_temp
        self._inj_vel = inj_vel
        self._inj_gamma_guess = inj_gamma_guess

        self._temp_sigma_injection = inj_temp_sigma
        self._vel_sigma_injection = inj_vel_sigma
        self._inj_mass_frac = inj_mass_frac
        self._inj_ytop = inj_ytop
        self._inj_ybottom = inj_ybottom
        self._inj_mach = inj_mach

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

        mach = zeros
        ytop = zeros
        ybottom = zeros
        theta = zeros
        gamma = self._gamma_guess

        theta_geom_top = get_theta_from_data(self._geom_top)
        theta_geom_bottom = get_theta_from_data(self._geom_bottom)

        # process the mesh piecemeal, one interval at a time
        # linearly interpolate between the data points
        area_ratio = ((self._geom_top[0][1] - self._geom_bottom[0][1]) /
                      self._throat_height)
        if self._geom_top[0][0] < self._x_throat:
            mach_left = getMachFromAreaRatio(area_ratio=area_ratio,
                                             gamma=gamma,
                                             mach_guess=0.01)
        elif self._geom_top[0][0] > self._x_throat:
            mach_left = getMachFromAreaRatio(area_ratio=area_ratio,
                                             gamma=gamma,
                                             mach_guess=1.01)
        else:
            mach_left = 1.0
        x_left = self._geom_top[0][0]
        ytop_left = self._geom_top[0][1]
        ybottom_left = self._geom_bottom[0][1]
        theta_top_left = theta_geom_top[0][1]
        theta_bottom_left = theta_geom_bottom[0][1]

        for ind in range(1, self._geom_top.shape[0]):
            area_ratio = ((self._geom_top[ind][1] - self._geom_bottom[ind][1]) /
                          self._throat_height)
            if self._geom_top[ind][0] < self._x_throat:
                mach_right = getMachFromAreaRatio(area_ratio=area_ratio,
                                                 gamma=gamma,
                                                 mach_guess=0.01)
            elif self._geom_top[ind][0] > self._x_throat:
                mach_right = getMachFromAreaRatio(area_ratio=area_ratio,
                                                 gamma=gamma,
                                                 mach_guess=1.01)
            else:
                mach_right = 1.0
            ytop_right = self._geom_top[ind][1]
            ybottom_right = self._geom_bottom[ind][1]
            theta_top_right = theta_geom_top[ind][1]
            theta_bottom_right = theta_geom_bottom[ind][1]

            # interpolate our data
            x_right = self._geom_top[ind][0]

            dx = x_right - x_left
            dm = mach_right - mach_left
            dyt = ytop_right - ytop_left
            dyb = ybottom_right - ybottom_left
            dtb = theta_bottom_right - theta_bottom_left
            dtt = theta_top_right - theta_top_left

            local_mach = mach_left + (xpos - x_left)*dm/dx
            local_ytop = ytop_left + (xpos - x_left)*dyt/dx
            local_ybottom = ybottom_left + (xpos - x_left)*dyb/dx
            local_theta_bottom = theta_bottom_left + (xpos - x_left)*dtb/dx
            local_theta_top = theta_top_left + (xpos - x_left)*dtt/dx

            local_theta = (local_theta_bottom +
                           (local_theta_top - local_theta_bottom) /
                           (local_ytop - local_ybottom)*(ypos - local_ybottom))

            # extend just a a little bit to catch the edges
            left_edge = actx.np.greater(xpos, x_left - 1.e-6)
            right_edge = actx.np.less(xpos, x_right + 1.e-6)
            inside_block = left_edge*right_edge

            mach = actx.np.where(inside_block, local_mach, mach)
            ytop = actx.np.where(inside_block, local_ytop, ytop)
            ybottom = actx.np.where(inside_block, local_ybottom, ybottom)
            theta = actx.np.where(inside_block, local_theta, theta)

            mach_left = mach_right
            ytop_left = ytop_right
            ybottom_left = ybottom_right
            theta_bottom_left = theta_bottom_right
            theta_top_left = theta_top_right
            x_left = x_right

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

        y = ones*self._mass_frac

        mass = eos.get_density(pressure=pressure, temperature=temperature,
                               species_mass_fractions=y)
        energy = mass*eos.get_internal_energy(temperature=temperature,
                                              species_mass_fractions=y)

        velocity = ones*np.zeros(self._dim, dtype=object)
        mom = mass*velocity
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
        #xc_right = zeros + 0.72163 + 0.000001
        xc_right = zeros + 0.726 + 0.000001
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

        # zero out the velocity
        for i in range(self._dim):
            velocity[i] = actx.np.where(inside_cavity, zeros, velocity[i])

        # fuel stream initialization
        # initially in pressure/temperature equilibrium with the cavity
        #inj_left = 0.71
        # even with the bottom corner
        inj_left = 0.70563
        # even with the top corner
        #inj_left = 0.7074
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
        aft_edge = ones
        fore_edge = ones
        if self._dim == 3:
            aft_edge = actx.np.greater(zpos, zc_aft)
            fore_edge = actx.np.less(zpos, zc_fore)
        inside_injector = (left_edge*right_edge*top_edge*bottom_edge *
                           aft_edge*fore_edge)

        inj_y = ones*self._inj_mass_frac

        inj_velocity = mach*np.zeros(self._dim, dtype=object)
        inj_velocity[0] = self._inj_vel[0]

        inj_mach = mach*0. + self._inj_mach

        # smooth out the injection profile
        # relax to the cavity temperature/pressure/velocity
        inj_x0 = 0.712
        # the entrace to the injector
        #inj_fuel_x0 = 0.7085
        # back inside the injector
        #inj_fuel_x0 = 0.717
        # out in the cavity
        inj_fuel_x0 = 0.7
        inj_fuel_y0 = -0.0243245 - 3.e-3
        inj_fuel_y1 = -0.0227345 + 3.e-3
        inj_fuel_z0 = 0.035/2. - 3.e-3
        inj_fuel_z1 = 0.035/2. + 3.e-3
        inj_sigma = 1500
        #gamma_guess_inj = gamma

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
        inj_mach = inj_weight*inj_mach

        # assume a smooth transition in gamma, could calculate it
        inj_gamma = (self._gamma_guess +
            (self._inj_gamma_guess - self._gamma_guess)*inj_weight)

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

        inj_mass = eos.get_density(pressure=inj_pressure,
                                   temperature=inj_temperature,
                                   species_mass_fractions=inj_y)
        inj_energy = inj_mass*eos.get_internal_energy(temperature=inj_temperature,
                                                      species_mass_fractions=inj_y)

        inj_velocity = mach*np.zeros(self._dim, dtype=object)
        inj_mom = inj_mass*inj_velocity

        # the velocity magnitude
        inj_cv = make_conserved(dim=self._dim, mass=inj_mass, momentum=inj_mom,
                                energy=inj_energy, species_mass=inj_mass*inj_y)

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

        inj_mass = eos.get_density(pressure=inj_pressure,
                                   temperature=inj_temperature,
                                   species_mass_fractions=inj_y)
        inj_energy = inj_mass*eos.get_internal_energy(temperature=inj_temperature,
                                                  species_mass_fractions=inj_y)

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
        mass = eos.get_density(pressure=pressure,
                               temperature=temperature,
                               species_mass_fractions=y)
        energy = mass*eos.get_internal_energy(temperature=temperature,
                                              species_mass_fractions=y)

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
def main(ctx_factory=cl.create_some_context, user_input_file=None,
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
    mu = 1.0e-5
    nspecies = 0

    # ACTII flow properties
    total_pres_inflow = 2.745e5
    total_temp_inflow = 2076.43

    # injection flow properties
    total_pres_inj = 50400
    total_temp_inj = 300.0
    mach_inj = 1.0

    # parameters to adjust the shape of the initialization
    #vel_sigma = 2000
    #temp_sigma = 2500
    vel_sigma = 1000
    temp_sigma = 1250
    # adjusted to match the mass flow rate
    vel_sigma_inj = 5000
    temp_sigma_inj = 5000
    temp_wall = 300

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
            total_pres_inj = float(input_data["total_pres_inj"])
        except KeyError:
            pass
        try:
            total_temp_inj = float(input_data["total_temp_inj"])
        except KeyError:
            pass
        try:
            mach_inj = float(input_data["mach_inj"])
        except KeyError:
            pass
        try:
            nspecies = int(input_data["nspecies"])
        except KeyError:
            pass
        try:
            vel_sigma = float(input_data["vel_sigma"])
        except KeyError:
            pass
        try:
            temp_sigma = float(input_data["temp_sigma"])
        except KeyError:
            pass
        try:
            vel_sigma_inj = float(input_data["vel_sigma_inj"])
        except KeyError:
            pass
        try:
            temp_sigma_inj = float(input_data["temp_sigma_inj"])
        except KeyError:
            pass

    if rank == 0:
        print("\n#### Simluation control data: ####")
        print(f"\torder = {order}")
        print(f"\tdimen = {dim}")
        print("#### Simluation control data: ####")

    if rank == 0:
        print("\n#### Simluation setup data: ####")
        print(f"\ttotal_pres_injection = {total_pres_inj}")
        print(f"\ttotal_temp_injection = {total_temp_inj}")
        print(f"\tvel_sigma = {vel_sigma}")
        print(f"\ttemp_sigma = {temp_sigma}")
        print(f"\tvel_sigma_injection = {vel_sigma_inj}")
        print(f"\ttemp_sigma_injection = {temp_sigma_inj}")
        print("#### Simluation setup data: ####")

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
        if nspecies == 0:
            print("\tno passive scalars, uniform ideal gas eos")
        elif nspecies == 2:
            print("\tpassive scalars to track air/fuel mixture, ideal gas eos")
        else:
            print("\tfull multi-species initialization with pyrometheus eos")
        print("#### Simluation material properties: ####")

    spec_diffusivity = 1.e-4 * np.ones(nspecies)
    transport_model = SimpleTransport(viscosity=mu, thermal_conductivity=kappa,
                                      species_diffusivity=spec_diffusivity)

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

    # initialize eos and species mass fractions
    y = np.zeros(nspecies)
    y_fuel = np.zeros(nspecies)
    if nspecies == 2:
        y[0] = 1
        y_fuel[1] = 1
        species_names = ["air", "fuel"]
    elif nspecies > 2:
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
        # Set the species mass fractions to the free-stream flow
        y_fuel[i_c2h4] = mf_c2h4
        y_fuel[i_h2] = mf_h2

        cantera_soln.TPY = init_temperature, 101325, y

    # make the eos
    if nspecies < 3:
        eos = IdealSingleGas(gamma=gamma, gas_const=r)
    else:
        from mirgecom.thermochemistry import make_pyrometheus_mechanism_class
        pyro_mech = make_pyrometheus_mechanism_class(cantera_soln)(actx.np)
        eos = PyrometheusMixture(pyro_mech, temperature_guess=init_temperature)
        species_names = pyro_mech.species_names

    gas_model = GasModel(eos=eos, transport=transport_model)

    inlet_mach = getMachFromAreaRatio(area_ratio=inlet_area_ratio,
                                      gamma=gamma,
                                      mach_guess=0.01)
    pres_inflow = getIsentropicPressure(mach=inlet_mach,
                                        P0=total_pres_inflow,
                                        gamma=gamma)
    temp_inflow = getIsentropicTemperature(mach=inlet_mach,
                                           T0=total_temp_inflow,
                                           gamma=gamma)
    if nspecies < 3:
        rho_inflow = pres_inflow/temp_inflow/r
        sos = math.sqrt(gamma*pres_inflow/rho_inflow)
    else:
        cantera_soln.TPY = temp_inflow, pres_inflow, y
        rho_inflow = cantera_soln.density
        gamma_loc = cantera_soln.cp_mass/cantera_soln.cv_mass
        sos = math.sqrt(gamma_loc*pres_inflow/rho_inflow)

    vel_inflow[0] = inlet_mach*sos

    if rank == 0:
        print("\n#### Simluation initialization data: ####")
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

    if nspecies < 3:
        rho_outflow = pres_outflow/temp_outflow/r
        sos = math.sqrt(gamma*pres_outflow/rho_outflow)
    else:
        cantera_soln.TPY = temp_outflow, pres_outflow, y
        rho_outflow = cantera_soln.density
        gamma_loc = cantera_soln.cp_mass/cantera_soln.cv_mass
        sos = math.sqrt(gamma_loc*pres_outflow/rho_outflow)

    vel_outflow[0] = outlet_mach*math.sqrt(gamma*pres_outflow/rho_outflow)

    if rank == 0:
        print("")
        print(f"\toutlet Mach number {outlet_mach}")
        print(f"\toutlet temperature {temp_outflow}")
        print(f"\toutlet pressure {pres_outflow}")
        print(f"\toutlet rho {rho_outflow}")
        print(f"\toutlet velocity {vel_outflow[0]}")

    # injection mach number
    if nspecies < 3:
        gamma_inj = gamma
    else:
        gamma_inj = 0.5*(1.24 + 1.4)

    pres_injection = getIsentropicPressure(mach=mach_inj,
                                           P0=total_pres_inj,
                                           gamma=gamma_inj)
    temp_injection = getIsentropicTemperature(mach=mach_inj,
                                              T0=total_temp_inj,
                                              gamma=gamma_inj)

    if nspecies < 3:
        rho_injection = pres_injection/temp_injection/r
        sos = math.sqrt(gamma*pres_injection/rho_injection)
    else:
        cantera_soln.TPY = temp_injection, pres_injection, y_fuel
        rho_injection = cantera_soln.density
        gamma_loc = cantera_soln.cp_mass/cantera_soln.cv_mass
        sos = math.sqrt(gamma_loc*pres_injection/rho_injection)
        if rank == 0:
            print(f"injection gamma guess {gamma_inj} cantera gamma {gamma_loc}")

    vel_injection[0] = -mach_inj*sos

    if rank == 0:
        print("")
        print(f"\tinjector Mach number {mach_inj}")
        print(f"\tinjector temperature {temp_injection}")
        print(f"\tinjector pressure {pres_injection}")
        print(f"\tinjector rho {rho_injection}")
        print(f"\tinjector velocity {vel_injection[0]}")
        print("#### Simluation initialization data: ####\n")

    # read geometry files
    geometry_bottom = None
    geometry_top = None
    if rank == 0:
        from numpy import loadtxt
        geometry_bottom = loadtxt("nozzleBottom.dat", comments="#", unpack=False)
        geometry_top = loadtxt("nozzleTop.dat", comments="#", unpack=False)
    geometry_bottom = comm.bcast(geometry_bottom, root=0)
    geometry_top = comm.bcast(geometry_top, root=0)

    inj_ymin = -0.0243245
    inj_ymax = -0.0227345
    bulk_init = InitACTII(dim=dim,
                          geom_top=geometry_top, geom_bottom=geometry_bottom,
                          P0=total_pres_inflow, T0=total_temp_inflow,
                          temp_wall=temp_wall, temp_sigma=temp_sigma,
                          vel_sigma=vel_sigma, nspecies=nspecies,
                          mass_frac=y, gamma_guess=gamma, inj_gamma_guess=gamma_inj,
                          inj_pres=total_pres_inj,
                          inj_temp=total_temp_inj,
                          inj_vel=vel_injection, inj_mass_frac=y_fuel,
                          inj_temp_sigma=temp_sigma_inj,
                          inj_vel_sigma=vel_sigma_inj,
                          inj_ytop=inj_ymax, inj_ybottom=inj_ymin,
                          inj_mach=mach_inj)

    viz_path = "viz_data/"
    vizname = viz_path + casename
    restart_path = "restart_data/"
    restart_pattern = (
        restart_path + "{cname}-{step:06d}-{rank:04d}.pkl"
    )

    local_mesh, global_nelements = generate_and_distribute_mesh(
        comm, get_mesh(dim=dim))
    #local_nelements = local_mesh.nelements

    if rank == 0:
        logging.info("Making discretization")

    discr = EagerDGDiscretization(actx, local_mesh, order, mpi_communicator=comm)

    if rank == 0:
        logging.info("Done making discretization")

    if rank == 0:
        logging.info("Initializing solution")

    current_cv = bulk_init(discr=discr, x_vec=thaw(discr.nodes(), actx),
                           eos=eos, time=0)
    current_state = make_fluid_state(current_cv, gas_model, init_temperature)

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
    my_write_viz(step=0, t=0, cv=current_state.cv, dv=current_state.dv)
    my_write_restart(step=0, t=0, cv=current_state.cv,
                     temperature_seed=current_state.dv.temperature)


if __name__ == "__main__":

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO)

    import argparse
    parser = argparse.ArgumentParser(
        description="MIRGE-Com Isentropic Nozzle Driver")
    parser.add_argument("-i", "--input_file", type=ascii, dest="input_file",
                        nargs="?", action="store", help="simulation config file")
    parser.add_argument("-c", "--casename", type=ascii, dest="casename", nargs="?",
                        action="store", help="simulation case name")
    parser.add_argument("--lazy", action="store_true", default=False,
                        help="enable lazy evaluation [OFF]")

    args = parser.parse_args()
    lazy = args.lazy

    # for writing output
    casename = "isolator_init"
    if args.casename:
        print(f"Custom casename {args.casename}")
        casename = args.casename.replace("'", "")
    else:
        print(f"Default casename {casename}")

    from grudge.array_context import get_reasonable_array_context_class
    actx_class = get_reasonable_array_context_class(lazy=lazy, distributed=True)

    input_file = None
    if args.input_file:
        input_file = args.input_file.replace("'", "")
        print(f"Ignoring user input from file: {input_file}")
    else:
        print("No user input file, using default values")

    print(f"Running {sys.argv[0]}\n")
    main(user_input_file=input_file, actx_class=actx_class, casename=casename,
         lazy=lazy)

# vim: foldmethod=marker
