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

    mass = cv.mass
    mass_flat = to_numpy(flatten(mass, actx), actx)

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

    print(f"rank {rank=} Minimum density {mass_min_val} at {mass_min_loc}")
    print(f"rank {rank=} Maximum density {mass_max_val} at {mass_max_loc}")

    pressure = dv.pressure
    pressure_flat = to_numpy(flatten(pressure, actx), actx)

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

    print(f"rank {rank=} Minimum pressure {pressure_min_val} at {pressure_min_loc}")
    print(f"rank {rank=} Maximum pressure {pressure_max_val} at {pressure_max_loc}")

    temperature = dv.temperature
    temperature_flat = to_numpy(flatten(temperature, actx), actx)

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

    print(f"rank {rank=} Minimum temperature {temperature_min_val} at {temperature_min_loc}")
    print(f"rank {rank=} Maximum temperature {temperature_max_val} at {temperature_max_loc}")


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

