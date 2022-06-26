# CEESD_y2-isolator

Isolator case using [MIRGE-Com](https://github.com/illinois-ceesd/mirgecom)

Driver and associated tools for simulating flow through the ACT-II facility in direct-connect configuration.
The simulation is viscous, 2D, single species.

Initialization considers the Mach number as a function of duct area ratio and inflow stagnation conditions. The velocity and tempearture are smoothed using tanh functions to match the near wall flow conditions for isothermal, noslip walls.  The velocity is further modified to match the duct geometry.

There are two configurations, [isolator.py](isolator.py) and [isolator_injection.py](isolator_injection.py). The former is a geometry/init without fuel injection while the later includes injection. Note that the isolator_injection driver is currently split into two pieces, run and init, both support lazy. The original isolator driver used an eager array context for intization and a strategy that could be quite time consuming.

## Directory structure

The main driver is isolator.py with the problem setup in [baseline](baseline) being the generally accepted way to run the simulation.

Simulation data (i.e. meshes) are located in [data](data)

Numerical experiments and/or driver variations can be located in [experiments](experiments), these are variations that may or may not derive from the current driver in baseline, although they generally have common ancestery.

The driver/data used to create the timing data is located in [timing_runs](timing_runs), and is a smaller version of the full baseline run.

Nightly CI is performed using the directory [smoke_test](smoke_test)

## Installation

```
./buildMirge.sh
```

Will checkout and build a local copy of emirge, complete with all the needed subpackages. A new conda environemnt will be created named mirgeDriver.Y2isolator. 

### Additional options

```
./buildMirge.sh -h
```

There are several optional build parameters that are detailed further

### Archiving MIRGE-Com version information

```
./updateVersionInfo.sh
```

Save the current build state of MIRGE-Com and associated packages into [platforms](platforms). Setting the environment variable MIRGE_PLATFORM will all the user to retrieve this version information when building MIRGE-Com using the buildMirge.sh script.

## Building the mesh

### Install gmsh
In the emirge directory activate the ceesd environment and install gmsh
```
conda activate mirgeDriver.Y2isolator
conda install gmsh
```

### Run gmsh
In the directory containing the case file generate the mesh
```
gmsh -o isolator.msh -nopopup -format msh2 ./isolator.geo -2
```

Additional options may be available for fine-tuning mesh generation. Most mesh directories contain a script make_mesh.sh demonstrating how to build a particular mesh version.

## Running a case

Activate the correct conda environment
```
conda activate mirgeDriver.Y2isolator
```

Most subdirectories contain a run.sh script that outlines how to run the problem.

The case can the be run similar to other MIRGE-Com applications.
For examples see the MIRGE-Com [documentation](https://mirgecom.readthedocs.io/en/latest/running/systems.html)

## Eager autotuning on Lassen

### Setup

Clone emirge and install with eager autotuning branches
```
git clone git@github.com:illinois-ceesd/emirge.git
cd emirge
./install.sh --fork=nchristensen --branch=production --conda-prefix=PATHTOCONDA
```

Install charm4py if using charm4py autotuning script (currently needed for Lassen)

```
# Setup environment
module load spectrum-mpi
module load gcc/8
export CHARM_EXTRA_BUILD_OPTS="mpicxx"
conda activate ceesd #Or the name the the environment you chose

# Install charm4py
pip install cython greenlet
git clone git@github.com:UIUC-PPL/charm4py.git
cd charm4py
mkdir charm_src
cd charm_src
git clone git@github.com:UIUC-PPL/charm.git
cd ..
pip install --install-option="--mpi" .

# Test charm4py 
bsub -nnodes 1 -Ip -XF -W 30 /bin/bash
srun -n 4 python -m mpi4py <path to charm4py>/examples/hello/array_hello.py
exit
```

Change to preferred directory and clone y2-isolator with autotuning driver.
```
git clone git@github.com:nchristensen/drivers_y2-isolator.git
```

### Running

We'll interactively for this example.

Request some nodes.
```
bsub -nnodes 4 -Ip -XF -W 240 /bin/bash
```

```
conda activate ceesd
module load gcc/8
module load spectrum-mpi
cd drivers_y2-isolator/test_mesh/eager/quarterX/data
./make_mesh.sh
cd ..
```

Edit `run_params.yaml` to execute a single time step and then run the the driver.

```
# Execute the driver on two nodes
# Export OPENCL_VENDOR_PATH if necessary.
#export OPENCL_VENDOR_PATH="$HOME/miniforge3/envs/ceesd/etc/OpenCL/vendors"
jsrun -n 2 -a 1 -g 1 python -O -m mpi4py isolator.py -i run_params.yaml --autotune
```

The pickled kernels are saved in the `pickled_kernels` directory by default.

Execute the autotuning script
```
#TODO: Fix grudge install so don't need path to grudge 
jsrun -n 16 -a 1 -g 1 python -O -m mpi4py PATH_TO_GRUDGE/grudge/loopy_dg_kernels/parallel_autotuning_charm4py.py
```

This script may occasionally fails with a CUDA error. Try running the script again if this occurs.

The script will use n-1 GPUs to execute autotuning on each of the pickled kernels and save hjson
transformation files in the `hjson` directory. With the hjson files created, run the driver again. It should now load the transformations from the hjson files
and execute much faster (provided the mesh is sufficiently large and the polynomial order sufficiently high).

```
jsrun -n 2 -a 1 -g 1 python -O -m mpi4py isolator.py -i run_params.yaml --autotune
```

### Notes
The autotuner fixes the kernel parameters before pickling the kernels so multiple .pickle files will be 
created when different ranks have different numbers of elements. Changing the number of MPI ranks will consequently
require re-running the autotuner. To minimize the autotuning time, run the driver with as few ranks as possible.

### Known issues
Pocl CUDA kernel execution times often have only a few digits of accuracy which may mean a suboptimal set of
transformations is selected.
