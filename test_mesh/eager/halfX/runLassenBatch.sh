#! /bin/bash --login
#BSUB -nnodes 2
#BSUB -G uiuc
#BSUB -W 120
#BSUB -J Y2_iso_3d_halfX
#BSUB -q pdebug
#BSUB -o runOutput.txt
#BSUB -e runOutput.txt

module load gcc/8.3.1
module load spectrum-mpi
conda deactivate
conda activate mirgeDriver.Y2isolator
export PYOPENCL_CTX="port:tesla"
#export PYOPENCL_CTX="0:2"
jsrun_cmd="jsrun -g 1 -a 1 -n 8"
export XDG_CACHE_HOME="/tmp/$USER/xdg-scratch"
export POCL_CACHE_DIR_ROOT="/tmp/$USER/pocl-cache"
$jsrun_cmd js_task_info
$jsrun_cmd bash -c 'POCL_CACHE_DIR=$POCL_CACHE_DIR_ROOT/$$ python -O -u -m mpi4py ./isolator.py -i run_params.yaml --log  > mirge-1.out'
