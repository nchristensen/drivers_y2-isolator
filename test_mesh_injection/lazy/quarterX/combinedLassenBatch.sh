#! /bin/bash --login
#BSUB -nnodes 5
#BSUB -G uiuc
#BSUB -W 720
#BSUB -J 3d_inj_quarterX_lazy
#BSUB -q pbatch
#BSUB -o runOutputComb.txt
#BSUB -e runOutputComb.txt

module load gcc/8.3.1
module load spectrum-mpi
conda deactivate
conda activate mirgeDriver.Y2isolator-fusion
export PYOPENCL_CTX="port:tesla"
#export PYOPENCL_CTX="0:2"
jsrun_cmd="jsrun -g 1 -a 1 -n 20"
export XDG_CACHE_HOME="/tmp/$USER/xdg-scratch"
export POCL_CACHE_DIR_ROOT="/tmp/$USER/pocl-cache"
$jsrun_cmd js_task_info
$jsrun_cmd bash -c 'POCL_CACHE_DIR=$POCL_CACHE_DIR_ROOT/$$ python -O -u -m mpi4py ./isolator_injection_init.py -i run_params.yaml --lazy > mirge-0.out'

$jsrun_cmd bash -c 'POCL_CACHE_DIR=$POCL_CACHE_DIR_ROOT/$$ python -O -u -m mpi4py ./isolator_injection_run.py -i run_params.yaml -r restart_data/isolator_init-000000 --log --lazy > mirge-1.out'
