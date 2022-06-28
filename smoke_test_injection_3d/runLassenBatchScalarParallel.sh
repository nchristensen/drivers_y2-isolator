#! /bin/bash --login
#BSUB -nnodes 1
#BSUB -G uiuc
#BSUB -W 720
#BSUB -J iso_inj_scalar_3d_smoke
#BSUB -q pbatch
#BSUB -o runOutputScalar.txt
#BSUB -e runOutputScalar.txt

module load gcc/8.3.1
module load spectrum-mpi
conda deactivate
conda activate mirgeDriver.Y2isolator
export PYOPENCL_CTX="port:tesla"
jsrun_cmd="jsrun -g 1 -a 1 -n 4"
export XDG_CACHE_HOME="/tmp/$USER/xdg-scratch"
export POCL_CACHE_DIR_ROOT="/tmp/$USER/pocl-cache"
$jsrun_cmd js_task_info

echo "Running init"
$jsrun_cmd bash -c 'POCL_CACHE_DIR=$POCL_CACHE_DIR_ROOT/$$ python -O -u -m mpi4py ./isolator_injection_init.py -c isolator_init_scalar -i run_params_scalar.yaml --lazy > mirge-scalar-0.out'

echo "Running cold cache run"
$jsrun_cmd bash -c 'POCL_CACHE_DIR=$POCL_CACHE_DIR_ROOT/$$ python -O -u -m mpi4py ./isolator_injection_run.py -c isolator_scalar -i run_params_scalar.yaml -r restart_data/isolator_init_scalar-000000 --log --lazy > mirge-scalar-1.out'

echo "Running hot cache run"
$jsrun_cmd bash -c 'POCL_CACHE_DIR=$POCL_CACHE_DIR_ROOT/$$ python -O -u -m mpi4py ./isolator_injection_run.py -c isolator_scalar -i run_params_scalar.yaml -r restart_data/isolator_init_scalar-000000 --log --lazy > mirge-scalar-2.out'
