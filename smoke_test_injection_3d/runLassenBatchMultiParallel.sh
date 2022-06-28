#! /bin/bash --login
#BSUB -nnodes 1
#BSUB -G uiuc
#BSUB -W 720
#BSUB -J iso_inj_multi_3d_smoke
#BSUB -q pbatch
#BSUB -o runOutputMulti.txt
#BSUB -e runOutputMulti.txt

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
$jsrun_cmd bash -c 'POCL_CACHE_DIR=$POCL_CACHE_DIR_ROOT/$$ python -O -u -m mpi4py ./isolator_injection_init.py -c isolator_init_multi -i run_params_multi.yaml --lazy > mirge-multi-0.out'

echo "Running cold cache run"
$jsrun_cmd bash -c 'POCL_CACHE_DIR=$POCL_CACHE_DIR_ROOT/$$ python -O -u -m mpi4py ./isolator_injection_run.py -c isolator_multi -i run_params_multi.yaml -r restart_data/isolator_init_multi-000000 --log --lazy > mirge-multi-1.out'

echo "Running hot cache run"
$jsrun_cmd bash -c 'POCL_CACHE_DIR=$POCL_CACHE_DIR_ROOT/$$ python -O -u -m mpi4py ./isolator_injection_run.py -c isolator_multi -i run_params_multi.yaml -r restart_data/isolator_init_multi-000000 --log --lazy > mirge-multi-2.out'
