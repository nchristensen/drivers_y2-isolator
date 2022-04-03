#!/bin/bash
#BSUB -nnodes 1                   # number of nodes
#BSUB -W 120                       # walltime in minutes
#BSUB -q pbatch                   # queue to use


set -ex

# Run this script with 'bsub lassen.bsub.sh'

# Put any environment activation here, e.g.:
# source ../../config/activate_env.sh

# OpenCL device selection:
export PYOPENCL_CTX="port:tesla"      # Run on Nvidia GPU with pocl
# export PYOPENCL_CTX="port:pthread"  # Run on CPU with pocl

nnodes=$(echo $LSB_MCPU_HOSTS | wc -w)
nnodes=$((nnodes/2-1))
nproc=$((4*nnodes)) # 4 ranks per node, 1 per GPU

echo nnodes=$nnodes nproc=$nproc

# -a 1: 1 task per resource set
# -g 1: 1 GPU per resource set
# -n $nproc: $nproc resource sets
jsrun_cmd="jsrun -g 1 -a 1 -n $nproc"

# See
# https://mirgecom.readthedocs.io/en/latest/running.html#avoiding-overheads-due-to-caching-of-kernels
# on why this is important
export XDG_CACHE_HOME="/tmp/$USER/xdg-scratch"

# Fixes https://github.com/illinois-ceesd/mirgecom/issues/292
# (each rank needs its own POCL cache dir)
export POCL_CACHE_DIR_ROOT="/tmp/$USER/pocl-cache"

# Print task allocation
$jsrun_cmd js_task_info

echo "----------------------------"

# Run application
# -O: switch on optimizations
# POCL_CACHE_DIR=...: each rank needs its own POCL cache dir

# isolator_injection_init.py must be run with the same config as isolator_injection_run.py, in particular:
# - same numbers of ranks in both cases
# - lazy runs need a lazy init
$jsrun_cmd bash -c 'POCL_CACHE_DIR=$POCL_CACHE_DIR_ROOT/$$ python -m mpi4py isolator_injection_init.py -i run_params.yaml --lazy'
$jsrun_cmd bash -c 'POCL_CACHE_DIR=$POCL_CACHE_DIR_ROOT/$$ python -m mpi4py isolator_injection_run.py -r restart_data/isolator_init-000000 -i run_params.yaml --lazy'
