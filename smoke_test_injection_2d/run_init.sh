#!/bin/bash
#mpirun -n 1 python -u -O -m mpi4py isolator_injection_init.py -i run_params.yaml
mpirun -n 2 python -u -O -m mpi4py isolator_injection_init.py -i run_params.yaml
#mpirun -n 1 python -u -m mpi4py isolator_injection_init.py -i run_params.yaml --lazy
#mpirun -n 2 python -u -m mpi4py isolator_injection_init.py -i run_params.yaml --lazy
