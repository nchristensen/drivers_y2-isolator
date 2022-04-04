#!/bin/bash
#mpirun -n 1 python -u -O -m mpi4py isolator_injection_init.py -i run_params.yaml
mpirun -n 1 python -u -m mpi4py isolator_injection_scalar_to_multispecies.py -i run_params_multi.yaml -r restart_data/isolator-000020
