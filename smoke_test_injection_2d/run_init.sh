#!/bin/bash
mpirun -n 1 python -u -O -m mpi4py isolator_injection_init.py -i run_params.yaml
