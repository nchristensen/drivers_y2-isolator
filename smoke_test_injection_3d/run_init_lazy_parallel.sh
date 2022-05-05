#!/bin/bash
mpirun -n 4 python -u -m mpi4py isolator_injection_init.py -i run_params.yaml --lazy
