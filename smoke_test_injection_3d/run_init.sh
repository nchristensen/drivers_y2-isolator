#!/bin/bash
mpirun -n 2 python -u -m mpi4py isolator_injection_init.py -i run_params.yaml
