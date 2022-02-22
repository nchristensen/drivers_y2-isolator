#!/bin/bash
mpirun -n 1 python -u -O -m mpi4py isolator_injection_run.py -i run_params.yaml -r restart_data/isolator_init-000000 --log
