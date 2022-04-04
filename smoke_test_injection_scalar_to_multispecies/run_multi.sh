#!/bin/bash
mpirun -n 1 python -u -O -m mpi4py isolator_injection_run.py -i run_params_multi.yaml -r restart_data/isolator_transfer-000020 --log --lazy
