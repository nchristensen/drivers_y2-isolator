#!/bin/bash
mpirun -n 4 python -u -O -m mpi4py isolator.py -i run_params.yaml --log --lazy
