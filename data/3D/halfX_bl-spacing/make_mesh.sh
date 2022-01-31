#!/bin/bash
gmsh -nt 100 -setnumber size 0.0016 -setnumber blratio 4 -setnumber blratioc 2 -o isolator.msh -nopopup -format msh2 ./isolator.geo -3
