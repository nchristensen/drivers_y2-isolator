#!/bin/bash
gmsh -nt 10 -setnumber size 0.0032 -setnumber blratio 4 -setnumber blratioc 2 -o isolator.msh -nopopup -format msh2 ./isolator.geo -3
