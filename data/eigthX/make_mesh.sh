#!/bin/bash
gmsh -setnumber size 0.0064 -setnumber blratio 1 -o isolator.msh -nopopup -format msh2 ./isolator.geo -2
