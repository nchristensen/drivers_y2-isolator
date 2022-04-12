#!/bin/bash
gmsh -setnumber size 0.0008 -setnumber blratio 4 -o isolator_with_surround.msh -nopopup -format msh2 ./isolator.geo -2
