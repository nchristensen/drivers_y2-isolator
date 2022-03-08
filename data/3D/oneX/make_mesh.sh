#!/bin/bash
gmsh -setnumber size 0.0008 -setnumber blratio 1 -setnumber blratiocavity 1 -o isolator.msh -nopopup -format msh2 ./isolator.geo -3
