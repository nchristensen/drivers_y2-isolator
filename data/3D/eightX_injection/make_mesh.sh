#!/bin/bash
gmsh -setnumber size 0.0064 -setnumber blratio 1 -setnumber blratiocavity 1 -setnumber blratioinjector 1 -setnumber injectorfac 4 -o isolator.msh -nopopup -format msh2 ./isolator.geo -3
