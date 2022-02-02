#!/bin/bash
gmsh -setnumber size 0.0016 -setnumber blratio 4 -setnumber blratiocavity 2 -setnumber blratioinjector 2 -setnumber injectorfac 20 -o isolator.msh -nopopup -format msh2 ./isolator.geo -3
