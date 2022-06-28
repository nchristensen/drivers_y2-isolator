#!/bin/bash


NCPUS=$(getconf _NPROCESSORS_ONLN)

gmsh -setnumber size 0.0064 -setnumber blratio 4 -setnumber blratiocavity 2 -setnumber blratioinjector 2 -setnumber injectorfac 3 -setnumber shearfac 4 -o isolator_new.msh -nopopup -format msh2 ./isolator.geo -3 -nt $NCPUS
