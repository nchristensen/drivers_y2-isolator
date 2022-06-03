#!/bin/bash
gmsh -setnumber size 0.0032 -setnumber blratio 4 -setnumber blratiocavity 2 -setnumber blratioinjector 2 -setnumber injectorfac 10 -o isolator_new.msh -nopopup -format msh2 ./isolator.geo -3
