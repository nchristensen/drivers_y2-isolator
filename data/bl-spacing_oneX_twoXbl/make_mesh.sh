#!/bin/bash
gmsh -setnumber size 0.0008 -setnumber blratio 8 -o isolator.msh -nopopup -format msh2 ./isolator.geo -2
