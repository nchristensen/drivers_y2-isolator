#!/bin/bash
gmsh -setnumber size 0.0032 -setnumber blratio 8 -o isolator_wall.msh -nopopup -format msh2 ./isolator_wall.geo -2
