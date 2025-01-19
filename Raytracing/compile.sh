#!/bin/bash
gfortran -static-libgfortran -O3 -o raytracer.exe fmm_stdalone_FG.f90
rm *.mod
#gfortran -static-libgfortran -lgfortran -lgcc -lSystem -nodefaultlibs -mmacosx-version-min=10.6  -fbacktrace -O3 -o raytracer.exe fmm_stdalone_FG.f90