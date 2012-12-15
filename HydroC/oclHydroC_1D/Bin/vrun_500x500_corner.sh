#!/bin/sh
set -x
RUNDIR=${HOME}/ptmp/Github/HydroC/oclHydroC_1D
EXEDIR=${PWD}/../Src/
INPDIR=${PWD}/../../../Input
# RUNCMD="ccc_mprun -p hybrid -n 1 -N 1 -x "

mkdir -p ${RUNDIR}

# WARNING: this version of the code requires to have the .cl file in
# the current directory
cp ${EXEDIR}/hydro_kernels.cl ${RUNDIR}
cp ${EXEDIR}/oclparam.h ${RUNDIR}

cd ${RUNDIR}
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-.}:/usr/lib64/openmpi/lib
# ${RUNCMD} ${EXEDIR}/hydro -i ${INPDIR}/input_500x500_corner.nml 
valgrind --tool=memcheck ${RUNCMD} ${EXEDIR}/hydro -i ${INPDIR}/input_500x500_corner.nml 
# gdb ${EXEDIR}/hydro <<EOF run  -i ${INPDIR}/input_500x500_corner.nml where quit EOF
exit 0
#EOF
