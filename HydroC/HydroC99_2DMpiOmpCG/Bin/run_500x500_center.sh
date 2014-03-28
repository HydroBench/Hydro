#!/bin/sh

RUNDIR=${HOME}/ptmp/Github/HydroC/HydroC99_2DMpi
EXEDIR=${PWD}/../Src/
INPDIR=${PWD}/../../../Input
RUNCMD="env OMP_NUM_THREADS=4 KMP_AFFINITY=compact ccc_mprun -p standard -n 8 -N 4 -x -c 4 "


mkdir -p ${RUNDIR}
cd ${RUNDIR}
set -x
${RUNCMD} ${EXEDIR}/hydro -i ${INPDIR}/input_500x500_center.nml 

#EOF
