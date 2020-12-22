#!/bin/sh

RUNDIR=${HOME}/ptmp/Github/HydroC/HydroC99_2DMpi
EXEDIR=${PWD}/../Src/
INPDIR=${PWD}/../../../Input
RUNCMD="ccc_mprun -p standard -n 1 -N 1 -x -T36000"


mkdir -p ${RUNDIR}
cd ${RUNDIR}
rm -rf Dep

${RUNCMD} ${EXEDIR}/hydro -i ${INPDIR}/input_23000x23000_corner.nml 

#EOF
