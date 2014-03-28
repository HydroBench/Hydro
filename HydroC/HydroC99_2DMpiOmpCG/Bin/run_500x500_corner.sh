#!/bin/sh

RUNDIR=${HOME}/ptmp/Github/HydroC/HydroC99_2DMpi
EXEDIR=${PWD}/../Src/
INPDIR=${PWD}/../../../Input
# RUNCMD="ccc_mprun -p standard -n 8 -N 4 -x "


mkdir -p ${RUNDIR}
cd ${RUNDIR}
${RUNCMD} ${EXEDIR}/hydro -i ${INPDIR}/input_500x500_corner.nml 

#EOF
