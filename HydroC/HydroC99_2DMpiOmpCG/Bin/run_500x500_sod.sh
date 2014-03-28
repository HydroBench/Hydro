#!/bin/sh

RUNDIR=${HOME}/ptmp/Github/HydroC/HydroC99_2DMpi
EXEDIR=${PWD}/../Src/
INPDIR=${PWD}/../../../Input
RUNCMD="ccc_mprun -p standard -n 16 -N 8 -x "
# RUNCMD="ccc_mprun -p standard -n 1 -x"

mkdir -p ${RUNDIR}
cd ${RUNDIR}
echo ${RUNDIR}
${RUNCMD} ${EXEDIR}/hydro -i ${INPDIR}/input_500x500_sod.nml 

#EOF
