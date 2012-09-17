#!/bin/sh

RUNDIR=${HOME}/ptmp/Github/HydroC/cuHydroC_2DMpi
EXEDIR=${PWD}/../Src/
INPDIR=${PWD}/../../../Input
RUNCMD="ccc_mprun -p hybrid -n 2 -N 2 -x "
RUNCMD="ccc_mprun -p hybrid -n 4 -N 2 -x "
RUNCMD="ccc_mprun -p hybrid -n 16 -N 8 -c4 -x "
#RUNCMD="ccc_mprun -p hybrid -n 1 -N 1 -x "


mkdir -p ${RUNDIR}
cd ${RUNDIR}
rm *.txt *.lst
${RUNCMD} ${EXEDIR}/hydro -i ${INPDIR}/input_500x500_center.nml 

#EOF
