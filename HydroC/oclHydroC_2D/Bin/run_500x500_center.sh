#!/bin/sh

RUNDIR=${HOME}/ptmp/Github/HydroC/oclHydroC_2DMpi
EXEDIR=${PWD}/../Src/
INPDIR=${PWD}/../../../Input
#RUNCMD="ccc_mprun -p hybrid -n 2 -N 1 -x "
RUNCMD="ccc_mprun -p hybrid -n 4 -N 2 -x "
# RUNCMD="ccc_mprun -p hybrid -n 16 -N 8 -c4 -x "
#RUNCMD="ccc_mprun -p hybrid -n 1 -N 1 -x "
set -x

mkdir -p ${RUNDIR}
cp ${EXEDIR}/hydro_kernels.cl ${RUNDIR}
cp ${EXEDIR}/oclparam.h ${RUNDIR}
cd ${RUNDIR}
rm *.txt *.lst
${RUNCMD} ${EXEDIR}/hydro -i ${INPDIR}/input_500x500_center.nml 
echo ${RUNCMD} gdb ${EXEDIR}/hydro 

#EOF
