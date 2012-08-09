#!/bin/sh

RUNDIR=${HOME}/ptmp/Github/HydroC/oclHydroC_2D
EXEDIR=${PWD}/../Src/
INPDIR=${PWD}/../../../Input
#RUNCMD="ccc_mprun -p hybrid -n 8 -N 4 -x "

mkdir -p ${RUNDIR}

cp ${EXEDIR}/hydro_kernels.cl ${RUNDIR}
cp ${EXEDIR}/oclparam.h ${RUNDIR}

cd ${RUNDIR}
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/lib64/openmpi/lib
${RUNCMD} valgrind --tool=memcheck ${EXEDIR}/hydro -i ${INPDIR}/input_500x500_corner.nml 

#EOF
