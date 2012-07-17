#!/bin/sh
#
# Copyright  (C) 1985-2012 Intel Corporation. All rights reserved.
#
# The information and source code contained herein is the exclusive property
# of Intel Corporation and may not be disclosed, examined, or reproduced in
# whole or in part without explicit written authorization from the Company.
#

test "$(hostname)" == "knc"			&& export PROD_DIR=/opt/intel/composer_xe_2013
test "$(hostname)" == "knf.caps-entreprise.com" && export PROD_DIR=/opt/intel/composerxe_mic

test "$(hostname)" == "knc"			&& . /opt/intel/impi/4.0.3.031/intel64/bin/mpivars.sh
test "$(hostname)" == "knf.caps-entreprise.com"	&& . /opt/intel/impi/4.0.3.014/intel64/bin/mpivars.sh

test "x${PROD_DIR}" == "x" && echo "No Product" && exit -1;

#. /scratch/morinl/intel/composer_xe_2013/base.sh
machine=intel64

if [ "$machine" != "ia32" -a "$machine" != "intel64" ]; then
  echo "ERROR: Unknown switch '$machine'. Accepted values: ia32, intel64"
  return 1;
fi


if [ -e $PROD_DIR/pkg_bin/idbvars.sh ]; then
   . $PROD_DIR/pkg_bin/idbvars.sh $machine 
fi
if [ -e $PROD_DIR/tbb/bin/tbbvars.sh ]; then
   . $PROD_DIR/tbb/bin/tbbvars.sh $machine 
fi
if [ -e $PROD_DIR/mkl/bin/mklvars.sh ]; then
   . $PROD_DIR/mkl/bin/mklvars.sh $machine 
fi
if [ -e $PROD_DIR/ipp/bin/ippvars.sh ]; then
   . $PROD_DIR/ipp/bin/ippvars.sh $machine 
fi
if [ -e $PROD_DIR/pkg_bin/compilervars_arch.sh ]; then
   . $PROD_DIR/pkg_bin/compilervars_arch.sh $machine 
fi

# Lang Safe
unset LC_MESSAGES
unset LC_COLLATE
unset LANG
unset LC_CTYPE



# MIC Part

export MIC_DIR=${PROD_DIR}/../mic
export COI_DIR=${MIC_DIR}/coi
export MIC_CC=icpc

export MIC_INCLUDES="-I${COI_DIR}/include"
export MIC_DEVICE_CFLAGS="-mmic"
export MIC_DEVICE_LDFLAGS="-L${COI_DIR}/device-linux-release/lib -lcoi_device -rdynamic -Wl,--enable-new-dtags -mmic"
export MIC_HOST_LDFLAGS="-L${COI_DIR}/host-linux-release/lib -lcoi_host -Wl,-rpath=\$$ORIGIN/../lib:${COI_DIR}/host-linux-release/lib"
