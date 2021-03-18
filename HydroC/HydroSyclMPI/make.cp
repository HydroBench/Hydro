CXX=compute++

ARCHEXT=cp

CFLAGS=-I/home/weilljc/codeplay/ComputeCpp-CE-2.4.0-x86_64-linux-gnu/include \
 -I/home/weilljc/intel/oneapi/compiler/2021.1.2/linux/include/sycl -DSYCL_EXTERNAL=""
 


ifneq ($(DBG),O)
   OPTIM+=-O3 -g
else
   OPTIM += -g -O0
endif


ifeq ($(MPI),O)
    # not yet
endif

LIBS=-lsycl -lOpenCLls lib