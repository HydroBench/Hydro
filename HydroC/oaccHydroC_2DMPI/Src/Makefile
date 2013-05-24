######################
# GNU Makefile version
######################

CFLAGS=-D__NO_INLINE__ 
ifeq ($(PREC),DOUBLE)
	CFLAGS=-D__NO_INLINE__ -DUSE_DOUBLE 
endif

#This variable can be overridden  from the command line
TARGET=opencl

ifeq ($(VERBOSITY),)
	VERBOSITY=warn
endif
CC=mpicc

CFLAGS:= $(CFLAGS) -std=c99 -I../common -w -O3 -Wno-unknown-pragmas -D_POSIX_C_SOURCE=199309L 
LDFLAGS = ../common/hmpp_util.o ../common/getopt.o -lm -lrt

export HMPPRT_LOG_LEVEL=$(VERBOSITY)


HMPPCG_FLAGS= --codelet-required --force --color
ifeq ($(TARGET),opencl)
 ###### TARGET OPENCL
	export OPENCL_INC_PATH=/opt/cuda/4.0/include
	export COMPUTE=:0
	export CL_KHR_FP64=1
	export GPU_USE_SYNC_OBJECTS=1
	export ACC_DEVICE_TYPE=acc_device_opencl
	HMPPCG_FLAGS:= $(HMPPCG_FLAGS) --openacc-target OpenCL 
else
ifeq ($(TARGET),mic)
	export HMPP_CODELET_COMPILER_CFLAGS=$(OPT)
	export HMPP_MIC_DEVICE_LIBRARY:=/opt/intel/mic/coi/device-linux-release/lib/libcoi_device.so

	export HMPPRT_LOG_LEVEL:=warn
#	export HMPPRT_NUM_THREADS:=120
	export HMPP_CODELET_COMPILER_CC=icpc
	export HMPP_CODELET_COMPILER_CFLAGS=-O3 -vec-report -unroll-aggressive -Wno-unknown-pragmas
	export ACC_DEVICE_TYPE=mic
	export HMPP_ACC_TARGET=MIC
	HMPPCG_FLAGS:= $(HMPPCG_FLAGS) --openacc-target MIC
	CC=mpiicc
else
 NVCC_FLAGS=--nvcc-options -Xptxas=-v,-prec-div=false,-gencode,\"arch=compute_20,code=sm_20\" 
endif
endif
 HMPPCC_OPENACC= hmpp $(HMPPCG_FLAGS) $(CC) $(CFLAGS)


INPUT=input



PGM=HydroC
HEADER=$(wildcard *.h)
SRC=cmpflx.c conservar.c equation_of_state.c hydro_godunov.c main.c parametres.c riemann.c trace.c vtkfile.c compute_deltat.c constoprim.c hydro_funcs.c hydro_utils.c make_boundary.c qleftright.c slope.c utils.c SplitSurface.c cclock.c

BCK_SRC=hydro_godunov.c riemann.c hggodunov_cuda.cu
HOST=$(shell uname -n | sed 's/[0-9]//g')

OBJ = $(patsubst %.c, %.o, ${SRC})
DEP = $(patsubst %.c, %.d, ${SRC}) $(DEP2)

run-acc: HydroC-acc.run

all: HydroC.run HydroC-acc.run

build: HydroC.exe HydroC-acc.exe
 
##################################################################

HydroC.run: HydroC.exe
	./HydroC.exe -i $(INPUT)

HydroC.exe: depend $(DEP) $(SRC)
	@echo ""
	@echo "ICC Version"
	$(CC) $(CFLAGS) -o HydroC.exe $(OPT) $(SRC) -lm -lrt $(HMPPENDFLAGS)

##################################################################

HydroC-acc.run: HydroC-acc.exe
	D=$$PWD; cp *.cl *.hmf  ~/ptmp; cd ~/ptmp; ccc_mprun -p hybrid -N1 -n1 -x -c4 $$D/HydroC-acc.exe -i $$D/$(INPUT)

HydroC-acc.exe: $(DEP) $(SRC)
	@echo ""
	@echo "HMPP/OpenACC Version"
	$(HMPPCC_OPENACC) -o HydroC-acc.exe $(OPT) $(SRC) -lm $(HMPPENDFLAGS)

##################################################################


#include $(DEP)

.SUFFIXES:  .o .cu .d .c 


depend:
	make ${DEP}

.c.d:
	$(CC) ${CPPFLAGS} ${CFLAGS} $(HFLAGS) -M $< -o $@	

dos2unix:
	dos2unix $(SRC) $(HEADER)
	perl -i.bak -p -e 's/\015//ig' $(SRC) $(HEADER)


clean:
	rm -rf *~ *.so *.o *.exe *.bic *.linkinfo *.cu.* *.hmf* *.hmc* *.i *.d 
	rm -rf *.cu* *extracted* *hdpp* *translated* *.hmg* *.tif

#EOF
