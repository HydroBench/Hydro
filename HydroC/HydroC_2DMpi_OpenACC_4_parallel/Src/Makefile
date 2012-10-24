PGM=HydroC
HEADER=$(wildcard *.h)
SRC=SplitSurface.c parametres.c vtkfile.c hydro_funcs.c hydro_utils.c utils.c main.c make_boundary.c hydro_godunov.c riemann.c compute_deltat.c constoprim.c slope.c qleftright.c cmpflx.c conservar.c equation_of_state.c trace.c
NOSRC=
SRCCU=$(wildcard *.cu)

OBJ = $(patsubst %.cu, %.o, ${SRCCU}) $(patsubst %.c,  %.o, ${SRC}) 

DEP = $(patsubst %.cu, %.d, ${SRCCU}) $(patsubst %.c,  %.d, ${SRC})

NVCC=gcc
CUDAINC=-I/opt/cuda/4.0/include 
MPIRUNCMD=ccc_mprun 

CC=gcc -Wall
OPT=-g
OPT=-O3 -DNDEBUG -DFAST

MACHINE=$(shell uname -n | sed  's/[0-9]//g')
ifeq ($(MACHINE), uchu)
#cg au lieu de ca pour que le L2 (sinon L1+L2)
NVCC=nvcc -arch=sm_13
endif
ifeq ($(MACHINE), capac)
NVCC=nvcc -arch=sm_20 -Xptxas -dlcm=cg
endif
ifeq ($(MACHINE), inti)
NVCC=nvcc -arch=sm_20 -Xptxas -dlcm=cg
MPIINC=-I$(MPI_HOME)/include
MPILIB=-L$(MPI_HOME)/lib64 -lmpi
endif
ifeq ($(MACHINE), todi)
CC=cc
OPT=-DNDEBUG -DFAST -h list=m $(CUDA_KERNELS)
NVCC=nvcc -arch=sm_20 -Xptxas -dlcm=cg -D__cudart_builtin__=__device__
MPIINC=-I$(MPICH_DIR)/include
MPILIB=-L$(MPICH_DIR)/lib -lmpich_cray
CUDAINC=$(CRAY_CUDA_INCLUDE_OPTS)
CUDALIB=-lcuda -lcudart
MPIRUNCMD=aprun -N1 -n
endif
ifeq ($(MACHINE), castor)
CC=pgcc
OPT=-DNDEBUG -DFAST -acc -ta=nvidia -Minfo=accel $(CUDA_KERNELS)
PPN=2
NVCC=nvcc -arch=sm_20 -Xptxas -dlcm=cg
MPIINC=-I$(MPI_ROOT)/include
MPILIB=-L$(MPI_ROOT)/lib -lmpich -lmpl
CUDAINC=-I/apps/castor/CUDA-4.1/cuda/include
CUDALIB=-L$(CUDADIR)/../lib64 -lcuda -lcudart
MPIRUNCMD=mpiexec.hydra -rmk slurm 
endif

# OPT+= -DFLOPS

CFLAGS=$(OPT) $(CUDAINC) $(MPIINC)

$(PGM): $(DEP) $(OBJ)
	$(CC) -o $(PGM) $(OPT) $(OBJ) -lm $(HMPPENDFLAGS) $(MPILIB) $(CUDALIB)

depend:
	-rm ${DEP}; make ${DEP}

LSRCH=$(shell echo $(HEADER) | tr ' ' '\n' | sort)
LSRCC=$(shell echo $(SRC) | tr ' ' '\n'| sort)
LSRCU=$(shell echo $(SRCCU) | tr ' ' '\n'| sort)

listing:
	a2ps --medium=A4dj -C -1 -R -l90 --toc -E --prologue=color --file-align=fill -o listing.ps Makefile $(LSRCH) $(LSRCC) $(LSRCU)
	ps2pdf listing.ps
	rm listing.ps

CUHYDROC:
	make clean
	cd $(HOME); tar czvf CUHYDROC.`date +%y%m%d_%H%M%S`.tgz cuHydroC/*.c cuHydroC/*.cu cuHydroC/*.h cuHydroC/Makefile cuHydroC/input

save:
	scp -r uchu:cuHydroC $(HOME)
	make clean
	-make listing
	cd $(HOME); tar czvf CUHYDROC.`date +%y%m%d_%H%M%S`.tgz cuHydroC

OKDIR=$(shell date +%y%m%d_%H%M%S)
ok:
	mkdir -p Archives/ok$(OKDIR)
	cp *.c *.cu *.h Makefile Archives/ok$(OKDIR)

clean   :
	-/bin/rm -f *.o *.so *~ *.vts  *.bic *.bak *.out ${PGM} *.csv *.log *.d *.lst *.cub *.ptx

indent:
	indent $(SRC) $(SRCCU) $(HEADER)

dos2unix:
	dos2unix $(SRC) $(HEADER)
	perl -i.bak -p -e 's/\015//ig' $(SRC) $(SRCCU) $(HEADER)


.SUFFIXES:  .o .d .c .cu
include $(DEP)

.c.d:
	@gcc ${CPPFLAGS}  $(CUDAINC) $(MPIINC) $(HFLAGS) -M $< -o $@

.cu.d:
	@$(NVCC) $(MPIINC) ${CPPFLAGS} $(HFLAGS) -M $< -o $@
	@perl -i -p -e "s+$(HOME)/cuHydroC//++" $@

.c.o    :
	${CC} ${CFLAGS} -c $< $(HMPPENDFLAGS)

.cpp.o  :
	${CC} ${CFLAGS} -c $< -o $@
.cu.o:
	${NVCC} $(MPIINC) -c $< -o $@

LNAME=$(shell basename $(PWD))
RUNDIR=$(SCRATCH)/prace/wp9/prace_hydro_accel.run/$(LNAME)
run: $(PGM)
	mkdir -p $(RUNDIR)
	cp $(PGM) input $(RUNDIR)
	cd $(RUNDIR); rm -rf Dep output*.vts
	# cd $(RUNDIR); time ./$(PGM) -i input
	cd $(RUNDIR); $(MPIRUNCMD) $(N) ./$(PGM) -i input

run2: $(PGM)
	mkdir -p $(RUNDIR)
	cp $(PGM) input2 $(RUNDIR)
	cd $(RUNDIR); rm -rf Dep output*.vts
	# cd $(RUNDIR); time ./$(PGM) -i input
	cd $(RUNDIR); $(MPIRUNCMD) $(N) ./$(PGM) -i input2

vrun: $(PGM)
	mkdir -p $(RUNDIR)
	cp $(PGM) input $(RUNDIR)
	cd $(RUNDIR); valgrind --tool=memcheck --leak-check=full ./$(PGM) -i input

prun: $(PGM)
	env CUDA_PROFILE=1 CUDA_PROFILE_LOG=./cudaProfile.log CUDA_PROFILE_CONFIG=./config.txt $(MPIRUNCMD) ./$(PGM) -i input
	./anallog ./cudaProfile.log | sort -n
	# ./$(PGM) -i input

fprun: $(PGM)
	env CUDA_PROFILE=1 CUDA_PROFILE_LOG=./cudaProfile.log CUDA_PROFILE_CONFIG=./config.txt ./$(PGM) -i input
	./anallog ./cudaProfile.log
	# ./$(PGM) -i input

frun: $(PGM)
	./$(PGM) -i input
	# ./$(PGM) -i input

lrun: $(PGM)
	 $(MPIRUNCMD) ./$(PGM) -i input
	# ./$(PGM) -i input

gdb: $(PGM)
	mkdir -p $(RUNDIR)
	cp $(PGM) input $(RUNDIR)
	cd $(RUNDIR); gdb ./$(PGM)

ptmp: $(PGM)
	mkdir -p /ptmp/ocre/coling/cuHydroC
	cp $(PGM) input  /ptmp/ocre/coling/cuHydroC
	cd  /ptmp/ocre/coling/cuHydroC;  $(MPIRUNCMD) ./$(PGM) -i input

ddt: $(PGM)
	mkdir -p $(RUNDIR)
	cp $(PGM) input $(RUNDIR)
	cd $(RUNDIR); /opt/allinea/ddt-2.5.alpha/bin/ddt ./$(PGM) -i $(HOME)/HYDRO/Mono/Input/input_sedov_noio_20x40.nml

FORCE:

#EOF
