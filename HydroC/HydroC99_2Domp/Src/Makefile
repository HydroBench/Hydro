CC:=icx
MPI:=O
DEF:=
#defaut behavior

ifeq ($(MPI),O)
CFLAGS_MPI+=-D MPI
endif

# OpenMP options
#---------------
ifeq ($(GPU),O)
DEF+=-DTARGETON
endif

ifeq ($(TRACK),O)
DEF+=-DTRACKDATA
endif

ifeq ($(CC),icx)
ifeq ($(MPI),O)
override CC=mpiicc -cc=icx
endif
ifeq ($(GPU),O)
CFLAGS_OMP+= -fiopenmp -fopenmp-targets=spir64
else
CFLAGS_OMP+= -fiopenmp
endif
CFLAGS_OMP+= -qopenmp-simd -O3 -mavx2 -fvectorize
# -fno-color-diagnostics 
ifeq ($(QUIET),O)
CFLAGS_OMP+= -W"no-\#pragma-messages" 
endif
endif # icx

ifeq ($(CC),icc)
CFLAGS_OMP+= -fopenmp -qopenmp-simd -O3 -mavx2
ifeq ($(MPI),O)
override CC=mpiicc
endif
endif # icc

ifeq ($(CC),gcc)
# CFLAGS_OMP=-fopenmp -fopenmp-simd -O3 -mavx2 -DTARGET=1 -foffload=-mavx2 -flto
CFLAGS_OMP+= -fopenmp -fopenmp-simd -O3 -mavx2 -flto -foffload=-lm -foffload=-latomic
endif

ifeq ($(CC),clang)
CFLAGS_OMP+= -fopenmp -fopenmp-simd -O3 -mavx2 -fno-color-diagnostics
endif # clang

ifeq ($(CC),/usr/bin/clang)
CFLAGS_OMP+= -fopenmp -fopenmp-simd -O3 -mavx2 -fno-color-diagnostics
ifeq ($(MPI),O)
CFLAGS+=-DMPI
ifeq ($(I_MPI_ROOT),)
CFLAGS_MPI=-I$(I_MPI_ROOT)/include
LDFLAGS_MPI=-L$(I_MPI_ROOT)/lib -lmpi
else
CFLAGS_MPI=-I/usr/lib/x86_64-linux-gnu/openmpi/include
LDFLAGS_MPI=-L/usr/lib/x86_64-linux-gnu/openmpi/lib/lib -lmpi
endif
endif
endif # system clang

ifeq ($(CC),llvm11)
CFLAGS_OMP+= -fopenmp -fopenmp-simd -O3 -fopenmp-targets=nvptx64-nvidia-cuda 
override CC=clang 
endif # system clang

ifeq ($(CC),pgi)
override CC=pgcc
# -Minfo=all -Mpreprocess
CFLAGS +=-ta=tesla,cc70 -mp=gpu 
CFLAGS_OMP+= -fopenmp -O3
endif # pgi compiler

ifeq ($(SP),O)
DEF +=-DPREC_SP
endif

ifeq ($(LOOP),O)
DEF +=-DLOOPFORM
endif

CFLAGS += $(CFLAGS_MPI) $(DEF)
CFLAGS += $(CFLAGS_OMP) -g 
# CFLAGS += -vec-report3
LDFLAGS +=$(LDFLAGS_MPI)
LDFLAGS +=$(LDFLAGS_OMP)
LDFLAGS +=-lrt -lm

# Sources directories
#--------------------
SRCDIR = .

# Objects
#------------
# OBJ = SplitSurface.o cmpflx.o compute_deltat.o conservar.o constoprim.o equation_of_state.o hydro_funcs.o hydro_godunov.o hydro_utils.o main.o make_boundary.o parametres.o perfcnt.o qleftright.o riemann.o slope.o trace.o utils.o vtkfile.o hydro_numa.o cclock.o

# Sources files
#--------------
SRC=SplitSurface.c hydro_funcs.c hydro_godunov.c hydro_utils.c mpibounds.c main.c parametres.c perfcnt.c utils.c vtkfile.c cclock.c
# SRC2 are the files that can be offloaded
SRC2=courantOnXY.c ComputeQEforRow.c cmpflx.c compute_deltat.c conservar.c constoprim.c equation_of_state.c make_boundary.c qleftright.c riemann.c slope.c trace.c
SRC+=$(SRC2)

DEPS=$(patsubst %.c,%.d,$(SRC))
OBJ=$(patsubst %.c,%.o,$(SRC))

# HEADER=$(wildcard *.h)

ifeq ($(PREC),SP)
CFLAGS += -DPREC_SP=1
endif

hydro:  $(OBJ)
	$(CC) $(CFLAGS) $(OBJ) $(LDFLAGS) -o hydro 
ifeq ($(GPU),O)
	touch .lastgpu
else
	touch .lastcpu
endif

clean:
	rm -f *.o *~ hydro

cleanall:
	rm -f *.o *~ hydro *.d

listing:
	#a2ps --medium=A4dj -C -1 -R -l90 --toc -E --prologue=color --file-align=fill -o listing.ps Makefile $(HEADER) $(SRC)
	a2ps --medium=A4dj -C -l90 --toc -E --prologue=color --file-align=fill -o listing.ps Makefile $(HEADER) $(SRC)
	ps2pdf -sPAPERSIZE=a4 listing.ps
	rm listing.ps

# Dependencies
#-------------
include $(DEPS)

# -qopt-report=max 
%.o: $(SRCDIR)/%.c
	$(CC) $(CFLAGS) -c $<

%.d: %.c
	@echo "Generating $@"
	@gcc -M $(DEFS)  $< | sed 's+ + \\\n+g' | egrep -v '/usr|/ccc/produc|/cea/prod' | egrep -v '^.\\$$' | egrep -v '^\\$$' > $@

# To make sure all dependences are correct, uncomment the next two
# lines and do a "make alldeps.d" to refresh the list. This command
# takes some time so I have not left it active. It is really useful
# only if you have added new files (either .c .cu or .h)

#alldeps.d: $(SRC)
#	env CFLAGS='$(CFLAGS)' ./makedeps > alldeps.d

TARGETFUNCS=*.o

runcpu:
	@if [ -e .lastgpu ] ; then rm -f $(TARGETFUNCS) .lastgpu; fi
	make -j GPU=N
	env OMP_NUM_THREADS=8 ./hydro -i input.nml

rungpu:
	@if [ -e .lastcpu ] ; then rm -f $(TARGETFUNCS) .lastcpu; fi
	make -j GPU=O QUIET=O
	env OMP_NUM_THREADS=1 LIBOMPTARGET_PLUGIN=OPENCL LIBOMPTARGET_PROFILE=T,msec LIBOMPTARGET_DEBUG=0 ./hydro -i input.nml

rungpusp:
	@if [ -e .lastcpu ] ; then rm -f $(TARGETFUNCS) .lastcpu; fi
	make -j GPU=O QUIET=O SP=O
	env OMP_NUM_THREADS=1 LIBOMPTARGET_PLUGIN=OPENCL LIBOMPTARGET_PROFILE=T,msec LIBOMPTARGET_DEBUG=0 ./hydro -i input.nml

rungpu0:
	@if [ -e .lastcpu ] ; then rm -f $(TARGETFUNCS) .lastcpu; fi
	make -j GPU=O QUIET=O
	env OMP_NUM_THREADS=1 LIBOMPTARGET_PLUGIN=LEVEL0 LIBOMPTARGET_PROFILE=T,msec LIBOMPTARGET_DEBUG=0 ./hydro -i input.nml

rungpu0d:
	@if [ -e .lastcpu ] ; then rm -f $(TARGETFUNCS) .lastcpu; fi
	make -j GPU=O QUIET=O
	env OMP_NUM_THREADS=1 LIBOMPTARGET_PLUGIN=LEVEL0 LIBOMPTARGET_PROFILE=T,msec LIBOMPTARGET_DEBUG=5 ./hydro -i input.nml

rungpuq:
	@if [ -e .lastcpu ] ; then rm -f $(TARGETFUNCS) .lastcpu; fi
	make -j GPU=O QUIET=O
	env OMP_NUM_THREADS=1 LIBOMPTARGET_PLUGIN=OPENCL ./hydro -i input.nml

rungpud: 
	@if [ -e .lastcpu ] ; then rm -f $(TARGETFUNCS) .lastcpu; fi
	make -j GPU=O QUIET=O 
	env OMP_NUM_THREADS=1 LIBOMPTARGET_PLUGIN=OPENCL LIBOMPTARGET_PROFILE=T,msec LIBOMPTARGET_DEBUG=1 ./hydro -i input.nml

rungpud5: 
	@if [ -e .lastcpu ] ; then rm -f $(TARGETFUNCS) .lastcpu; fi
	make clean
	make -j GPU=O QUIET=O TRACK=O
	env OMP_NUM_THREADS=1 LIBOMPTARGET_PLUGIN=OPENCL LIBOMPTARGET_PROFILE=T,msec LIBOMPTARGET_DEBUG=5 ./hydro -i input.nml

run8: hydro
	env OMP_NUM_THREADS=8 LIBOMPTARGET_PLUGIN=OPENCL LIBOMPTARGET_PROFILE=T,msec LIBOMPTARGET_DEBUG=1 ./hydro -i input.nml

runpargpu:
	@if [ -e .lastcpu ] ; then rm -f $(TARGETFUNCS) .lastcpu; fi
	make -j GPU=O QUIET=O MPI=O hydro
	env OMP_NUM_THREADS=2 LIBOMPTARGET_PLUGIN=OPENCL LIBOMPTARGET_PROFILE=T,msec LIBOMPTARGET_DEBUG=0 mpirun -n 4 ./hydro -i input.nml

runpargpud:
	@if [ -e .lastcpu ] ; then rm -f $(TARGETFUNCS) .lastcpu; fi
	make -j GPU=O QUIET=O MPI=O hydro
	env OMP_NUM_THREADS=2 LIBOMPTARGET_PLUGIN=OPENCL LIBOMPTARGET_PROFILE=T,msec LIBOMPTARGET_DEBUG=1 mpirun -n 4 ./hydro -i input.nml

runpar:
	@if [ -e .lastgpu ] ; then rm -f $(TARGETFUNCS) .lastgpu; fi
	make -j GPU=N QUIET=O MPI=O hydro
	env OMP_NUM_THREADS=2 mpirun -n 4 ./hydro -i input.nml

DATEHOUR=$(shell date +%Y%m%d%H%M )
tarball:
	make clean CC=gcc
	tar czvf tarball_$(DATEHOUR).tgz Makefile *.h *.c input.nml

runall:
	make clean runcpu SP=N | tail -9
	make clean rungpuq SP=N | tail -9
	make clean runcpu SP=O | tail -9
	make clean rungpuq SP=O | tail -9

inti:
	ccc_mprun -p v100 -x -c12 -n1 env OMP_NUM_THREADS=12 ./hydro -i input.nml

intipar:
	ccc_mprun -p v100 -x -c12 -n4 env OMP_NUM_THREADS=12 ./hydro -i input.nml

#EOF
