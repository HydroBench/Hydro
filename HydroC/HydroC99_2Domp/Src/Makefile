CC=icx
ifeq ($(QUIET),O)
CFLAGS_OMP+= -W"no-\#pragma-messages" 
endif
# OpenMP options
#---------------
ifeq ($(GPU),O)
CFLAGS_OMP+=-DTARGETON
endif
ifeq ($(CC),icx)
#
ifeq ($(GPU),O)
CFLAGS_OMP+= -fiopenmp -fopenmp-targets=spir64
else
CFLAGS_OMP+= -fiopenmp
endif
CFLAGS_OMP+= -qopenmp-simd -pg -O3 -mavx2 -fno-color-diagnostics -fvectorize
endif
ifeq ($(CC),icc)
CFLAGS_OMP+= -fopenmp -qopenmp-simd -O3 -mavx2
endif
ifeq ($(CC),gcc)
# CFLAGS_OMP=-fopenmp -fopenmp-simd -O3 -mavx2 -DTARGET=1 -foffload=-mavx2 -flto
CFLAGS_OMP+= -fopenmp -fopenmp-simd -O3 -mavx2 -flto
endif
ifeq ($(CC),clang)
CFLAGS_OMP+= -fopenmp -fopenmp-simd -O3 -mavx2 -fno-color-diagnostics
endif
ifeq ($(CC),/usr/bin/clang)
CFLAGS_OMP+= -fopenmp -fopenmp-simd -O3 -mavx2 -fno-color-diagnostics
endif

CFLAGS += $(CFLAGS_MPI) $(DEF)
CFLAGS += $(CFLAGS_OMP) -g 
CFLAGS += $(CFLAGS_FTI)
# CFLAGS += -vec-report3
LDFLAGS += $(LDFLAGS_MPI)
LDFLAGS += $(LDFLAGS_OMP)
LDFLAGS += $(LDFLAGS_FTI)
LDFLAGS += -lrt

# Sources directories
#--------------------
SRCDIR = .

# Objects
#------------
# OBJ = SplitSurface.o cmpflx.o compute_deltat.o conservar.o constoprim.o equation_of_state.o hydro_funcs.o hydro_godunov.o hydro_utils.o main.o make_boundary.o parametres.o perfcnt.o qleftright.o riemann.o slope.o trace.o utils.o vtkfile.o hydro_numa.o cclock.o

# Sources files
#--------------
SRC=SplitSurface.c hydro_funcs.c hydro_godunov.c hydro_utils.c main.c parametres.c perfcnt.c utils.c vtkfile.c cclock.c
# SRC2 are the files that can be translated to CUDA or OpenCL
SRC2=cmpflx.c compute_deltat.c conservar.c constoprim.c equation_of_state.c make_boundary.c qleftright.c riemann.c slope.c trace.c 
SRC+=$(SRC2)

DEPS=$(patsubst %.c,%.d,$(SRC))
OBJ=$(patsubst %.c,%.o,$(SRC))

# HEADER=$(wildcard *.h)

ifeq ($(PREC),SP)
CFLAGS += -DPREC_SP=1
endif

hydro:  $(OBJ)
	$(CC) $(CFLAGS) $(OBJ) -o hydro $(LDFLAGS) -lm
	$(CC) --version

clean:
	rm -f *.o *~ hydro *.d

cleanall:
	rm -f *.o *~ hydro

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
	sh -c "if [ -e lastgpu ] ; then rm -f $(TARGETFUNCS) lastgpu; fi"
	make -j GPU=N
	env OMP_NUM_THREADS=8 ./hydro -i input.nml
	touch lastcpu

rungpu:
	if [ -e lastcpu ] ; then rm -f $(TARGETFUNCS) lastcpu; fi
	make -j GPU=O QUIET=O
	env OMP_NUM_THREADS=1 LIBOMPTARGET_PLUGIN=OPENCL LIBOMPTARGET_PROFILE=T,msec LIBOMPTARGET_DEBUG=0 ./hydro -i input.nml
	touch lastgpu

rungpud: hydro
	rm -f $(TARGETFUNCS)
	make GPU=O QUIET=O
	env OMP_NUM_THREADS=1 LIBOMPTARGET_PLUGIN=OPENCL LIBOMPTARGET_PROFILE=T,msec LIBOMPTARGET_DEBUG=5 ./hydro -i input.nml

run8: hydro
	env OMP_NUM_THREADS=8 LIBOMPTARGET_PLUGIN=OPENCL LIBOMPTARGET_PROFILE=T,msec LIBOMPTARGET_DEBUG=1 ./hydro -i input.nml

DATEHOUR=$(shell date +%Y%m%d%H%M )
tarball:
	make clean CC=gcc
	tar czvf tarball_$(DATEHOUR).tgz Makefile *.h *.c input.nml

runall:
	make runcpu | tail -7
	make rungpu | tail -7
#EOF
