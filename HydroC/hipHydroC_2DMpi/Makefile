# Link make.inc with the targeted architecture in Arch directory
# For example : ln -s make_sp6 make.inc
#--------------------------------------
include make.inc

# OpenMP options
#---------------
CFLAGS += $(CFLAGS_MPI)
NVCFLAGS += $(NVCFLAGS_MPI)
LDFLAGS += $(LDFLAGS_MPI)

# Sources directories
#--------------------
SRCDIR = .

# Objects
#------------
OBJ = GetDevice.o SplitSurface.o cuCmpflx.o cuComputeDeltat.o cuConservar.o cuConstoprim.o cuEquationOfState.o cuHydroGodunov.o cuMakeBoundary.o cuQleftright.o cuRiemann.o cuSlope.o cuTrace.o gridfuncs.o hydro_funcs.o hydro_utils.o main.o parametres.o perfcnt.o utils.o vtkfile.o

# Sources files
#--------------
SRCC=SplitSurface.c hydro_funcs.c hydro_utils.c main.c parametres.c utils.c vtkfile.c

SRCCU=GetDevice.cu cuComputeDeltat.cu cuConservar.cu cuConstoprim.cu cuEquationOfState.cu cuHydroGodunov.cu cuMakeBoundary.cu cuQleftright.cu cuRiemann.cu cuSlope.cu cuTrace.cu perfcnt.cu cuCmpflx.cu gridfuncs.cu

SRC=$(SRCC) $(SRCCU) 

DEPS=$(patsubst %.c,%.d,$(SRCC)) $(patsubst %.cu,%.d,$(SRCCU))

hydro:  $(OBJ)
	$(NVCC) $(LDFLAGS) $(OBJ) -o hydro

clean:
	rm -f *.o *~ 

cleanall:
	rm -f *.o *~ hydro

# Dependencies
#-------------
include $(DEPS)

%.d: %.c
	@echo "Generating $@"
	gcc -M $(DEFS) $(CFLAGS) $< | sed 's+ + \\\n+g' | egrep -v '/usr|/ccc/produc|/cea/prod' | egrep -v '^.\\$$' | egrep -v '^\\$$' > $@
%.d: %.cu
	@echo "Generating $@"
	hipcc -M $(DEFS) $(CFLAGS) $< | sed 's+ + \\\n+g' | egrep -v '/usr|/ccc/produc|/cea/prod' | egrep -v '^.\\$$' | egrep -v '^\\$$' > $@

%.o: $(SRCDIR)/%.c
	$(CC) $(CFLAGS) -c $<

%.o: $(SRCDIR)/%.cu
	$(NVCC) $(NVCFLAGS) -c $<

run: hydro
	ccc_mprun -p hybridk40 -E-JcuHydroC ./hydro -i input.nml

burn: hydro
	ccc_mprun -E-JcuBurnK40 -p hybridk40 -n6 -N3 -T36000 ./hydro -i inputMEDIUM.nml

runa100: hydro
	ccc_mprun -p a100 -c 32 -n 1 -x ./hydro -i input.nml
