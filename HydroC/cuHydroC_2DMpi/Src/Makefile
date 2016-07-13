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
SRC=GetDevice.cu SplitSurface.c cmpflx.c compute_deltat.c conservar.c constoprim.c cuCmpflx.cu cuComputeDeltat.cu cuConservar.cu cuConstoprim.cu cuEquationOfState.cu cuHydroGodunov.cu cuMakeBoundary.cu cuQleftright.cu cuRiemann.cu cuSlope.cu cuTrace.cu equation_of_state.c gridfuncs.cu hydro_funcs.c hydro_godunov.c hydro_utils.c main.c make_boundary.c parametres.c perfcnt.cu qleftright.c riemann.c slope.c trace.c utils.c vtkfile.c 


hydro:  $(OBJ)
	$(NVCC) $(LDFLAGS) $(OBJ) -o hydro

clean:
	rm -f *.o *~

cleanall:
	rm -f *.o *~ hydro

%.o: $(SRCDIR)/%.c
	$(CC) $(CFLAGS) -c $<

%.o: $(SRCDIR)/%.cu
	$(NVCC) $(NVCFLAGS) -c $<

# Dependencies
#-------------
include alldeps.d

# To make sure all dependences are correct, uncomment the next two
# lines and do a "make alldeps.d" to refresh the list. This command
# takes some time so I have not left it active. It is really useful
# only if you have added new files (either .c .cu or .h)

#alldeps.d: $(SRC)
#	env CFLAGS='$(CFLAGS)' NVCFLAGS='$(NVCFLAGS)' makedeps > alldeps.d


run: hydro
	ccc_mprun -p hybridk40 -E-JcuHydroC ./hydro -i input.nml

burn: hydro
	ccc_mprun -E-JcuBurnK40 -p hybridk40 -n6 -N3 -T36000 ./hydro -i inputMEDIUM.nml
