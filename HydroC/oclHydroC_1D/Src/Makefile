# Link make.inc with the targeted architecture in Arch directory
# For example : ln -s ../../../Arch/make_sp6 make.inc
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
OBJ=cmpflx.o compute_deltat.o conservar.o constoprim.o equation_of_state.o hydro_funcs.o hydro_godunov.o hydro_utils.o main.o make_boundary.o oclCmpflx.o oclComputeDeltat.o oclConservar.o oclConstoprim.o oclEquationOfState.o oclHydroGodunov.o oclInit.o oclMakeBoundary.o oclQleftright.o oclReduce.o oclRiemann.o oclSlope.o oclTrace.o oclerror.o ocltools.o parametres.o qleftright.o riemann.o slope.o trace.o utils.o vtkfile.o

# Sources files
#--------------
SRC=cmpflx.c compute_deltat.c conservar.c constoprim.c equation_of_state.c hydro_funcs.c hydro_godunov.c hydro_utils.c main.c make_boundary.c oclCmpflx.c oclComputeDeltat.c oclConservar.c oclConstoprim.c oclEquationOfState.c oclHydroGodunov.c oclInit.c oclMakeBoundary.c oclQleftright.c oclReduce.c oclRiemann.c oclSlope.c oclTrace.c oclerror.c ocltools.c parametres.c qleftright.c riemann.c slope.c trace.c utils.c vtkfile.c



hydro:  $(OBJ)
	$(CC) $(OBJ) -o hydro $(LDFLAGS) 

clean:
	rm -f *.o *~

cleanall:
	rm -f *.o *~ hydro

%.o: $(SRCDIR)/%.c
	$(CC) $(CFLAGS) -c $<

%.o: $(SRCDIR)/%.cu
	$(CC) $(NVCFLAGS) -c $<

# Dependencies
#-------------
include alldeps.d

# To make sure all dependences are correct, uncomment the next two
# lines and do a "make alldeps.d" to refresh the list. This command
# takes some time so I have not left it active. It is really useful
# only if you have added new files (either .c .cu or .h)

alldeps.d: $(SRC)
	env CFLAGS='$(CFLAGS)' NVCFLAGS='$(NVCFLAGS)' ./makedeps > alldeps.d
