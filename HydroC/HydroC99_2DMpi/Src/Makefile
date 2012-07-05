# Link make.inc with the targeted architecture in Arch directory
# For example : ln -s make_sp6 make.inc
#--------------------------------------
include ../../../Arch/make.inc

# OpenMP options
#---------------
CFLAGS += $(CFLAGS_MPI)
CFLAGS += $(CFLAGS_OMP)
LDFLAGS += $(LDFLAGS_MPI)
LDFLAGS += $(LDFLAGS_OMP)

# Sources directories
#--------------------
SRCDIR = .

# Objects
#------------
OBJ = SplitSurface.o cmpflx.o compute_deltat.o conservar.o constoprim.o equation_of_state.o hydro_funcs.o hydro_godunov.o hydro_utils.o main.o make_boundary.o parametres.o perfcnt.o qleftright.o riemann.o slope.o trace.o utils.o vtkfile.o 

# Sources files
#--------------
SRC=SplitSurface.c cmpflx.c compute_deltat.c conservar.c constoprim.c equation_of_state.c hydro_funcs.c hydro_godunov.c hydro_utils.c main.c make_boundary.c parametres.c perfcnt.c qleftright.c riemann.c slope.c trace.c utils.c vtkfile.c 


hydro:  $(OBJ)
	$(CC) $(LDFLAGS) $(OBJ) -o hydro

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
#	env CFLAGS='$(CFLAGS)' makedeps > alldeps.d
