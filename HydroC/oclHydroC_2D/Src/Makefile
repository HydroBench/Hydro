# Link make.inc with the targeted architecture in Arch directory
# For example : ln -s ../../../Arch/make_sp6 make.inc
#--------------------------------------
include make.inc

# Optimization level wanted
#--------------------------
OPTFLAGS=-O3

# OpenMP options
#---------------
CFLAGS += $(CFLAGS_MPI) $(OPTFLAGS)
NVCFLAGS += $(NVCFLAGS_MPI) $(OPTFLAGS)
LDFLAGS += $(LDFLAGS_MPI) $(OPTFLAGS)
SYSLIB+=-lrt

# Sources directories
#--------------------
SRCDIR = .

# Objects
#------------
OBJ = SplitSurface.o hydro_funcs.o hydro_utils.o main.o oclCmpflx.o oclComputeDeltat.o oclConservar.o oclConstoprim.o oclEquationOfState.o oclHydroGodunov.o oclInit.o oclMakeBoundary.o oclQleftright.o oclReduce.o oclRiemann.o oclSlope.o oclTrace.o oclerror.o ocltools.o parametres.o utils.o vtkfile.o cclock.o getDevice.o
# Sources files
#--------------
SRC=SplitSurface.c hydro_funcs.c hydro_utils.c main.c oclCmpflx.c oclComputeDeltat.c oclConservar.c oclConstoprim.c oclEquationOfState.c oclHydroGodunov.c oclInit.c oclMakeBoundary.c oclQleftright.c oclReduce.c oclRiemann.c oclSlope.c oclTrace.c oclerror.c ocltools.c parametres.c utils.c vtkfile.c cclock.c getDevice.c
 


hydro:  $(OBJ)
	$(CC) $(LDFLAGS) $(OBJ) -o hydro $(SYSLIB)

clean:
	rm -f *.o *~

cleanall:
	rm -f *.o *~ hydro

%.o: $(SRCDIR)/%.c
	$(CC) $(CFLAGS) -c $<

listing:
	a2ps  --medium=A4dj -C -1 -R -l90 --toc -E --prologue=color --file-align=fill -o listing.ps Makefile GetDevice.h SplitSurface.h gridfuncs.h hydro_funcs.h hydro_utils.h oclCmpflx.h oclComputeDeltat.h oclConservar.h oclConstoprim.h oclEquationOfState.h oclHydroGodunov.h oclInit.h oclMakeBoundary.h oclQleftright.h oclReduce.h oclRiemann.h oclSlope.h oclTrace.h oclerror.h oclparam.h ocltools.h parametres.h perfcnt.h utils.h vtkfile.h SplitSurface.c hydro_funcs.c hydro_utils.c main.c oclCmpflx.c oclComputeDeltat.c oclConservar.c oclConstoprim.c oclEquationOfState.c oclHydroGodunov.c oclInit.c oclMakeBoundary.c oclQleftright.c oclReduce.c oclRiemann.c oclSlope.c oclTrace.c oclerror.c ocltools.c parametres.c utils.c hydro_kernels.cl 
	ps2pdf -sPAPERSIZE=a4 listing.ps
	rm listing.ps
book:
	a2ps  --medium=A4dj -C -=book -l90 --toc -E --prologue=color --file-align=fill main.c oclCmpflx.c oclComputeDeltat.c oclConservar.c oclConstoprim.c oclEquationOfState.c oclHydroGodunov.c oclInit.c oclMakeBoundary.c oclQleftright.c oclReduce.c oclRiemann.c oclSlope.c oclTrace.c parametres.c hydro_kernels.cl 
# rm listing.ps

# Dependencies
#-------------
include alldeps.d

# To make sure all dependences are correct, uncomment the next two
# lines and do a "make alldeps.d" to refresh the list. This command
# takes some time so I have not left it active. It is really useful
# only if you have added new files (either .c .cu or .h)

alldeps.d: $(SRC)
	env CFLAGS='$(CFLAGS)' NVCFLAGS='$(NVCFLAGS)' makedeps > alldeps.d

burn: hydro
	rm -rf ~/.nv	# to empty the stupid cache of compiled OpenCL...
	ccc_mprun -E-JoclBurnK40 -p k40 -n2 -N1 -T36000 ./hydro -i inputMEDIUM.nml -u g
burnh: hydro
	ccc_mprun -E-JoclBurnHSW -p haswellb16 -A rdintel -n35 -N35 -T36000 ./hydro -i inputMEDIUM.nml -u c

mic: hydro
	./hydro -i input.nml -u a

#EOF
