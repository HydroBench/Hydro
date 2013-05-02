# Link make.inc with the targeted architecture in Arch directory
# For example : ln -s make_sp6 make.inc
#--------------------------------------
include make.inc

# OpenMP options
#---------------
CFLAGS += $(CFLAGS_MPI) $(DEF)
CFLAGS += $(CFLAGS_OMP)
CFLAGS += -vec-report3
LDFLAGS += $(LDFLAGS_MPI)
LDFLAGS += $(LDFLAGS_OMP)
LDFLAGS += -lrt

# Sources directories
#--------------------
SRCDIR = .

# Objects
#------------
OBJ = SplitSurface.o cmpflx.o compute_deltat.o conservar.o constoprim.o equation_of_state.o hydro_funcs.o hydro_godunov.o hydro_utils.o main.o make_boundary.o parametres.o perfcnt.o qleftright.o riemann.o slope.o trace.o utils.o vtkfile.o hydro_numa.o cclock.o

# Sources files
#--------------
SRC=SplitSurface.c hydro_funcs.c hydro_numa.c hydro_godunov.c hydro_utils.c main.c parametres.c perfcnt.c utils.c vtkfile.c cclock.c
# SRC2 are the files that can be translated to CUDA or OpenCL
SRC2=cmpflx.c compute_deltat.c conservar.c constoprim.c equation_of_state.c make_boundary.c qleftright.c riemann.c slope.c trace.c 
SRC+=$(SRC2)

DEPS=$(subst .c,.d,$(SRC))

# HEADER=$(wildcard *.h)

hydro:  $(OBJ)
	$(CC) $(OBJ) -o hydro $(LDFLAGS) 

clean:
	rm -f *.o *~ hydro

cleanall:
	rm -f *.o *~ hydro

listing:
	a2ps --medium=A4dj -C -1 -R -l90 --toc -E --prologue=color --file-align=fill -o listing.ps Makefile $(HEADER) $(SRC)
	ps2pdf listing.ps
	rm listing.ps

%.o: $(SRCDIR)/%.c
	$(CC) $(CFLAGS) -c $<

%.o: $(SRCDIR)/%.cu
	$(NVCC) $(NVCFLAGS) -c $<

%.d: %.c
	@echo Generating $@
	@$(CC) $(CFLAGS) -M $< > $@


# Dependencies
#-------------
include $(DEPS)

# To make sure all dependences are correct, uncomment the next two
# lines and do a "make alldeps.d" to refresh the list. This command
# takes some time so I have not left it active. It is really useful
# only if you have added new files (either .c .cu or .h)

vg: hydro
	ccc_mprun -x -T36000 -c1 -n1 valgrind --tool=callgrind --main-stacksize=35000000 --trace-jump=yes --trace-children=yes --branch-sim=yes --cache-sim=yes --callgrind-out-file=toto ./hydro -i input.nml
	callgrind_annotate  toto qleftright.c
lvg: hydro
	valgrind --tool=memcheck ./hydro -i input.nml

	callgrind_annotate  toto qleftright.c

hugerun: hydro
	env OMP_NUM_THREADS=4 ccc_mprun -x -T36000 -c4 -n1 env HUGETLB_MORECORE=yes LD_PRELOAD=/usr/lib64/libhugetlbfs.so hydro -i input.nml

run: hydro
	env OMP_NUM_THREADS=8 ccc_mprun -x -T360 -c8 -n1 env KMP_AFFINITY=compact hydro -i input.nml

lrun: hydro
	env OMP_NUM_THREADS=32 KMP_AFFINITY=compact ./hydro -i input.nml

lrunhuge: hydro
	env OMP_NUM_THREADS=8 KMP_AFFINITY=compact HUGETLB_MORECORE=yes LD_PRELOAD=/usr/lib64/libhugetlbfs.so ./hydro -i input.nml

urun: hydro
	env OMP_NUM_THREADS=1 KMP_AFFINITY=scatter ./hydro -i input.nml

erun: hydro
	env OMP_NUM_THREADS=8 KMP_AFFINITY=compact ./hydro -i input.nml

rmic: hydro
	ssh $(HOSTNAME)-mic0 "cd $(PWD); env OMP_NUM_THREADS=240 KMP_AFFINITY=compact mpiexec -np 1 ./hydro -i input.nml" 

rscan: hydro
	for i in `seq 10 10 244` ; do ssh $(HOSTNAME)-mic0 "cd $(PWD); env OMP_NUM_THREADS=$$i KMP_AFFINITY=compact mpiexec -np 1 ./hydro -i input.nml " ; done 
hugemic: hydro
	ssh $(HOSTNAME)-mic0 "cd $(PWD); export LD_PRELOAD=/lib64/libhugetlbfs.so ; env OMP_NUM_THREADS=240 KMP_AFFINITY=compact HUGETLB_MORECORE=yes mpiexec -np 1 ./hydro -i input.nml" 
mmic: hydro
	ssh $(HOSTNAME)-mic1 "cd $(PWD); env OMP_NUM_THREADS=60 KMP_AFFINITY=compact mpiexec -np 4 ./hydro -i input.nml" 
fmic: hydro
	ssh $(HOSTNAME)-mic0 "cd $(PWD); env OMP_NUM_THREADS=8 KMP_AFFINITY=compact mpiexec -np 30 ./hydro -i input.nml" 

cirrus: hydro
	ccc_mprun -p knca -N4 -n8 -c4 env OMP_NUM_THREADS=4 ./hydro -i input.nml

FORCE:
