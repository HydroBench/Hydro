# Link make.inc with the targeted architecture in Arch directory
# For example : ln -s make_sp6 make.inc
#--------------------------------------
include make.inc

# OpenMP options
#---------------
CFLAGS += $(CFLAGS_MPI) $(DEF)
CFLAGS += $(CFLAGS_OMP) -g
CFLAGS += $(CFLAGS_FTI)
CFLAGS += -vec-report3
LDFLAGS += $(LDFLAGS_MPI)
LDFLAGS += $(LDFLAGS_OMP)
LDFLAGS += $(LDFLAGS_FTI)
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

# HEADER=$(wildcard *.h)

ifeq ($(PREC),SP)
CFLAGS += -DPREC_SP=1
endif

hydro:  $(OBJ)
	$(CC) $(OBJ) -o hydro $(LDFLAGS) 

clean:
	rm -f *.o *~ hydro

cleanall:
	rm -f *.o *~ hydro

listing:
	a2ps --medium=A4dj -C -1 -R -l90 --toc -E --prologue=color --file-align=fill -o listing.ps Makefile $(HEADER) $(SRC)
	ps2pdf -sPAPERSIZE=a4 listing.ps
	rm listing.ps

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
#	env CFLAGS='$(CFLAGS)' ./makedeps > alldeps.d

vg: hydro
	ccc_mprun -x -T36000 -c1 -n1 valgrind --tool=callgrind --main-stacksize=35000000 --trace-jump=yes --trace-children=yes --branch-sim=yes --cache-sim=yes --callgrind-out-file=toto ./hydro -i input.nml
	callgrind_annotate  toto qleftright.c

hugerun: hydro
	env OMP_NUM_THREADS=4 ccc_mprun -x -T36000 -c4 -n1 env HUGETLB_MORECORE=yes LD_PRELOAD=/usr/lib64/libhugetlbfs.so hydro -i input.nml

run: hydro
	env OMP_NUM_THREADS=8 ccc_mprun -p kncn -x -T360 -c8 -n1 env KMP_AFFINITY=compact hydro -i input.nml

brun16: hydro
	ccc_mprun -p hybrid -x -T360 -c16 -n1 env OMP_NUM_THREADS=16 KMP_AFFINITY=compact hydro -i input.nml

lrun: hydro
	env OMP_NUM_THREADS=16 KMP_AFFINITY=compact ./hydro -i input.nml

prun: hydro
	ccc_mprun -p knc -N4 -n16 ./hydro -i input.nml

lrunhuge: hydro
	env OMP_NUM_THREADS=8 KMP_AFFINITY=compact HUGETLB_MORECORE=yes LD_PRELOAD=/usr/lib64/libhugetlbfs.so ./hydro -i input.nml

urun: hydro
	env OMP_NUM_THREADS=1 KMP_AFFINITY=scatter ./hydro -i input.nml

erun: hydro
	env OMP_NUM_THREADS=8 KMP_AFFINITY=compact ./hydro -i input.nml

rmic: hydro
	ccc_mprun -p knc -n 1 -x env LD_PRELOAD=/lib64/libhugetlbfs.so OMP_NUM_THREADS=240 ./hydro -i input.nml
hugemic: hydro
	ssh $(HOSTNAME)-mic0 "cd $(PWD); export LD_PRELOAD=/lib64/libhugetlbfs.so ; env OMP_NUM_THREADS=240 KMP_AFFINITY=compact HUGETLB_MORECORE=yes mpiexec -np 1 ./hydro -i input.nml" 
mmic: hydro
	ssh $(HOSTNAME)-mic1 "cd $(PWD); env OMP_NUM_THREADS=60 KMP_AFFINITY=compact mpiexec -np 4 ./hydro -i input.nml" 
fmic: hydro
	ssh $(HOSTNAME)-mic0 "cd $(PWD); env OMP_NUM_THREADS=8 KMP_AFFINITY=compact mpiexec -np 30 ./hydro -i input.nml" 

rscan: hydro
	for i in `seq 10 10 244` ; do ssh $(HOSTNAME)-mic0 "cd $(PWD); env OMP_NUM_THREADS=$$i KMP_AFFINITY=compact mpiexec -np 1 ./hydro -i input.nml " ; done 

maq: hydro
	rm -rf toto maqao_funcs.lst maqao_loopslist.lst
	env OMP_NUM_THREADS=1 ccc_mprun -M 1024 -x -p knc maqao perf -mIOMP -bin="hydro" -t=SX -op=toto -g=small -uarch=SANDY_BRIDGE -- -i input.nml | tee maqao_run.lst
	maqao perf -d=SFX -bin='hydro' -uarch=SANDY_BRIDGE -op=toto | tee maqao_funcs.lst		
	maqao perf -d=SLX -bin='hydro' -uarch=SANDY_BRIDGE -op=toto | tee maqao_loopslist.lst	
	rm -f maqao_loopsdetail.lst
	for i in `deploop.pl < maqao_loopslist.lst | sort -n | uniq ` ; do maqao analyze -s -bin='hydro' -uarch=SANDY_BRIDGE -loop=$$i >> maqao_loopsdetail.lst; done
	for i in `deploop.pl < maqao_loopslist.lst | sort -n | uniq ` ; do maqao analyze -s -bin='hydro' -uarch=SANDY_BRIDGE -loop=$$i -lvl=1 >> maqao_loopsdetail.lst; done

natif: hydro
	ccc_mprun -K -p knc -n 1 env OMP_NUM_THREADS=240 ccc_mprun hydro -i input.nml


snbhsw: hydro
	-ccc_mprun -p knc     -T 7200 -n 1 env OMP_NUM_THREADS=1 hydro -i input.nml
	-ccc_mprun -p haswell -T 7200 -n 1 env OMP_NUM_THREADS=1 hydro -i input.nml

scanopt: 
	echo "Opt    GATCON    CONPRI    EOS       SLOPE     TRACE     QLEFTR    RIEMAN    CMPFLX    UPDCON    COMPDT    MAKBOU    ALLRED  " > ScanOpts.lst
	for s in 1 23 32 ; do for n in 256 512 1024 2048 4096 5647; do sed "s/NX/$$n/;s/NS/$$s/" < inputMASK.nml > inputCUR.nml; echo "$${n}_$${s}" >> ScanOpts.lst; echo "Opt    GATCON    CONPRI    EOS       SLOPE     TRACE     QLEFTR    RIEMAN    CMPFLX    UPDCON    COMPDT    MAKBOU    ALLRED  " >> ScanOpts.lst; for p in SP DP ; do for i in NVC SE4 AVX AV2 ; do echo $$i; make PREC=$$p MMACH=$$i clean hydro >> compile.lst; ccc_mprun -x -p haswell -n 1 env OMP_NUM_THREADS=1 hydro -i inputCUR.nml | grep 'PE0' | sed "s/PE0/$$i/" >> ScanOpts.lst ; done; echo >> ScanOpts.lst; done; done; done

DATEHOUR=$(shell date +%Y%m%d%H%M )
Andres:
	tar czvf Andres_$(DATEHOUR).tgz Makefile *.h *.c input.nml make* alldeps.d

#EOF
