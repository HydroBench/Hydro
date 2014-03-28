SRCS=TimeLimit.cpp Utilities.cpp Matrix.cpp Soa.cpp ThreadBuffers.cpp Tile.cpp Domain.cpp Domain_Compute.cpp Domain_Boundary.cpp Domain_Prot.cpp Domain_VTK.cpp Domain_Image.cpp Domain_TestCase.cpp main.cpp 
DEPS=$(patsubst %.cpp,%.d,$(SRCS))
OBJS=$(patsubst %.cpp,%.o,$(SRCS)) cclock.o

ifeq ($(MPI),O)
ifeq ($(MIC),O)
LIBS+=-L$(I_MPI_ROOT)/mic -lmpi
else
LIBS+=-L$(I_MPI_ROOT)/lib64 -lmpi
endif
INCS+=-I$(I_MPI_ROOT)/intel64/include
DEFS+=-DMPI_ON $(INCS)
endif
ifneq ($(MIC),O)
ifeq ($(PNG),O)
DEFS+=-D WITHPNG=1
endif
endif

CXX=icpc

CFLAGS=$(DEFS)
CFLAGS+=-g

OPTIM+=-O3
ifeq ($(CXX),icpc)
OPTIM+=-mkl
OPTIM+=-openmp
OPTIM+=-ansi-alias 
ifeq ($(MIC),O)
OPTIM+=-mmic -opt-prefetch=2
else
ifeq ($(PG),O)
OPTIM+=-pg
endif
ifeq ($(XHOST),O)
OPTIM+=-xHost 
endif
ifeq ($(AVX),O)
OPTIM+=-mavx -DAVX
endif
ifeq ($(SSE),O)
OPTIM+=-msse4 -DSSE
endif
ifeq ($(AVX2),O)
OPTIM+=-march=core-avx2 -DAVX
endif
ifeq ($(ATOM),O)
OPTIM+=-xATOM_SSE4.2
endif
endif
ifeq ($(VERB),O)
OPTIM+=-vec-report1
endif
# OPTIM+=-openmp-report1 
ifeq ($(NOVEC),O)
OPTIM+=-no-vec
endif
# OPTIM+=-parallel -guide
else
OPTIM+=-fopenmp
ifeq ($(NOVEC),O)
OPTIM+=-fno-tree-vectorize
else
OPTIM+=-ftree-loop-distribution -ftree-loop-im -ftree-loop-ivcanon -ftree-vectorize -ftree-vect-loop-version -ftracer -freciprocal-math -mrecip -funroll-loops --param max-cse-path-length=100 --param max-cse-path-length=10000 --param loop-invariant-max-bbs-in-loop=100000
ifeq ($(VERB),O)
OPTIM+=-ftree-vectorizer-verbose=1
endif
endif
# OPTIM+=-mavx
# OPTIM+=-pg
endif
# OPTIM+=-parallel -guide
ifeq ($(PNG),O)
LIBS+=-lpng 
endif

hydroc: $(OBJS)
	$(CXX) -o hydroc $(CFLAGS) $(OPTIM) $(OBJS) $(LIBS) -lm -lrt
ifeq ($(PG),O)
	touch .waspg
else
	rm -f .waspg
endif
ifeq ($(MIC),O)
	touch .wasmic
else
	rm -f .wasmic
endif


include $(DEPS)

%.o: %.cpp
	$(CXX) -o $@ -c $(CFLAGS) $(OPTIM) $< 

%.o: %.c
	$(CXX) -o $@ -c $(CFLAGS) $(OPTIM) $< 

%.d: %.cpp
	$(CXX) -M $< $(DEFS)  | sed 's+ + \\\n+g' | egrep -v '/usr|/ccc/produc|/cea/prod' | egrep -v '^.\\$$' | egrep -v '^\\$$' > $@

clean:
	rm -rf *.o hydroc *~ toto

DATEHOUR=$(shell date +%Y%m%d%H%M )
Archiv:
	tar czvf Archiv_$(DATEHOUR).tgz Makefile *.hpp *.cpp *.h *.c input.nml

indent:
	indent *.cpp *.hpp *.c *.h

#
# - - - section to launch runs easily through a songle make command. They should be tailored to YOUR needs - - -
#

run: 
	if [ -f .wasmic ] ; then make clean ; fi
	if [ -f .waspg ] ; then make clean ; fi
	make hydroc
	rm -rf Dep/* Image.png
	env OMP_NUM_THREADS= time ccc_mprun -p hybridk40 -T 6400 ./hydroc -i input.nml
	if [ -f Image.png ] ; then display Image.png; fi

grun: 
	if [ -f .wasmic ] ; then make clean ; fi
	if [ -f .waspg ] ; then make clean ; fi
	make hydroc
	rm -rf Dep/* Image.png
	env OMP_NUM_THREADS=12 time ccc_mprun -n16 -N8 -p hybridk40 -T 6400 gdb -x runcmd ./hydroc
	if [ -f Image.png ] ; then display Image.png; fi

runk40: 
	if [ -f .wasmic ] ; then make clean ; fi
	if [ -f .waspg ] ; then make clean ; fi
	make hydroc
	rm -rf Dep/* Image.png
	env OMP_NUM_THREADS=4 time ccc_mprun -p hybridk40 -T 6400 ./hydroc -i input.nml
	-display Image.png

runatom: 
	if [ -f .wasmic ] ; then make clean ; fi
	if [ -f .waspg ] ; then make clean ; fi
	make hydroc
	rm -rf Dep/*
	env OMP_NUM_THREADS=8 time ccc_mprun -p atom -T 6400 ./hydroc -i input.nml
	-display Image.png

haswell: 
	if [ -f .wasmic ] ; then make clean ; fi
	if [ -f .waspg ] ; then make clean ; fi
	make clean AVX=O hydroc
	rm -rf Dep/*
	env OMP_NUM_THREADS=4 ccc_mprun -p haswell -T 6400 ./hydroc -i input.nml
	make clean AVX2=O hydroc
	rm -rf Dep/*
	env OMP_NUM_THREADS=4 ccc_mprun -p haswell -T 6400 ./hydroc -i input.nml
	# ccc_mprun -p haswell maqao perf -d=SX -uarch=HASWELL -xp=toto -- hydroc
	make clean NOVEC=O hydroc
	rm -rf Dep/*
	env OMP_NUM_THREADS=4 ccc_mprun -p haswell -T 6400 ./hydroc -i input.nml

run16: 
	if [ -f .wasmic ] ; then make clean ; fi
	if [ -f .waspg ] ; then make clean ; fi
	make hydroc
	rm -rf Dep/*
	env OMP_NUM_THREADS=16 time ./hydroc -i input.nml
	-display Image.png

brun16: 
	if [ -f .wasmic ] ; then make clean ; fi
	if [ -f .waspg ] ; then make clean ; fi
	make hydroc
	rm -rf Dep/*
	ccc_mprun -p hybrid -N1 -n1 -c16 env OMP_NUM_THREADS=16 time ./hydroc -i input.nml

big: 
	if [ -f .wasmic ] ; then make clean ; fi
	if [ -f .waspg ] ; then make clean ; fi
	make hydroc
	rm -rf Dep/*
	ccc_mprun -p standard -N40 -n80 -c4 env OMP_NUM_THREADS=4 time ./hydroc -i big.nml
pgrun: 
	if [ ! -f .waspg ] ; then make clean ; fi
	make PG=O hydroc
	rm -rf Dep/*
	env OMP_NUM_THREADS=1 ccc_mprun -p knc ./hydroc -i input.nml
	gprof hydroc gmon.out | head -40 | cut -c '1-100'

vrun: hydroc
	rm -rf Dep/*
	env OMP_NUM_THREADS=8 time ccc_mprun -p hybridk40 -T 6400 valgrind --tool=memcheck --leak-check=full --show-reachable=yes --track-origins=no --gen-suppressions=no --suppressions=./valgrind.supp ./hydroc -i input.nml

prun: hydroc
	rm -rf Dep/*
	-rm STOP
	ccc_mprun -p hybridk40 -n16 -N8 ./hydroc -i input.nml
# -display Image.png

verybig: hydroc
	rm -rf Dep/*
	ccc_mprun -p standard -T 57600 -n 128 -c4  env OMP_NUM_THREADS=4 ./hydroc -i verybig.nml
	# ccc_mprun -p standard  -T 57600 -n 600  env OMP_NUM_THREADS=1 gdb -x bigcmd ./hydroc
	# ccc_mprun -p standard  -T 57600 -n 8 -c 4 env OMP_NUM_THREADS=4 valgrind --tool=memcheck ./hydroc -i verybig.nml
	display Image.png

rmic: 
	if [ ! -f .wasmic ] ; then make clean ; fi
	make MIC=O hydroc
	# ccc_mprun -p knc -n 1 env OMP_NUM_THREADS=240 KMP_AFFINITY=balanced OMP_WAIT_POLICY=passive ./hydroc -i input.nml
	ccc_mprun -p knc -n 1 -N 1 env OMP_NUM_THREADS=240 KMP_AFFINITY=balanced OMP_WAIT_POLICY=passive ./hydroc -i input.nml
	# KMP_AFFINITY=scatter ou compact ou balanced

prmic: 
	if [ ! -f .wasmic ] ; then make clean ; fi
	make MIC=O hydroc
	ccc_mprun -p knc -n2 -N1 env OMP_NUM_THREADS=120 KMP_AFFINITY=balanced OMP_WAIT_POLICY=passive mpiexec -n2 ./hydroc -i input.nml

rmicg: hydroc
	ccc_mprun -p knc -n 1 gdb -x runcmd ./hydroc 

scan: hydroc
	for i in 1 2 4 8 16 32 ; do env OMP_NUM_THREADS=$$i ./hydroc -i input.nml | tail -1 ; done

mscan: hydroc
	for i in 1 2 4 8 16 32 64 128 240 ; do env OMP_NUM_THREADS=$$i ccc_mprun -p knc ./hydroc -i input.nml | tail -2 ; done

lscan: hydroc
	for i in 1 2 4 8 16 32; do env OMP_NUM_THREADS=$$i ./hydroc -i input.nml | tail -2 ; done

prof: hydroc
	./hydroc
	gprof hydroc gmon.out | head -20

maq: hydroc
	if [ -f .wasmic ] ; then make clean ; make hydroc ; fi
	rm -rf toto/*
	env OMP_NUM_THREADS=1 ccc_mprun -p hybrid maqao perf -t=SX -xp=toto -g=small -uarch=SANDY_BRIDGE -- hydroc -i input.nml | tee maqao_run.lst	
	rm -rf loops.csv maqao_loopslist.lst maqao_loopsdetail.lst maqao_loopsdetailLVL1.lst
	maqao --version
	#maqao perf -d=SX  -uarch=SANDY_BRIDGE -xp=toto -oformat=html -- hydroc
	maqao perf -d=SFX -uarch=SANDY_BRIDGE -xp=toto -- hydroc
	maqao perf -d=SLX -uarch=SANDY_BRIDGE -xp=toto -- hydroc > maqao_loopslist.lst
	for i in `deploop.pl < maqao_loopslist.lst | sort -n | uniq ` ; do echo $$i ; maqao cqa -uarch=SANDY_BRIDGE -loop=$$i  -- hydroc >> maqao_loopsdetail.lst; done
	for i in `deploop.pl < maqao_loopslist.lst | sort -n | uniq ` ; do echo $$i ; maqao cqa -uarch=SANDY_BRIDGE -loop=$$i  -lvl=1 -- hydroc >> maqao_loopsdetailLVL1.lst; done

maqmicrep:
	maqaoi64phi --version
	ccc_mprun -p knc -n1 maqaophi perf -d=SX -uarch=KNC -xp=toto -oformat=html --system-map=/lib/firmware/mic/System.map -- hydroc
	maqaoi64phi perf -d=SFX -bin='hydroc' -uarch=KNC -xp=toto | tee maqao_funcs.lst		
	maqaoi64phi perf -d=SLX -bin='hydroc' -uarch=KNC -xp=toto | tee maqao_loopslist.lst	
	for i in `deploop.pl < maqao_loopslist.lst | sort -n | uniq ` ; do maqaoi64phi cqa -uarch=KNC -loop=$$i  -- hydroc >> maqao_loopsdetail.lst; done
	# for i in `deploop.pl < maqao_loopslist.lst | sort -n | uniq ` ; do maqaoi64phi cqa -uarch=KNC -loop=$$i -lvl=1 -- hydroc >> maqao_loopsdetail.lst; done

maqmic: 
	if [ ! -f .wasmic ] ; then make clean ; fi
	make MIC=O hydroc
	rm -rf toto/* maqao_funcs.lst maqao_loopslist.lst maqao_loopsdetail.lst
	env OMP_NUM_THREADS=240 KMP_AFFINITY=balanced OMP_WAIT_POLICY=passive ccc_mprun -p knc -n1 maqaophi perf -mIOMP -t=SX -xp=toto -g=small -uarch=KNC -- hydroc -i input.nml | tee maqao_run.lst
	# make maqmicrep

#
# - - - end of runs section - - -
#

listing:
	a2ps --medium=A4dj -C -1 -R -l90 --toc -E --prologue=color --file-align=fill -o listing.ps *.hpp *.cpp Makefile
	ps2pdf -sPAPERSIZE=a4 listing.ps
	rm listing.ps


FORCE:
