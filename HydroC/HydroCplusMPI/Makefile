SRCS=TimeLimit.cpp Utilities.cpp Matrix.cpp Soa.cpp ThreadBuffers.cpp Tile.cpp Domain.cpp Domain_Compute.cpp Domain_Boundary.cpp Domain_Prot.cpp Domain_VTK.cpp Domain_Image.cpp Domain_TestCase.cpp main.cpp 
DEPS=$(patsubst %.cpp,%.d,$(SRCS))
OBJS=$(patsubst %.cpp,%.o,$(SRCS)) cclock.o
OPENMP=O
PNG=O
MPI=O
HOST=$(shell uname -n | sed 's/[0-9]//g')
ARCHEXT=std

# INCS+=-I$(HOME)/vecadv/advisor_xe/include

CXX=icpc
DEFS=-std=c++11
DEFS+=$(DEFSEXT)

ifeq ($(ARM),O)
include make.arm
endif

ifeq ($(ATOM),O)
include make.atom
endif

ifeq ($(MIC),O)
include make.mic
endif

ifeq ($(SSE),O)
include make.sse
endif

ifeq ($(KNL),O)
include make.knl
endif

ifeq ($(AVX2),O)
include make.avx2
endif

ifeq ($(AVX),O)
include make.avx
endif

CFLAGS=$(DEFS)
CFLAGS+=-g


ifeq ($(CXX),icpc)
# OPTIM+=-mkl
  ifeq ($(OPENMP),O)
     OPTIM+=-qopenmp
  endif
  OPTIM+=-ansi-alias 
  ifeq ($(PG),O)
     OPTIM+=-pg
  endif
  ifeq ($(XHOST),O)
     OPTIM+=-xHost 
  endif
endif
ifeq ($(VERB),O)
# OPTIM+=-vec-report6
# for compiler >= 15
OPTIM+=-qopt-report=5
endif

# OPTIM+=-openmp-report1 
ifeq ($(NOVEC),O)
OPTIM+=-no-vec
endif

ifeq ($(CXX),g++)
 ifeq ($(AVX),O)
  ARCHEXT=avx
  OPTIM+=-mavx -DAVX
 endif
 ifeq ($(OPENMP),O)
  OPTIM+=-fopenmp
 endif
 ifeq ($(NOVEC),O)
  OPTIM+=-fno-tree-vectorize
 else
   OPTIM+=-ftree-loop-distribution -ftree-loop-im -ftree-loop-ivcanon -ftree-vectorize -ftree-vect-loop-version -ftracer -freciprocal-math -mrecip -funroll-loops --param max-cse-path-length=100 --param max-cse-path-length=10000 --param loop-invariant-max-bbs-in-loop=100000
 endif
 ifeq ($(VERB),O)
  OPTIM+=-ftree-vectorizer-verbose=1 -Wall
 endif
endif

ifeq ($(PNG),O)
LIBS+=-lpng 
endif

ifeq ($(HBM),O)
#OPTIM+=-DMEMKIND -DWITHHBW -I${MEMKINDDEV_INCDIR}
#LIBS+=-L${MEMKINDDEV_LIBDIR} -lmemkind
OPTIM+=-DMEMKIND -DWITHHBW 
LIBS+=-lmemkind
endif

CFLAGS+=$(INCS)

hydroc: $(OBJS)
	-$(CXX) --version
	$(CXX) -o hydroc $(CFLAGS) $(OPTIM) $(OBJS) $(LIBS) -lm -lrt -lpthread
# -lrt -ldl
	cp hydroc hydroc.$(ARCHEXT)

include $(DEPS)

%.o: %.cpp
	$(CXX) -o $@ -c $(CFLAGS) $(OPTIM) $< 

%.o: %.c
	$(CXX) -o $@ -c $(CFLAGS) $(OPTIM) $< 

%.d: %.cpp
	@$(CXX) -M $< $(DEFS) $(CFLAGS) | sed 's+ + \\\n+g' | egrep -v '/usr|/ccc/produc|/cea/prod' | egrep -v '^.\\$$' | egrep -v '^\\$$' > $@

clean:
	rm -rf *.o hydroc *~ toto *.optrpt

DATEHOUR=$(shell date +%Y%m%d%H%M )
Archiv:
	tar czvf Archiv_$(DATEHOUR).tgz Makefile *.hpp *.cpp *.h *.c input.nml

indent:
	indent *.cpp *.hpp *.c *.h

#
# - - - section to launch runs easily through a songle make command. They should be tailored to YOUR needs - - -
#

include make.runs

FORCE:
