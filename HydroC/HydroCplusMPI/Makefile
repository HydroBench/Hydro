SRCS=TimeLimit.cpp Utilities.cpp Matrix.cpp Soa.cpp ThreadBuffers.cpp Tile.cpp Domain.cpp Domain_Compute.cpp Domain_Boundary.cpp Domain_Prot.cpp Domain_VTK.cpp Domain_Image.cpp Domain_TestCase.cpp FakeRead.cpp main.cpp 
DEPS=$(patsubst %.cpp,%.d,$(SRCS))
OBJS=$(patsubst %.cpp,%.o,$(SRCS)) cclock.o
OPENMP=O
PNG=O
MPI=O
HOST=$(shell uname -n | sed 's/[0-9]//g')
ARCHEXT=std

# INCS+=-I$(HOME)/vecadv/advisor_xe/include

CXX=icpc
DEFS+=$(DEFSEXT)

ifeq ($(PI),O)
include make.pi
endif

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

ifeq ($(CRAY),O)
include make.cray
endif

CFLAGS+=$(DEFS)

include make.opts

ifeq ($(PNG),O)
LIBS+=-lpng 
endif

CFLAGS+=$(INCS)

hydroc: $(OBJS)
	$(CXX) $(CFLAGS) $(OPTIM) $(OBJS) $(LIBS) -lm -lrt -lpthread -o hydroc 
# -lrt -ldl
	cp hydroc hydroc.$(ARCHEXT)

include $(DEPS)

%.o: %.cpp
	$(CXX) -c $(CFLAGS) $(OPTIM) $< -o $@ 

%.o: %.c
	$(CXX) -c $(CFLAGS) $(OPTIM) $< -o $@ 

ifeq ($(CRAY),O)
%.d: %.cpp
	touch $@
else
%.d: %.cpp
	$(CXX) -M $< $(DEFS) $(CFLAGS) | sed 's+ + \\\n+g' | egrep -v '/usr|/ccc/produc|/cea/prod' | egrep -v '^.\\$$' | egrep -v '^\\$$' > $@
endif
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
