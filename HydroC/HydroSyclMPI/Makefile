# Makefile for DPCPP


SRCS=TimeLimit.cpp Utilities.cpp Matrix.cpp Soa.cpp ThreadBuffers.cpp Tile.cpp Domain.cpp Domain_Compute.cpp Domain_Boundary.cpp Domain_Prot.cpp Domain_DumpLine.cpp Domain_VTK.cpp Domain_Image.cpp Domain_TestCase.cpp FakeRead.cpp main.cpp Timers.cpp
DEPS=$(patsubst %.cpp,%.d,$(SRCS))
OBJS=$(patsubst %.cpp,%.o,$(SRCS)) cclock.o


OPENMP=N
PNG=O
MPI=N
HOST=$(shell uname -n | sed 's/[0-9]//g')
ARCHEXT=dpcpp
NODEFAULT=no
DPCPP=O
DBG=O


DEFS+=$(DEFSEXT)

# for compiler selection
include make.compil

ifeq ($(DPCPP),O)
include make.dpcpp
NODEFAULT=yes
endif


ifeq ($(NODEFAULT),no)
# capture the default case where no arch indication has been given to make
# note that make.local is not in the git system to avoid overidding it
include make.local
endif


CFLAGS+=$(DEFS)

# include make.opts

ifeq ($(PNG),O)
LIBS+=-lpng 
endif

CFLAGS+=$(INCS)

PTH=


hydroc: $(OBJS)
	$(CXX) $(CFLAGS) $(OPTIM) $(OBJS) $(STATICOPT) $(LDFLAGS) $(LIBS) $(PTH) -lm -lrt -o hydroc 
# -lrt -ldl
	cp hydroc hydroc.$(ARCHEXT)
ifeq ($(STATIC),O)
	cp hydroc hydroc.$(ARCHEXT).static
endif


#include $(DEPS)

%.o: %.cpp
	$(CXX) -c $(CXXFLAGS) $(CFLAGS) $(OPTIM) $< -o $@ 

%.o: %.c
	$(CC) -c -O3 $< -o $@ 

ifeq ($(CRAY),O)
%.d: %.cpp
	touch $@
else
%.d: %.cpp
	@echo "Generating $@"
	@$(CXX) -M $(DEFS) $(CFLAGS) $< | sed 's+ + \\\n+g' | egrep -v '/usr|/ccc/produc|/cea/prod' | egrep -v '^.\\$$' | egrep -v '^\\$$' > $@
endif
clean:
	rm -rf *.o hydroc *~ toto *.optrpt

cleanall:
	rm -rf *.d *.o hydroc *~ toto *.optrpt

DATEHOUR=$(shell date +%Y%m%d%H%M )
Archiv:
	rm -f *~
	tar czvf Archiv_$(DATEHOUR).tgz hydroAverage README Makefile make.* *.hpp *.cpp *.h *.c inputlong.nml cmpDumps cmpCompilos
# config.lua goOneview

# TestThx2_Omp TestThx2_1 README.after 

indent:
	clang-format -i *.cpp *.hpp
	#indent *.cpp *.hpp *.c *.h

#
# - - - section to launch runs easily through a songle make command. They should be tailored to YOUR needs - - -
#

include make.runs

FORCE:
