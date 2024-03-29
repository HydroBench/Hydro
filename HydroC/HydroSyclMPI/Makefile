# Makefile for Sycl


SRCS=TimeLimit.cpp Utilities.cpp Matrix.cpp Soa.cpp \
Tile.cpp Domain.cpp Domain_Compute.cpp Domain_Boundary.cpp Domain_Prot.cpp \
Domain_DumpLine.cpp Domain_VTK.cpp Domain_Image.cpp Domain_TestCase.cpp \
FakeRead.cpp main.cpp Timers.cpp ParallelInfo.cpp cclock.cpp \
SoaDevice.cpp Tile_Shared_Variables.cpp Domain_Device.cpp \
Domain_compute_StepbyStep.cpp

DEPS=$(patsubst %.cpp,%.d,$(SRCS))
OBJS=$(patsubst %.cpp,%.o,$(SRCS)) 


OPENMP=N
PNG=N
MPI=O
HOST=$(shell uname -n | sed 's/[0-9]//g')
#ARCHEXT=dpcpp.aot
ARCHEXT=dpcpp
NODEFAULT=no
DPCPP=O
DBG=N


DEFS+=$(DEFSEXT)

# for compiler selection
include make.compil

ifeq ($(DPCPP),O)
include make.$(ARCHEXT)


NODEFAULT=yes
endif

ifeq ($(CODEPLAY),O)
include make.cp
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


include $(DEPS)

%.o: %.cpp
	$(CXX) -c $(CXXFLAGS) $(CFLAGS) $(OPTIM) $< -o $@ 

%.o: %.c
	$(CC) -c -O3 $< -o $@ 

DPCPP_INCLUDE= -I$(shell which dpcpp | sed "s,bin/dpcpp,include/sycl,g")
DPCPP_INCLUDE+= -I$(shell which dpcpp | sed "s,bin/dpcpp,include,g")

%.d: %.cpp
	@echo "Generating $@"
	@$(CXX) -M $(DEFS) $(CFLAGS) $(DPCPP_INCLUDE) -MF $@ $<


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
