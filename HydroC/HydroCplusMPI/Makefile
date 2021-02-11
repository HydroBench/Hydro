SRCS=TimeLimit.cpp Utilities.cpp Matrix.cpp Soa.cpp ThreadBuffers.cpp Tile.cpp Domain.cpp Domain_Compute.cpp Domain_Boundary.cpp Domain_Prot.cpp Domain_DumpLine.cpp Domain_VTK.cpp Domain_Image.cpp Domain_TestCase.cpp FakeRead.cpp main.cpp Timers.cpp
DEPS=$(patsubst %.cpp,%.d,$(SRCS))
OBJS=$(patsubst %.cpp,%.o,$(SRCS)) cclock.o
OPENMP=O
PNG=O
MPI=O
HOST=$(shell uname -n | sed 's/[0-9]//g')
ARCHEXT=std
NODEFAULT=no

# INCS+=-I$(HOME)/vecadv/advisor_xe/include

DEFS+=$(DEFSEXT)

# for compiler selection
include make.compil

ifeq ($(PI),O)
include make.pi
NODEFAULT=yes
endif

ifeq ($(ARM),O)
include make.arm
NODEFAULT=yes
endif

ifeq ($(ATOM),O)
include make.atom
NODEFAULT=yes
endif

ifeq ($(MIC),O)
include make.mic
NODEFAULT=yes
endif

ifeq ($(SSE),O)
include make.sse
NODEFAULT=yes
endif

ifeq ($(KNL),O)
# Knights Landing using AVX512 with specificities of KNL
include make.knl
NODEFAULT=yes
endif

ifeq ($(AVX3),O)
# Skylake Xeon and beyond
include make.avx3
NODEFAULT=yes
endif

ifeq ($(AVX512),O)
# Skylake Xeon and beyond
include make.avx3
NODEFAULT=yes
endif

ifeq ($(AVX2),O)
include make.avx2
NODEFAULT=yes
endif

ifeq ($(AVX),O)
include make.avx
NODEFAULT=yes
endif

ifeq ($(CRAY),O)
include make.cray
NODEFAULT=yes
endif

ifeq ($(THX2),O)
include make.thx2
NODEFAULT=yes
endif

ifeq ($(EPYC),O)
include make.epyc
NODEFAULT=yes
endif

ifeq ($(SVE),O)
include make.sve
NODEFAULT=yes
endif

ifeq ($(GEN9),O)
include make.gen9
NODEFAULT=yes
endif

ifeq ($(NODEFAULT),no)
# capture the default case where no arch indication has been given to make
# note that make.local is not in the git system to avoid overidding it
include make.local
endif


CFLAGS+=$(DEFS)

include make.opts

ifeq ($(PNG),O)
LIBS+=-lpng 
endif

CFLAGS+=$(INCS)

PTH=
ifeq ($(OPENMP),O)
PTH=-L/usr/lib64 -lpthread
endif

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
	$(CC) -c $< -o $@ 

ifeq ($(CRAY),O)
%.d: %.cpp
	touch $@
else
%.d: %.cpp
	@echo "Generating $@"
	@$(CXX) -M $(DEFS) $(CFLAGS) $< | sed 's+ + \\\n+g' | egrep -v '/usr|/ccc/produc|/cea/prod' | egrep -v '^.\\$$' | egrep -v '^\\$$' > $@
endif

cleanall:
	rm -f *.d *.o hydroc *~ toto *.optrpt

DATEHOUR=$(shell date +%Y%m%d%H%M )
Archiv:
	rm -f *~
	tar czvf Archiv_$(DATEHOUR).tgz hydroAverage README Makefile make.* *.hpp *.cpp *.h *.c inputlong.nml cmpDumps cmpCompilos
# config.lua goOneview

# TestThx2_Omp TestThx2_1 README.after 

indent:
	indent *.cpp *.hpp *.c *.h

#
# - - - section to launch runs easily through a songle make command. They should be tailored to YOUR needs - - -
#

include make.runs

FORCE:


clean: FORCE
	rm -f *.o hydroc *~ toto *.optrpt

