# C compiler
#------------------
CC = gcc

# Cuda compiler
#------------------
# NVCC = nvcc -gencode arch=compute_70,code=sm_70
NVCC = hipcc --amdgpu-target=gfx908
#NVCC = nvcc -gencode arch=compute_35,code=sm_35 -dlink

# Fortran compilers
#------------------
F90 = ifort

# MPI C compiler
#------------------
MPICC = mpcc

# MPI Fortran compilers
#------------------
MPIF90 = mpif90

# C compilation options
#-----------------------------
CFLAGS = -O3
NVCFLAGS =  -O3

# Fortran compilation options
#-----------------------------
FFLAGS = -O3

# Linker options
#---------------
LDFLAGS = -O3

# OpenMP C compilation options
#---------------------------
CFLAGS_OMP = -openmp

# Cuda compilation options
#---------------------------
NVCFLAGS_OMP =

# OpenMP Fortran compilation options
#---------------------------
FFLAGS_OMP = -openmp

ifeq ($(MPI),O)
# MPI C compilation options
#---------------------------
#MPIROOT=/opt/mpi/bullxmpi/1.1.14.1
MPIROOT=$(BULLXMPI_ROOT)
MPIROOT=$(I_MPI_ROOT)/intel64
# CFLAGS_MPI = -DMPI -I$(MPI_ROOT)/include
# NVCFLAGS_MPI = -DWITHMPI -I$(MPI_ROOT)/include
CFLAGS_MPI = -DMPI -I$(MPI_ROOT)/include
NVCFLAGS_MPI = -DWITHMPI -I$(MPI_ROOT)/include
# MPI link options
#--------------------
LDFLAGS_MPI = -L$(MPI_ROOT)/lib -lmpi
endif # MPI


# MPI Fortran compilation options
#---------------------------
FFLAGS_MPI =

# OpenMP link options
#--------------------
LDFLAGS_OMP = -openmp

# MPI link options
#--------------------
# LDFLAGS_MPI = -L$(MPI_ROOT)/lib -lmpi
