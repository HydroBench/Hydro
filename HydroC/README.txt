				HydroC
				======

		   (C) Guillaume Colin de Verdiere

The benchmark is a full hydro code implementing an 2D Eulerian scheme
using a Godunov method.

It has been implemented in various versions yet it yields (otherwise
it a bug) the same results.

The most interesting versions are described below. At the end of this
file, a short description of the input file is given.

HydroC99_2DMpi
	A fine grain OpenMP + MPI version using C99

HydroCplusMPI
	A coarse grain OpenMP + MPI using C++

HydroC_2DMpi_OpenACC_4_parallel
	A OpenACC + MPI version

cuHydroC_2DMpi
	An implementation using CUDA + MPI

oaccHydroC_2DMPI
	Another OpenACC + MPI implementation

oclHydroC_2D
	An implementation using OpenCL + MPI



To run the code the basic form is 

			  hydro -i input.nml

(depending on the implementation hydro executable could be named also
hydroc, see the corresponding Makefile to be sure).

The OpenCL version has also an extra option -u to select the computing
unit. Its value can be c for CPU, g for GPU or a for
ACCELERATOR. Depending on the hardware, it is easy to compare
performances of a CPU versus a GPU or an accelerator (Xeon Phi).

To run in parallel using MPI, one has just to specify the number of
tasks at launch time. E.g. mpirun -n 100 ./hydro -i input.nml. This
command will compute the domain using 100 subdomains. Note that in
case of a serie of runs with checkpoint/restart activated, the number
of MPI tasks must be kept constant all along.

For an hybrid (MPI + OpenMP) one can use the following (to be adapted
to your system): env OMP_NUM_THREADS=8 mpirun -n 100 ./hydro -i
input.nml. In that case, each MPI task will use 8 threads.


Here is the input file of HydroCplusMPI which is the richest. Other
version rely on less options (see parametres.c or Domain.cpp to have
the exact list of what each version understands). Keywords that are
not understood are just ignored.

A line starting with a # is seen as a comment and ignored alltogether.

______________________________________________________________
This namelist contains various input parameters for HYDRO runs

&RUN
tend=20000
#tend=3
# noutput=1
nstepmax=200
# dtoutput=0.1
# dtimage=10
# chkpt=1
/

&MESH
nx=2000
ny=2000
nxystep=480
tilesize=60
numa=0
morton=0
prt=0
dx=0.05
boundary_left=1
boundary_right=1
boundary_down=1
boundary_up=1
testcase=0
/

&HYDRO
courant_factor=0.8
niter_riemann=10
/
______________________________________________________________
RUN section
===========

The simulation runs until the PHYSICAL simulation time tend is reached
or the elaps time limit is reached (controled by the environment
variable BRIDGE_MPRUN_MAXTIME or BRIDGE_MSUB_MAXTIME or HYDROC_MAXTIME
expressed in seconds). 

Furthermore, the code checks also the content of HYDROC_START_TIME
(set by `date +%s`) which should be set at the beginning of the script
to make sure that the compute time is the proper one, should the
actions before laucnhing the code be (too) long.

If the code stops before tend, there is the possibility to write a
restart dump using chkpt=1. It will write a Continue.dump file and if
relaunched, the code will restart where it stopped.

Another way to stop the code is to force only a limited number of
iterations per runs; it is done with the nstepmax parameter.

The code can produce VTK files (one per MPI task and output time --
which might stress the inode capacity). Those files can be written
either every n iterations (using noutput) or every dtoutput physical
time increments. The former (noutput) is OK for debugging, the latter
(dtouput) is good to produce a movie with a linear evolution of
time. The main file for Paraview is Hydro.pvd and all the files are in
the Dep directory. For large values of NX and NY with small output
increment, the Dep directory can become rather large...

The code can also produce PNG images at regular physical intervals
using dtimage. This is convenient to produce a movie while not
overflowing the disks.

Note that the files (VTK or PNG) are written in the CURRENT
directory. 

MESH section
============

The GLOBAL domain is of size nx x ny. In case of a MPI run with
multiple tasks, the global domain is subdived amongst all the MPi
tasks.

The testcase is selected by the testcase keyword.

nxystep specifies the width of the subdomain to process (fine grain
OpenMP, CUDA, OpenCL). The code processes at first NXxNXYSTEP bands then
NYxNXSTEP bands and so on. This parameter is ignored in the coarse
grain C++ version.

For the coarse grain (CG) version, the local domain is split in square
tiles of tilesize side. There is a default compile time value which
can be overidden by this keyword.

numa=1 indicates that the CG version should initialize its memory in a
numa aware fashion.

morton=1 says that the tiles are processed according to a morton curve
numbering.

prt=1 says that the code should dump its arrays (REALLY verbose).

Everything else should be left alone. Note that niter_riemann should
be kept as a user defined parameter even if in most cases a very
limited number of iterations are needed. This puts intentionnaly an
extra burden in the manner of coding the Riemann() routine.


References
==========

Guillaume Colin de Verdière. Hydro benchmark, September
2013. https://github.com/ HydroBench/Hydro.git.

Pierre-François Lavallée, Guillaume Colin de Verdière, Philippe
Wautelet, Dimitri Lecas, and Jean- Michel Dupays. Porting and
optimizing hydro to new platforms and programming paradigms– lessons
learn. Technical report, PRACE, December 2012. Parallel programming
interfaces.

Philip L. Roe. Approximate riemann solvers, parameter vectors, and
difference schemes. Journal of computational physics, 43(2):357–372,
1981.

Bram van Leer. Towards the ultimate conservative difference
scheme. iv. a new approach to numerical convection. Journal of
computational physics, 23(3):276–299, 1977.

Godunov, S. K. (1959), "A Difference Scheme for Numerical Solution of
Discontinuous Solution of Hydrodynamic Equations", Math. Sbornik, 47,
271–306, translated US Joint Publ. Res. Service, JPRS 7226, 1969.


see also : http://en.wikipedia.org/wiki/Godunov%27s_scheme

