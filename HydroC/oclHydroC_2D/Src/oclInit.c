/*
  A simple 2D hydro code
  (C) Romain Teyssier : CEA/IRFU           -- original F90 code
  (C) Pierre-Francois Lavallee : IDRIS      -- original F90 code
  (C) Guillaume Colin de Verdiere : CEA/DAM -- for the C version
*/

/*

This software is governed by the CeCILL license under French law and
abiding by the rules of distribution of free software.  You can  use, 
modify and/ or redistribute the software under the terms of the CeCILL
license as circulated by CEA, CNRS and INRIA at the following URL
"http://www.cecill.info". 

As a counterpart to the access to the source code and  rights to copy,
modify and redistribute granted by the license, users are provided only
with a limited warranty  and the software's author,  the holder of the
economic rights,  and the successive licensors  have only  limited
liability. 

In this respect, the user's attention is drawn to the risks associated
with loading,  using,  modifying and/or developing or reproducing the
software by the user in light of its specific status of free software,
that may mean  that it is complicated to manipulate,  and  that  also
therefore means  that it is reserved for developers  and  experienced
professionals having in-depth computer knowledge. Users are therefore
encouraged to load and test the software's suitability as regards their
requirements in conditions enabling the security of their systems and/or 
data to be ensured and,  more generally, to use and operate it in the 
same conditions as regards security. 

The fact that you are presently reading this means that you have had
knowledge of the CeCILL license and that you accept its terms.

*/

#include <CL/cl.h>
#include <stdio.h>
#include <assert.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifdef MPI
#include <mpi.h>
#endif

#include "getDevice.h"
#include "oclInit.h"
#include "ocltools.h"
#include "oclparam.h"

cl_command_queue cqueue = 0;
cl_context ctx = 0;
cl_program pgm = 0;
int devselected = 0;
int platformselected = 0;

cl_kernel *ker = NULL;

void
oclMemset(cl_mem a, cl_int v, size_t lbyte)
{
  cl_int err = 0;
  int maxthr;
  size_t lgr;
  cl_kernel kern = ker[KernelMemset]; 

  lgr = lbyte;
  lgr /= (size_t) sizeof(real_t);
  OCLSETARG03(kern, a, v, lgr); 
  oclLaunchKernel(ker[KernelMemset], cqueue, lgr, 1024, __FILE__, __LINE__);
}

void
oclMakeHydroKernels()
{
  // All the kernels are created in one shot to save time.

  // the preprocessor transforms #a in "a"

  assert(pgm != 0);

  ker = (cl_kernel *) calloc(LastEntryKernel, sizeof(cl_kernel));
  assert(ker != NULL);

  CREATEKER(pgm, ker[Loop1KcuCmpflx], Loop1KcuCmpflx);
  CREATEKER(pgm, ker[Loop2KcuCmpflx], Loop2KcuCmpflx);
  CREATEKER(pgm, ker[LoopKQEforRow], LoopKQEforRow);
  CREATEKER(pgm, ker[LoopKcourant], LoopKcourant);
  CREATEKER(pgm, ker[Loop1KcuGather], Loop1KcuGather);
  CREATEKER(pgm, ker[Loop2KcuGather], Loop2KcuGather);
  CREATEKER(pgm, ker[Loop3KcuGather], Loop3KcuGather);
  CREATEKER(pgm, ker[Loop4KcuGather], Loop4KcuGather);
  CREATEKER(pgm, ker[Loop1KcuUpdate], Loop1KcuUpdate);
  CREATEKER(pgm, ker[Loop2KcuUpdate], Loop2KcuUpdate);
  CREATEKER(pgm, ker[Loop3KcuUpdate], Loop3KcuUpdate);
  CREATEKER(pgm, ker[Loop4KcuUpdate], Loop4KcuUpdate);
  CREATEKER(pgm, ker[Loop1KcuConstoprim], Loop1KcuConstoprim);
  CREATEKER(pgm, ker[Loop2KcuConstoprim], Loop2KcuConstoprim);
  CREATEKER(pgm, ker[LoopEOS], LoopEOS);
  CREATEKER(pgm, ker[Loop1KcuMakeBoundary], Loop1KcuMakeBoundary);
  CREATEKER(pgm, ker[Loop2KcuMakeBoundary], Loop2KcuMakeBoundary);
  CREATEKER(pgm, ker[Loop1KcuQleftright], Loop1KcuQleftright);
  CREATEKER(pgm, ker[Loop1KcuRiemann], Loop1KcuRiemann);
  CREATEKER(pgm, ker[Loop10KcuRiemann], Loop10KcuRiemann);
  CREATEKER(pgm, ker[LoopKcuSlope], LoopKcuSlope);
  CREATEKER(pgm, ker[Loop1KcuTrace], Loop1KcuTrace);
  CREATEKER(pgm, ker[Loop2KcuTrace], Loop2KcuTrace);
  CREATEKER(pgm, ker[LoopKredMaxReal], reduceMaxReal);
  CREATEKER(pgm, ker[KernelMemset], KernelMemset);
  CREATEKER(pgm, ker[KernelMemsetV4], KernelMemsetV4);
  CREATEKER(pgm, ker[kpack_arrayv], kpack_arrayv);
  CREATEKER(pgm, ker[kunpack_arrayv], kunpack_arrayv);
  CREATEKER(pgm, ker[kpack_arrayh], kpack_arrayh);
  CREATEKER(pgm, ker[kunpack_arrayh], kunpack_arrayh);
  CREATEKER(pgm, ker[LoopKComputeDeltat], LoopKComputeDeltat);
}

void
oclReleaseHydroKernels()
{
  assert(ker != NULL);
  
  FREEKER( ker[Loop1KcuCmpflx], Loop1KcuCmpflx);
  FREEKER( ker[Loop2KcuCmpflx], Loop2KcuCmpflx);
  FREEKER( ker[LoopKQEforRow], LoopKQEforRow);
  FREEKER( ker[LoopKcourant], LoopKcourant);
  FREEKER( ker[Loop1KcuGather], Loop1KcuGather);
  FREEKER( ker[Loop2KcuGather], Loop2KcuGather);
  FREEKER( ker[Loop3KcuGather], Loop3KcuGather);
  FREEKER( ker[Loop4KcuGather], Loop4KcuGather);
  FREEKER( ker[Loop1KcuUpdate], Loop1KcuUpdate);
  FREEKER( ker[Loop2KcuUpdate], Loop2KcuUpdate);
  FREEKER( ker[Loop3KcuUpdate], Loop3KcuUpdate);
  FREEKER( ker[Loop4KcuUpdate], Loop4KcuUpdate);
  FREEKER( ker[Loop1KcuConstoprim], Loop1KcuConstoprim);
  FREEKER( ker[Loop2KcuConstoprim], Loop2KcuConstoprim);
  FREEKER( ker[LoopEOS], LoopEOS);
  FREEKER( ker[Loop1KcuMakeBoundary], Loop1KcuMakeBoundary);
  FREEKER( ker[Loop2KcuMakeBoundary], Loop2KcuMakeBoundary);
  FREEKER( ker[Loop1KcuQleftright], Loop1KcuQleftright);
  FREEKER( ker[Loop1KcuRiemann], Loop1KcuRiemann);
  FREEKER( ker[Loop10KcuRiemann], Loop10KcuRiemann);
  FREEKER( ker[LoopKcuSlope], LoopKcuSlope);
  FREEKER( ker[Loop1KcuTrace], Loop1KcuTrace);
  FREEKER( ker[Loop2KcuTrace], Loop2KcuTrace);
  FREEKER( ker[LoopKredMaxReal], reduceMaxReal);
  FREEKER( ker[KernelMemset], KernelMemset);
  FREEKER( ker[KernelMemsetV4], KernelMemsetV4);
  FREEKER( ker[kpack_arrayv], kpack_arrayv);
  FREEKER( ker[kunpack_arrayv], kunpack_arrayv);
  FREEKER( ker[kpack_arrayh], kpack_arrayh);
  FREEKER( ker[kunpack_arrayh], kunpack_arrayh);
  FREEKER( ker[LoopKComputeDeltat], LoopKComputeDeltat);

  free( (void*) ker );
}

void
oclCloseupCode()
{
  oclReleaseHydroKernels();
  
  if ( pgm ) { clReleaseProgram( pgm ); pgm = 0; }
  if ( cqueue ) { clReleaseCommandQueue( cqueue ); cqueue = 0; }
  if ( ctx ) { clReleaseContext( ctx ); ctx = 0; }
}


void
oclInitCode(const int nproc, const int mype)
{
  int verbose = 1;
  int nbplatf = 0;
  int nbgpu = 0;
  int nbcpu = 0;
  int nbacc = 0;
  char srcdir[1024];

  if (mype > 0) verbose = 0; // assumes a homogeneous machine :-(
  nbplatf = oclGetNbPlatforms(verbose);
  if (nbplatf == 0) {
    fprintf(stderr, "No OpenCL platform available\n");
    abort();
  }

  devselected = -1;
  for (platformselected = 0; platformselected < nbplatf; platformselected++) {

    nbgpu = oclGetNbOfGpu(platformselected);
    fprintf(stdout, "Hydro: %03d has %d GPU\n", mype, nbgpu);
    if ((runUnit == RUN_GPU) && (nbgpu > 0)) {
	    int gpuSel = 0;
	    if (mype == 0) fprintf(stderr, "Building a GPU version\n");
	    if (nproc == 1) {
		    devselected = oclGetGpuDev(platformselected, gpuSel);
	    } else {
		    if (nbgpu > 1) gpuSel = GetDevice(nbgpu);
		    devselected = oclGetGpuDev(platformselected, gpuSel);
		    if (devselected == -1) {
			    fprintf(stderr, "Error: more MPI ranks on a node than GPU cards\n");
#ifdef MPI
			    MPI_Abort(MPI_COMM_WORLD, 9);
#else
			    exit(9);
#endif
		    }
	    }
	    fprintf(stdout, "Hydro: %03d uses GPU %d\n", mype, gpuSel);
	    fflush(stdout);
	    break;
    }
    
    nbacc = oclGetNbOfAcc(platformselected);
    fprintf(stdout, "Hydro: %03d has %d ACC\n", mype, nbacc);
    if ((runUnit == RUN_ACC) && (nbacc > 0)) {
	    int accSel = 0;
	    if (mype == 0) fprintf(stderr, "Building an ACC version\n");
	    if (nproc == 1) {
		    devselected = oclGetAccDev(platformselected, accSel);
	    } else {
		    if (nbacc > 1) accSel = GetDevice(nbacc);
		    devselected = oclGetAccDev(platformselected, accSel);
	    }
	    fprintf(stdout, "Hydro: %03d uses ACC %d\n", mype, accSel);
	    fflush(stdout);
	    break;
    }

    nbcpu = oclGetNbOfCpu(platformselected);
    fprintf(stdout, "Hydro: %03d has %d CPU\n", mype, nbcpu);
    if ((runUnit == RUN_CPU) && (nbcpu > 0)) {
      if (mype == 0) fprintf(stderr, "Building a CPU version\n");
      devselected = oclGetCpuDev(platformselected, 0);
      break;
    }
  }

  ctx = oclCreateCtxForPlatform(platformselected, verbose);
  cqueue = oclCreateCommandQueueForDev(platformselected, devselected, ctx, 0);
  getcwd(srcdir, 1023);
  pgm = oclCreatePgmFromCtx("hydro_kernels.cl", srcdir, ctx, platformselected, devselected, verbose);
  // exit(2);
  oclMakeHydroKernels();
}

//EOF
