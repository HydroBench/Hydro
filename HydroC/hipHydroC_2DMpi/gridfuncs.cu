#include "hip/hip_runtime.h"
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
#include <stdio.h>
#include <limits.h>
#include <hip/hip_runtime.h>
#include <assert.h>
#include <math.h>
#include <unistd.h>

#include "gridfuncs.h"
#include "utils.h"

#define VERIF(x, ou) if ((x) != hipSuccess)  { CheckErr((ou)); }

void
SetBlockDims(long nelmts, long NTHREADS, dim3 & block, dim3 & grid) {

  // fill a 2D grid if necessary
  long totalblocks = (nelmts + (NTHREADS) - 1) / (NTHREADS);
  long blocksx = totalblocks;
  long blocksy = 1;
  while (blocksx > 65534) {
    blocksx /= 2;
    blocksy *= 2;
  }
  if ((blocksx * blocksy * NTHREADS) < nelmts)
    blocksx++;
  grid.x = blocksx;
  grid.y = blocksy;
  block.x = NTHREADS;
  block.y = 1;

//     if (verbosity > 1) {
//         fprintf(stderr, "N=%d: bx=%d by=%d gx=%d gY=%d\n", nelmts, block->x,
//                 block->y, grid->x, grid->y);

//     }
}

void
initDevice(long myCard) {
  hipSetDevice(myCard);
}

void
releaseDevice(long myCard) {
  hipDeviceReset();
  CheckErr("releaseDevice");
}

void
CheckErr(const char *where) {
  hipError_t cerror;
  cerror = hipGetLastError();
  if (cerror != hipSuccess) {
    char host[256];
    char message[1024];
    gethostname(host, 256);
    sprintf(message, "CudaError: %s (%s) on %s\n", hipGetErrorString(cerror), where, host);
    fputs(message, stderr);
    exit(1);
  }
}

long
getDeviceCapability(int *nDevice, long *maxMemOnDevice, long *maxThreads) {
  int deviceCount;
  long dev;
  long memorySeen = LONG_MAX;
  long maxth = INT_MAX;
  hipError_t status;
  status = hipGetDeviceCount(&deviceCount);
  if (status != hipSuccess) {
    CheckErr("hipGetDeviceCount");
    return 1;
  }
  if (deviceCount == 0) {
    printf("There is no device supporting CUDA\n");
    return 1;
  }
  for (dev = 0; dev < deviceCount; ++dev) {
    hipDeviceProp_t deviceProp;
    status = hipGetDeviceProperties(&deviceProp, dev);
    if (status != hipSuccess) {
      CheckErr("hipGetDeviceProperties");
      return 1;
    }
    if (dev == 0) {
      if (deviceProp.major == 9999 && deviceProp.minor == 9999) {
        fprintf(stderr, "There is no device supporting CUDA.\n");
        return 1;
      }
    }
    if (deviceProp.totalGlobalMem < memorySeen)
      memorySeen = deviceProp.totalGlobalMem;
    if (maxth > deviceProp.maxThreadsPerBlock)
      maxth = deviceProp.maxThreadsPerBlock;
  }
  *nDevice = deviceCount;
  *maxMemOnDevice = memorySeen;
  *maxThreads = maxth;
  return 0;
}

__global__ void
LoopKredMaxDble(double *src, double *res, const long nb) {
  __shared__ double sdata[512];
  int blockSize = blockDim.x * blockDim.y * blockDim.z;
  int tidL = threadIdx.x;
  int myblock = blockIdx.x + blockIdx.y * blockDim.x + blockIdx.z * blockDim.x * blockDim.y;
  int i = idx1d();

  // protection pour les cas ou on n'est pas multiple du block
  // par defaut le max est le premier element
  sdata[threadIdx.x] = src[0];
  if (i < nb) {
    sdata[threadIdx.x] = src[i];
  }
  __syncthreads();

  // do the reduction in parallel
  if (tidL < 64) {
    if (blockSize >= 128) {
      sdata[tidL] = MAX(sdata[tidL], sdata[tidL + 64]);
      __syncthreads();
    }
    if (blockSize >= 64) {
      sdata[tidL] = MAX(sdata[tidL], sdata[tidL + 32]);
      __syncthreads();
    }
    if (blockSize >= 32) {
      sdata[tidL] = MAX(sdata[tidL], sdata[tidL + 16]);
      __syncthreads();
    }
    if (blockSize >= 16) {
      sdata[tidL] = MAX(sdata[tidL], sdata[tidL + 8]);
      __syncthreads();
    }
    if (blockSize >= 8) {
      sdata[tidL] = MAX(sdata[tidL], sdata[tidL + 4]);
      __syncthreads();
    }
    if (blockSize >= 4) {
      sdata[tidL] = MAX(sdata[tidL], sdata[tidL + 2]);
      __syncthreads();
    }
    if (blockSize >= 2) {
      sdata[tidL] = MAX(sdata[tidL], sdata[tidL + 1]);
      __syncthreads();
    }
  }
  // get the partial result from this block
  if (tidL == 0) {
    res[myblock] = sdata[0];
    // printf("%d %lf\n", blockSize, sdata[0]);
  }
}

double
reduceMax(double *array, long nb) {
  int bs = 32;
  dim3 grid, block;
  int nbb = nb / bs;
  double resultat = 0;
  hipError_t status;
  double *temp1, *temp2;

  nbb = (nb + bs - 1) / bs;

  status = hipMalloc((void **) &temp1, nbb * sizeof(double));
  VERIF(status, "hipMalloc temp1");
  status = hipMalloc((void **) &temp2, nbb * sizeof(double));
  VERIF(status, "hipMalloc temp2");
  double *tmp;

  // on traite d'abord le tableau d'origine
  SetBlockDims(nb, bs, block, grid);
  hipLaunchKernelGGL(LoopKredMaxDble, dim3(grid), dim3(block ), 0, 0, array, temp1, nb);
  CheckErr("KredMaxDble");
  hipDeviceSynchronize();
  CheckErr("reducMax");

  // ici on a nbb maxima locaux

  while (nbb > 1) {
    SetBlockDims(nbb, bs, block, grid);
    hipLaunchKernelGGL(LoopKredMaxDble, dim3(grid), dim3(block ), 0, 0, temp1, temp2, nbb);
    CheckErr("KredMaxDble 2");
    hipDeviceSynchronize();
    CheckErr("reducMax 2");
    // on permute les tableaux pour une eventuelle iteration suivante,
    tmp = temp1;
    temp1 = temp2;
    temp2 = tmp;
    // on rediminue la taille du probleme
    nbb = (nbb + bs - 1) / bs;
    // fprintf(stderr, "n=%d b=%d\n", nbb, bs);
  }

  hipMemcpy(&resultat, temp1, sizeof(double), hipMemcpyDeviceToHost);
  // printf("R=%lf\n", resultat);
  hipFree(temp1);
  hipFree(temp2);
  return resultat;
}

