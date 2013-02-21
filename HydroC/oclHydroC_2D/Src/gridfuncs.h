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

#ifndef GRIDFUNCS_H
#define GRIDFUNCS_H

inline __device__ int
idx1dl(void) {
  return threadIdx.y * blockDim.x + threadIdx.x;
}

inline __device__ int
idx1d(void) {
  return blockIdx.y * (gridDim.x * blockDim.x) + blockDim.x * blockIdx.x + threadIdx.x;
}

inline __device__ void
idx2d(int &x, int &y, const int nx) {
  int i1d = idx1d();
  y = i1d / nx;
  x = i1d - y * nx;
  // printf("idx2d: %ld %ld => %ld %ld \n", i1d, nx, x, y);
}

inline __device__ void
idx3d(int &x, int &y, int &z, const int nx, const int ny) {
  int i1d = idx1d();
  int plan;
  z = i1d / (nx * ny);
  plan = i1d - z * (nx * ny);
  y = plan / nx;
  x = plan - y * nx;
}

inline __device__ int
blcknum1d(void) {
  return blockIdx.y * gridDim.x + blockIdx.x;
}

inline __device__ int
nbblcks(void) {
  return gridDim.y * gridDim.x;
}

#define THREADSSZ 128
#define THREADSSZs 64

void SetBlockDims(long nelmts, long NTHREADS, dim3 & block, dim3 & grid);
void CheckErr(const char *where);
void initDevice(long myCard);
void releaseDevice(long myCard);
long getDeviceCapability(int *nDevice, long *maxMemOnDevice, long *maxThreads);

real_t reduceMax(real_t *array, long nb);
#endif
