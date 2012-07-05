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
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <stdio.h>
#include "parametres.h"
#include "perfcnt.h"
#include "utils.h"
#include "gridfuncs.h"
#include "cuSlope.h"
#define DABS(x) (double) fabs((x))
//
__global__ void
LoopKcuSlope(const long Hnvar, const long Hnxyt, const double slope_type, const long ijmin, const long ijmax, const int slices, const int Hnxystep,  //
             double *RESTRICT q, double *RESTRICT dq) {
  int n;
  double dlft, drgt, dcen, dsgn, slop, dlim;
  int i, j;
  idx2d(i, j, Hnxyt);
  if (j >= slices)
    return;

  // printf("LoopKcuSlope: %ld %ld \n", i, j);

  //  idx3d(i, j, n, Hnxyt, Hstep);
  //   if (n >= Hnvar)
  //  return;
  if (i < 1) return;
  if (i >= ijmax -1) return;


  for (n = 0; n < Hnvar; n++) {
    dlft = slope_type * (q[IHVWS(i,j,n)] - q[IHVWS(i - 1,j,n)]);
    drgt = slope_type * (q[IHVWS(i + 1,j,n)] - q[IHVWS(i,j,n)]);
    dcen = half * (dlft + drgt) / slope_type;
    dsgn = (dcen > 0) ? (double) 1.0 : (double) -1.0;     // sign(one, dcen);
    slop = (double) MIN(DABS(dlft), DABS(drgt));
    dlim = ((dlft * drgt) <= zero) ? zero : slop;
    //         if ((dlft * drgt) <= zero) {
    //             dlim = zero;
    //         }
    dq[IHVWS(i,j,n)] = dsgn * (double) MIN(dlim, DABS(dcen));
  }
}

void
cuSlope(const long narray, const long Hnvar, const long Hnxyt, const double slope_type, const int slices, const int Hnxystep,        // 
        double *RESTRICT qDEV,  // [Hnvar][Hnxystep][Hnxyt]
        double *RESTRICT dqDEV  // [Hnvar][Hnxystep][Hnxyt]
	) {
  long ijmin, ijmax;
  dim3 grid, block;
  int nops;

  WHERE("slope");
  ijmin = 1;
  ijmax = narray;
  SetBlockDims(Hnxyt * slices, THREADSSZ, block, grid); // Hnvar * 
  LoopKcuSlope <<< grid, block >>> (Hnvar, Hnxyt, slope_type, ijmin, ijmax, slices, Hnxystep, qDEV, dqDEV);
  CheckErr("LoopKcuSlope");
  cudaThreadSynchronize();
  nops = Hnvar * slices * ((ijmax - 1) - (ijmin));
  FLOPS(8 * nops, 1 * nops, 6 * nops, 0 * nops);}                               // slope

//EOF
