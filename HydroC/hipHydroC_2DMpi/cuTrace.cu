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

#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <stdio.h>

#include "parametres.h"
#include "utils.h"
#include "gridfuncs.h"
#include "cuTrace.h"
#include "perfcnt.h"

__global__ void
Loop1KcuTrace(const double dtdx,
              const long Hnxyt,
              const long imin,
              const long imax, const double zeror, const double zerol, const double project, const int slices, const int Hnxystep,
              double *RESTRICT q, double *RESTRICT dq, double *RESTRICT c, double *RESTRICT qxm, double *RESTRICT qxp) {
  double cc, csq, r, u, v, p;
  double dr, du, dv, dp;
  double alpham, alphap, alpha0r, alpha0v;
  double spminus, spplus, spzero;
  double apright, amright, azrright, azv1right;
  double apleft, amleft, azrleft, azv1left;

  long i, j, idx;
  idx = idx1d();
  j = idx / Hnxyt;
  i = idx % Hnxyt;

  if (j >= slices)
    return;

  if (i < (imin + 1))
    return;
  if (i >= (imax - 1))
    return;

  cc = c[IHS(i,j)];
  csq = Square(cc);
  r = q[IHVWS(i,j,ID)];
  u = q[IHVWS(i,j,IU)];
  v = q[IHVWS(i,j,IV)];
  p = q[IHVWS(i,j,IP)];
  dr = dq[IHVWS(i,j,ID)];
  du = dq[IHVWS(i,j,IU)];
  dv = dq[IHVWS(i,j,IV)];
  dp = dq[IHVWS(i,j,IP)];
  alpham = half * (dp / (r * cc) - du) * r / cc;
  alphap = half * (dp / (r * cc) + du) * r / cc;
  alpha0r = dr - dp / csq;
  alpha0v = dv;

  // Right state
  spminus = (u - cc) * dtdx + one;
  spplus = (u + cc) * dtdx + one;
  spzero = u * dtdx + one;
  if ((u - cc) >= zeror) {
    spminus = project;
  }
  if ((u + cc) >= zeror) {
    spplus = project;
  }
  if (u >= zeror) {
    spzero = project;
  }
  apright = -half * spplus * alphap;
  amright = -half * spminus * alpham;
  azrright = -half * spzero * alpha0r;
  azv1right = -half * spzero * alpha0v;
  qxp[IHVWS(i,j,ID)] = r + (apright + amright + azrright);
  qxp[IHVWS(i,j,IU)] = u + (apright - amright) * cc / r;
  qxp[IHVWS(i,j,IV)] = v + (azv1right);
  qxp[IHVWS(i,j,IP)] = p + (apright + amright) * csq;

  // Left state
  spminus = (u - cc) * dtdx - one;
  spplus = (u + cc) * dtdx - one;
  spzero = u * dtdx - one;
  if ((u - cc) <= zerol) {
    spminus = -project;
  }
  if ((u + cc) <= zerol) {
    spplus = -project;
  }
  if (u <= zerol) {
    spzero = -project;
  }
  apleft = -half * spplus * alphap;
  amleft = -half * spminus * alpham;
  azrleft = -half * spzero * alpha0r;
  azv1left = -half * spzero * alpha0v;
  qxm[IHVWS(i,j,ID)] = r + (apleft + amleft + azrleft);
  qxm[IHVWS(i,j,IU)] = u + (apleft - amleft) * cc / r;
  qxm[IHVWS(i,j,IV)] = v + (azv1left);
  qxm[IHVWS(i,j,IP)] = p + (apleft + amleft) * csq;
}

__global__ void
Loop2KcuTrace(const double dtdx,
              const long Hnvar,
              const long Hnxyt,
              const long imin,
              const long imax, const double zeror, const double zerol, const double project, const int slices, const int Hnxystep,//
              double *RESTRICT q, // 
	      double *RESTRICT dq, // 
	      double *RESTRICT qxm,  // 
	      double *RESTRICT qxp // 
	      ) {
  int IN;
  double u, a;
  double da;
  double spzero;
  double acmpright;
  double acmpleft;

  int i, j;
  idx2d(i, j, Hnxyt);
  if (j >= slices)
    return;
  if (i < imin + 1)
    return;
  if (i >= imax - 1)
    return;

  printf("Passe par ici\n");

  for (IN = IP + 1; IN < Hnvar; IN++) {
    u = q[IHVWS(i,j,IU)];
    a = q[IHVWS(i,j,IN)];
    da = dq[IHVWS(i,j,IN)];

    // Right state
    spzero = u * dtdx + one;
    if (u >= zeror) {
      spzero = project;
    }
    acmpright = -half * spzero * da;
    qxp[IHVWS(i,j,IN)] = a + acmpright;

    // Left state
    spzero = u * dtdx - one;
    if (u <= zerol) {
      spzero = -project;
    }
    acmpleft = -half * spzero * da;
    qxm[IHVWS(i,j,IN)] = a + acmpleft;
  }
}

void
cuTrace(const double dtdx, const long n, const long Hscheme, const long Hnvar, const long Hnxyt, const int slices,  const int Hnxystep,      //
        double *RESTRICT qDEV,   // [Hnvar][Hnxystep][Hnxyt]
        double *RESTRICT dqDEV,  // [Hnvar][Hnxystep][Hnxyt]
        double *RESTRICT cDEV,   //        [Hnxystep][Hnxyt]
        double *RESTRICT qxmDEV, // [Hnvar][Hnxystep][Hnxyt] 
        double *RESTRICT qxpDEV  // [Hnvar][Hnxystep][Hnxyt]
	) {
  long ijmin, ijmax;
  double zerol = 0.0, zeror = 0.0, project = 0.;
  dim3 grid, block;
  int nops;

  WHERE("trace");
  ijmin = 0;
  ijmax = n;

  // if (strcmp(Hscheme, "muscl") == 0) {       // MUSCL-Hancock method
  if (Hscheme == HSCHEME_MUSCL) {       // MUSCL-Hancock method
    zerol = -hundred / dtdx;
    zeror = hundred / dtdx;
    project = one;
  }
  // if (strcmp(Hscheme, "plmde") == 0) {       // standard PLMDE
  if (Hscheme == HSCHEME_PLMDE) {       // standard PLMDE
    zerol = zero;
    zeror = zero;
    project = one;
  }
  // if (strcmp(Hscheme, "collela") == 0) {     // Collela's method
  if (Hscheme == HSCHEME_COLLELA) {     // Collela's method
    zerol = zero;
    zeror = zero;
    project = zero;
  }

  SetBlockDims(Hnxyt * slices, THREADSSZs, block, grid);
  hipLaunchKernelGGL(Loop1KcuTrace, dim3(grid), dim3(block ), 0, 0, dtdx, Hnxyt, ijmin, ijmax, zeror, zerol, project, slices, Hnxystep, qDEV, dqDEV, cDEV, qxmDEV, qxpDEV);
  CheckErr("Loop1KcuTrace");
  hipDeviceSynchronize();

  nops = slices * ((ijmax - 1) - (ijmin + 1));
  FLOPS(77 * nops, 7 * nops, 0 * nops, 0 * nops);  

  if (Hnvar > IP + 1) {
    hipLaunchKernelGGL(Loop2KcuTrace, dim3(grid), dim3(block ), 0, 0, dtdx, Hnvar, Hnxyt, ijmin, ijmax, zeror, zerol, project, slices, Hnxystep, qDEV, dqDEV, qxmDEV, qxpDEV);
    CheckErr("Loop2KcuTrace");
    hipDeviceSynchronize();
  }
}                               // trace

#undef IHVW

//EOF
