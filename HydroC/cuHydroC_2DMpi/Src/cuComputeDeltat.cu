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

#ifdef WITHMPI
 #ifdef SEEK_SET
  #undef SEEK_SET
  #undef SEEK_CUR
  #undef SEEK_END
 #endif
#include <mpi.h>
#endif
#include <stdio.h>
// #include <stdlib.h>
#include <malloc.h>
// #include <unistd.h>
#include <math.h>
#include <cuda.h>

#include "parametres.h"
#include "cuComputeDeltat.h"
#include "cuHydroGodunov.h"
#include "gridfuncs.h"
#include "utils.h"
#include "perfcnt.h"
#include "cuEquationOfState.h"

#define DABS(x) (double) fabs((x))
#define VERIF(x, ou) if ((x) != cudaSuccess)  { CheckErr((ou)); }

__global__ void
LoopKQEforRows(const long j, double *uold, double *q, double *e, const double Hsmallr,  //
               const long Hnxt, const long Hnyt, const long Hnxyt,      //
               const int slices, const long Hnxystep, const long n) {
  double eken;
  int i, s;
  double qID, qIP, qIU, qIV;
  idx2d(i, s, Hnxyt);
  if (s >= slices)
    return;

  if (i >= n)
    return;
  

  qID = MAX(uold[IHU(i + ExtraLayer, j + s, ID)], Hsmallr);
  qIU = uold[IHU(i + ExtraLayer, j + s, IU)] / qID;
  qIV = uold[IHU(i + ExtraLayer, j + s, IV)] / qID;
  eken = half * (Square(qIU) + Square(qIV));
  qIP = uold[IHU(i + ExtraLayer, j + s, IP)] / qID - eken;
  e[IHS(i, s)] = qIP;
  q[IHVWS(i, s, ID)] = qID;
  q[IHVWS(i, s, IP)] = qIP;
  q[IHVWS(i, s, IU)] = qIU;
  q[IHVWS(i, s, IV)] = qIV;
}

void
cuComputeQEforRows(const long j, double *uold, double *q, double *e, const double Hsmallr, const long Hnx,      //
                   const long Hnxt, const long Hnyt, const long Hnxyt, const int slices,
                   const long Hnxystep) {
  dim3 grid, block;

  SetBlockDims(Hnx * slices, THREADSSZ, block, grid);
  LoopKQEforRows <<< grid, block >>> (j, uold, q, e, Hsmallr, Hnxt, Hnyt, Hnxyt, slices, Hnxystep,
                                      Hnx);
  CheckErr("courantOnXY");
  cudaThreadSynchronize();
  CheckErr("courantOnXY");
}

__global__ void
LoopKcourant(double *q, double *courant, const double Hsmallc, const double *c, //
             const long Hnxyt, const int slices, const long Hnxystep, const long n) {
  double cournox, cournoy, courantl;
  int i, s;
  idx2d(i, s, Hnxyt);
  if (s >= slices)
    return;

  if (i >= n)
    return;

  cournox = cournoy = 0.;

  cournox = c[IHS(i, s)] + DABS(q[IHVWS(i, s, IU)]);
  cournoy = c[IHS(i, s)] + DABS(q[IHVWS(i, s, IV)]);
  courantl = MAX(cournox, MAX(cournoy, Hsmallc));
  courant[IHS(i, s)] = MAX(courant[IHS(i, s)], courantl);
}

void
cuCourantOnXY(double *courant, const long Hnx, const long Hnxyt, const int slices,
              const long Hnxystep, double *c, double *q, double Hsmallc) {
  dim3 grid, block;

  SetBlockDims(Hnx * slices, THREADSSZ, block, grid);
  LoopKcourant <<< grid, block >>> (q, courant, Hsmallc, c, Hnxyt, slices, Hnxystep, Hnx);
  CheckErr("courantOnXY");
  cudaThreadSynchronize();
  CheckErr("courantOnXY");
}

extern "C" void
cuComputeDeltat(double *dt, const hydroparam_t H, hydrowork_t * Hw, hydrovar_t * Hv,
                hydrovarwork_t * Hvw) {
  long j;
  double *uoldDEV, *qDEV, *eDEV, *cDEV, *courantDEV;
  double *courant;
  double deltat;
  double maxCourant = 0;
  long Hnxyt = H.nxyt;
  cudaError_t status;
  long Hmin, Hmax;
  long slices, jend, Hstep, Hnxystep;
  int nops;
  WHERE("compute_deltat");

  //   compute time step on grid interior
  Hnxystep = H.nxystep;
  Hmin = H.jmin + ExtraLayer;
  Hmax = H.jmax - ExtraLayer;

  Hstep = H.nxystep;

  // on recupere les buffers du device qui sont deja alloues
  cuGetUoldQECDevicePtr(&uoldDEV, &qDEV, &eDEV, &cDEV);
  status = cudaMalloc((void **) &courantDEV, Hnxystep * H.nxyt * sizeof(double));
  VERIF(status, "cudaMalloc cuComputeDeltat");
  status = cudaMemset(courantDEV, 0, Hnxystep * H.nxyt * sizeof(double));
  VERIF(status, "cudaMemset cuComputeDeltat");

  double *qIDDEV = &qDEV[IHVWS(0, 0, ID)];
  double *qIPDEV = &qDEV[IHVWS(0, 0, IP)];

  for (j = Hmin; j < Hmax; j += Hstep) {
    jend = j + Hstep;
    if (jend >= Hmax)
      jend = Hmax;
    slices = jend - j;

    cuComputeQEforRows(j, uoldDEV, qDEV, eDEV, H.smallr, H.nx, H.nxt, H.nyt, H.nxyt, slices,
                       Hnxystep);
    nops = slices * H.nx;
    FLOPS(5 * nops, 3 * nops, 1 * nops, 0 * nops);
    cuEquationOfState(0, H.nx, H.smallc, H.gamma, H.nxyt, slices, qIDDEV, eDEV, qIPDEV, cDEV);
    // on calcule courant pour chaque cellule de la ligne pour tous les j
    cuCourantOnXY(courantDEV, H.nx, H.nxyt, slices, Hnxystep, cDEV, qDEV, H.smallc);
    FLOPS(2 * nops, 0 * nops, 5 * nops, 0 * nops);
  }

  //   courant = (double *) malloc(H.nxyt * Hnxystep * sizeof(double));
  //   cudaMemcpy(courant, courantDEV, H.nxyt * Hnxystep * sizeof(double), cudaMemcpyDeviceToHost);
  //   printarray(stdout, courant, H.nxyt, "Courant", H);
  //   free(courant);
  // on cherche le max global des max locaux
  maxCourant = reduceMax(courantDEV, H.nxyt * Hstep);

  deltat = H.courant_factor * H.dx / maxCourant;
  *dt = deltat;
  cudaFree(courantDEV);
  // fprintf(stdout, "compute_deltat: %lg %lg %lg %lg\n", maxCourant, H.courant_factor, H.dx, deltat);
}                               // compute_deltat


//EOF
