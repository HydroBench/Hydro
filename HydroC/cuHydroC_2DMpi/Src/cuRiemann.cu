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
#include <cuda.h>

#include "parametres.h"
#include "utils.h"
#include "cuRiemann.h"
#include "gridfuncs.h"
#include "perfcnt.h"

#define DABS(x) (double) fabs((x))

typedef struct _Args {
  double *qleft;
  double *qright;
  double *qgdnv;
  long *sgnm;
  long narray;
  double Hsmallr;
  double Hsmallc;
  double Hgamma;
  long Hniter_riemann;
  long Hnvar;
  long Hnxyt;
  long Hnxystep;
  int slices;
} Args_t;

// memoire de constante sur le device qui va contenir les arguments de riemann
// on les transmets en bloc en une fois et les differents kernels pourront y acceder.
__constant__ Args_t K;

__global__ void
Loop1KcuRiemann(int *perfvals) {
  double smallp, gamma6, ql, qr, usr, usl, wwl, wwr, smallpp;
  int iter;
  double ulS, rlS;
  double plS, wlS;
  double clS;
  double urS, rrS;
  double prS, wrS;
  double crS;
  double uoS, roS, poS, coS, woS;
  double delpS;
  double poldS;
  double pstarS, ustarS, rstarS, cstar;
  double spoutS, spinS, fracS, ushockS, scrS, sgnmS;
  int indS;
  int Hnxyt = K.Hnxyt;
  int Hnxystep = K.Hnxystep;
  int slices = K.slices;
#define LOOPPERF 1
#if LOOPPERF==1
  int flops[4];
  __shared__ int flopsS[4];
#define CUFLOPS(a, b, c, d) do { flops[0]+=(a);flops[1]+=(b);flops[2]+=(c);flops[3]+=(d);} while (0)
#endif

  int i, j, idx;
  idx = idx1d();
  j = idx / Hnxyt;
  i = idx % Hnxyt;

  if (j >= slices)
    return;
  if (i >= K.narray)
    return;

#if LOOPPERF==1
  flops[0] = 0;
  flops[1] = 0;
  flops[2] = 0;
  flops[3] = 0;
  if (threadIdx.x == 0) {
    flopsS[0] = 0;
    flopsS[1] = 0;
    flopsS[2] = 0;
    flopsS[3] = 0;
  }
#endif

  smallp = Square(K.Hsmallc) / K.Hgamma;

  rlS = MAX(K.qleft[IHVWS(i, j, ID)], K.Hsmallr);
  ulS = K.qleft[IHVWS(i, j, IU)];
  plS = MAX(K.qleft[IHVWS(i, j, IP)], (double) (rlS * smallp));
  rrS = MAX(K.qright[IHVWS(i, j, ID)], K.Hsmallr);
  urS = K.qright[IHVWS(i, j, IU)];
  prS = MAX(K.qright[IHVWS(i, j, IP)], (double) (rrS * smallp));
  // Lagrangian sound speed
  clS = K.Hgamma * plS * rlS;
  crS = K.Hgamma * prS * rrS;
  // First guess
  wlS = sqrt(clS);
  wrS = sqrt(crS);
  pstarS = ((wrS * plS + wlS * prS) + wlS * wrS * (ulS - urS)) / (wlS + wrS);
  pstarS = MAX(pstarS, 0.0);
  poldS = pstarS;
  // indS est un masque de traitement pour le newton
  indS = 1;                     // toutes les cellules sont a traiter

  smallp = Square(K.Hsmallc) / K.Hgamma;
  smallpp = K.Hsmallr * smallp;
  gamma6 = (K.Hgamma + one) / (two * K.Hgamma);

  long indi = indS;

  for (iter = 0; iter < K.Hniter_riemann; iter++) {
    double precision = 1.e-6;
    wwl = sqrt(clS * (one + gamma6 * (poldS - plS) / plS));
    wwr = sqrt(crS * (one + gamma6 * (poldS - prS) / prS));
    ql = two * wwl * Square(wwl) / (Square(wwl) + clS);
    qr = two * wwr * Square(wwr) / (Square(wwr) + crS);
    usl = ulS - (poldS - plS) / wwl;
    usr = urS + (poldS - prS) / wwr;
    double t1 = qr * ql / (qr + ql) * (usl - usr);
    double t2 = -poldS;
    delpS = MAX(t1, t2);
    poldS = poldS + delpS;
    uoS = DABS(delpS / (poldS + smallpp));
    indi = uoS > precision;
#if LOOPPERF==1
    // CUFLOPS(29, 10, 2, 0);
#endif
    if (!indi)
      break;
  }

  gamma6 = (K.Hgamma + one) / (two * K.Hgamma);

  pstarS = poldS;
  wlS = sqrt(clS * (one + gamma6 * (pstarS - plS) / plS));
  wrS = sqrt(crS * (one + gamma6 * (pstarS - prS) / prS));

  ustarS = half * (ulS + (plS - pstarS) / wlS + urS - (prS - pstarS) / wrS);
  sgnmS = (ustarS > 0) ? 1 : -1;
  if (sgnmS == 1) {
    roS = rlS;
    uoS = ulS;
    poS = plS;
    woS = wlS;
  } else {
    roS = rrS;
    uoS = urS;
    poS = prS;
    woS = wrS;
  }
  coS = MAX(K.Hsmallc, sqrt(DABS(K.Hgamma * poS / roS)));
  rstarS = roS / (one + roS * (poS - pstarS) / Square(woS));
  rstarS = MAX(rstarS, K.Hsmallr);
  cstar = MAX(K.Hsmallc, sqrt(DABS(K.Hgamma * pstarS / rstarS)));
  spoutS = coS - sgnmS * uoS;
  spinS = cstar - sgnmS * ustarS;
  ushockS = woS / roS - sgnmS * uoS;
  if (pstarS >= poS) {
    spinS = ushockS;
    spoutS = ushockS;
  }
  scrS = MAX((double) (spoutS - spinS), (double) (K.Hsmallc + DABS(spoutS + spinS)));
  fracS = (one + (spoutS + spinS) / scrS) * half;
  fracS = MAX(zero, (double) (MIN(one, fracS)));

  K.qgdnv[IHVWS(i, j, ID)] = fracS * rstarS + (one - fracS) * roS;
  K.qgdnv[IHVWS(i, j, IU)] = fracS * ustarS + (one - fracS) * uoS;
  K.qgdnv[IHVWS(i, j, IP)] = fracS * pstarS + (one - fracS) * poS;

  if (spoutS < zero) {
    K.qgdnv[IHVWS(i, j, ID)] = roS;
    K.qgdnv[IHVWS(i, j, IU)] = uoS;
    K.qgdnv[IHVWS(i, j, IP)] = poS;
  }
  if (spinS > zero) {
    K.qgdnv[IHVWS(i, j, ID)] = rstarS;
    K.qgdnv[IHVWS(i, j, IU)] = ustarS;
    K.qgdnv[IHVWS(i, j, IP)] = pstarS;
  }

  if (sgnmS == 1) {
    K.qgdnv[IHVWS(i, j, IV)] = K.qleft[IHVWS(i, j, IV)];
  } else {
    K.qgdnv[IHVWS(i, j, IV)] = K.qright[IHVWS(i, j, IV)];
  }
  K.sgnm[IHS(i, j)] = sgnmS;
#if LOOPPERF==1
  atomicAdd(&flopsS[0], flops[0]);
  atomicAdd(&flopsS[1], flops[1]);
  atomicAdd(&flopsS[2], flops[2]);
  atomicAdd(&flopsS[3], flops[3]);
  if (threadIdx.x == 0) {
    atomicAdd(&perfvals[0], flopsS[0]);
    atomicAdd(&perfvals[1], flopsS[1]);
    atomicAdd(&perfvals[2], flopsS[2]);
    atomicAdd(&perfvals[3], flopsS[3]);
  }
#endif
}

__global__ void
Loop10KcuRiemann() {
  int invar;
  int i;
  int Hnxyt = K.Hnxyt;
  int slices = K.slices;
  int Hnxystep = K.Hnxystep;
  int j;

  idx2d(i, j, Hnxyt);
  if (j >= slices)
    return;

  for (invar = IP + 1; invar < K.Hnvar; invar++) {
    if (K.sgnm[IHS(i, j)] == 1) {
      K.qgdnv[IHVWS(i, j, invar)] = K.qleft[IHVWS(i, j, invar)];
    }
    if (K.sgnm[IHS(i, j)] != 1) {
      K.qgdnv[IHVWS(i, j, invar)] = K.qright[IHVWS(i, j, invar)];
    }
  }
}

void
cuRiemann(const long narray, const double Hsmallr, const double Hsmallc, const double Hgamma,   //
          const long Hniter_riemann, const long Hnvar, const long Hnxyt, const int slices, const int Hnxystep,  //
          double *RESTRICT qleftDEV,    // [Hnvar][Hnxystep][Hnxyt]
          double *RESTRICT qrightDEV,   // [Hnvar][Hnxystep][Hnxyt]
          double *RESTRICT qgdnvDEV,    // [Hnvar][Hnxystep][Hnxyt]
          long *RESTRICT sgnmDEV       // [Hnxystep][narray]
  ) {
  // Local variables
  dim3 block, grid;
  Args_t k;
  int nops;

  WHERE("riemann");
  k.qleft = qleftDEV;
  k.qright = qrightDEV;
  k.qgdnv = qgdnvDEV;
  k.sgnm = sgnmDEV;
  //
  k.narray = narray;
  k.Hsmallr = Hsmallr;
  k.Hsmallc = Hsmallc;
  k.Hgamma = Hgamma;
  k.Hniter_riemann = Hniter_riemann;
  k.Hnvar = Hnvar;
  k.Hnxyt = Hnxyt;
  k.Hnxystep = Hnxystep;
  k.slices = slices;

  cudaMemcpyToSymbol(K, &k, sizeof(Args_t), 0, cudaMemcpyHostToDevice);
  CheckErr("cudaMemcpyToSymbol");

  // 64 threads donnent le meilleur rendement compte-tenu de la complexite du kernel
  SetBlockDims(Hnxyt * slices, 192, block, grid);

#if CUDA_VERSION > 2000
  // cudaFuncSetCacheConfig(Loop1KcuRiemann, cudaFuncCachePreferShared);
  cudaFuncSetCacheConfig(Loop1KcuRiemann, cudaFuncCachePreferL1);
  // cudaFuncSetCacheConfig(Loop1KcuRiemann, cudaFuncCachePreferNone);
#endif


  // Pressure, density and velocity
  cuPerfInit();
  Loop1KcuRiemann <<< grid, block >>> (flops_dev);
  CheckErr("Avant synchronize Loop1KcuRiemann");
  cudaThreadSynchronize();
  CheckErr("After synchronize Loop1KcuRiemann");
  cuPerfGet();
  nops = slices * narray;
  FLOPS(57 * nops, 17 * nops, 14 * nops, 0 * nops);

  // other passive variables
  if (Hnvar > IP + 1) {
    Loop10KcuRiemann <<< grid, block >>> ();
    cudaThreadSynchronize();
    CheckErr("After synchronize Loop10KcuRiemann");
  }
}                               // riemann


//EOF
