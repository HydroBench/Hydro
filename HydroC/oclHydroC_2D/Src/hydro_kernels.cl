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

#ifdef AMDATI
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif
#ifdef NVIDIA
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif

#ifdef INTEL
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif


#include "oclparam.h"

#define ID     (0)
#define IU     (1)
#define IV     (2)
#define IP     (3)
#define ExtraLayer    (2)
#define ExtraLayerTot (2 * 2)

inline void
idx2d(long *x, long *y, const long nx) {
  long i1d =
    get_global_id(0) + (get_global_size(1) - 1) * (get_global_id(1) + (get_global_size(2) - 1) * get_global_id(2));
  *y = i1d / nx;
  *x = i1d - *y * nx;
}

inline double
Square(const double x) {
  return x * x;
}

/*
 * Here are defined a couple of placeholders of mathematical functions
 * waiting for AMD to provide them as part of the OpenCL language. The
 * implementation is crude and not efficient. It's only to get a
 * running version of the code in double precision.
 */

#ifdef AMDATI
#define CPUVERSION 1
#endif

#ifdef NVIDIA
#define CPUVERSION 0
#endif

#ifdef INTEL
#define CPUVERSION 0
#endif

#if CPUVERSION == 0
#define Max fmax
#define Min fmin
#define Fabs fabs
#define Sqrt sqrt
#else

inline double
Max(const double a, const double b) {
  return (a > b) ? a : b;
}

inline double
Min(const double a, const double b) {
  return (a < b) ? a : b;
}

inline double
Fabs(const double a) {
  return (a > 0) ? a : -a;
}

inline double
Sqrt(const double a) {
  double v = 0;
  double vn = 0;
  float x0 = (float) a;
  double error = (double) 1.e-8;
  double prec = (double) 1.;

  // initial value: to speedup the process we take the float approximation
  x0 = sqrt(x0);
  vn = (double) x0;

  prec = Fabs((v - vn) / vn);

  if (prec > error) {
    v = (double) 0.5 *(vn + a / vn);
    prec = Fabs((v - vn) / vn);
    if (prec > error) {
      vn = v;
      v = (double) 0.5 *(vn + a / vn);
      prec = Fabs((v - vn) / vn);
      if (prec > error) {
        vn = v;
        v = (double) 0.5 *(vn + a / vn);
        prec = Fabs((v - vn) / vn);
        if (prec > error) {
          vn = v;
          v = (double) 0.5 *(vn + a / vn);
          prec = Fabs((v - vn) / vn);
          if (prec > error) {
            vn = v;
            v = (double) 0.5 *(vn + a / vn);
            prec = Fabs((v - vn) / vn);
          }
        }
      }
    }
  }

  return v;
}
#endif
/*
 * End math functions 
 */

#define one (double) 1.0
#define two (double) 2.0
#define demi (double) 0.5
#define zero (double) 0.0

// const double one = 1.0;
// const double two = 2.0;
// const double demi = 0.5;
// const double zero = 0.;

// #define IHVW(i,v) ((i) + (v) * Hnxyt)
// #define IHU(i, j, v)  ((i) + Hnxt * ((j) + Hnyt * (v)))
// #define IHV(i, j, v)  ((i) + Hnxt * ((j) + Hnyt * (v)))
inline size_t
IHVW(int i, int v, int Hnxyt) {
  return (i) + (v) * Hnxyt;
}

inline size_t
IHU(int i, int j, int v, int Hnxt, int Hnyt) {
  return (i) + (Hnxt * ((j) + Hnyt * (v)));
}

inline size_t
IHV(const int i, const int j, const int v, const int Hnxt, const int Hnyt) {
  return (i) + (Hnxt * ((j) + Hnyt * (v)));
}

__kernel void
Loop1KcuCmpflx(__global double *qgdnv, __global double *flux, const long narray,
               const long Hnxyt, const double Hgamma, const int slices, const int Hnxystep) {
  double entho = 0, ekin = 0, etot = 0;
  long s, i = get_global_id(0);

  idx2d(&i, &s, Hnxyt);
  if (s >= slices)
    return;
  if (i >= narray)
    return;

  size_t idxID = IHVWS(i, s, ID, Hnxyt, Hnxystep);
  size_t idxIP = IHVWS(i, s, IP, Hnxyt, Hnxystep);
  size_t idxIU = IHVWS(i, s, IU, Hnxyt, Hnxystep);
  size_t idxIV = IHVWS(i, s, IV, Hnxyt, Hnxystep);

  entho = one / (Hgamma - one);
  // Mass density
  flux[idxID] = qgdnv[idxID] * qgdnv[idxIU];
  // Normal momentum
  flux[idxIU] = flux[idxID] * qgdnv[idxIU] + qgdnv[idxIP];
  // Transverse momentum 1
  flux[idxIV] = flux[idxID] * qgdnv[idxIV];
  // Total energy
  ekin = demi * qgdnv[idxID] * ((qgdnv[idxIU] * qgdnv[idxIU]) + (qgdnv[idxIV] * qgdnv[idxIV]));
  etot = qgdnv[idxIP] * entho + ekin;
  flux[idxIP] = qgdnv[idxIU] * (etot + qgdnv[idxIP]);
}


__kernel void
Loop2KcuCmpflx(__global double *qgdnv, __global double *flux, const long narray, const long Hnxyt, const long Hnvar,
               const int slices, const int Hnxystep) {
  long IN, s, i = get_global_id(0);
  idx2d(&i, &s, Hnxyt);
  if (s >= slices)
    return;

  if (i >= narray)
    return;

  for (IN = IP + 1; IN < Hnvar; IN++) {
    size_t idxIN = IHVWS(i, s, IN, Hnxyt, Hnxystep);
    flux[idxIN] = flux[idxIN] * qgdnv[idxIN];
  }
}

__kernel void
LoopKQEforRow(const long j, __global double *uold, __global double *q, __global double *e,
              const double Hsmallr,
              const long Hnxt, const long Hnyt, const long Hnxyt, const long n, const int slices, const int Hnxystep) {
  double eken;
  long i, s;
  idx2d(&i, &s, Hnxyt);
  if (s >= slices)
    return;

  if (i >= n)
    return;

  long idxuID = IHV(i + ExtraLayer, j + s, ID, Hnxt, Hnyt);
  long idxuIU = IHV(i + ExtraLayer, j + s, IU, Hnxt, Hnyt);
  long idxuIV = IHV(i + ExtraLayer, j + s, IV, Hnxt, Hnyt);
  long idxuIP = IHV(i + ExtraLayer, j + s, IP, Hnxt, Hnyt);

  size_t idxID = IHVWS(i, s, ID, Hnxyt, Hnxystep);
  size_t idxIP = IHVWS(i, s, IP, Hnxyt, Hnxystep);
  size_t idxIU = IHVWS(i, s, IU, Hnxyt, Hnxystep);
  size_t idxIV = IHVWS(i, s, IV, Hnxyt, Hnxystep);
  size_t is = IHS(i, s, Hnxyt);

  q[idxID] = Max(uold[idxuID], Hsmallr);
  q[idxIU] = uold[idxuIU] / q[idxID];
  q[idxIV] = uold[idxuIV] / q[idxID];
  eken = demi * (Square(q[idxIU]) + Square(q[idxIV]));
  q[idxIP] = uold[idxuIP] / q[idxID] - eken;
  e[is] = q[idxIP];
}

__kernel void
LoopKcourant(__global double *q, 
	     __global double *courant, 
	     const double Hsmallc, 
	     __global const double *c,
             const long Hnxyt, const long n, 
	     const int slices, const int Hnxystep) {
  double cournox, cournoy, courantl;
  long i, s;
  idx2d(&i, &s, Hnxyt);
  if (s >= slices)
    return;

  if (i >= n)
    return;

  size_t idxID = IHVWS(i, s, ID, Hnxyt, Hnxystep);
  size_t idxIP = IHVWS(i, s, IP, Hnxyt, Hnxystep);
  size_t idxIU = IHVWS(i, s, IU, Hnxyt, Hnxystep);
  size_t idxIV = IHVWS(i, s, IV, Hnxyt, Hnxystep);
  size_t is = IHS(i, s, Hnxyt);

  cournox = cournoy = 0.;

  cournox = c[is] + Fabs(q[idxIU]);
  cournoy = c[is] + Fabs(q[idxIV]);
  courantl = Max(cournox, Max(cournoy, Hsmallc));
  courant[is] = Max(courant[is], courantl);
}


__kernel void
Loop1KcuGather(__global double *uold,
               __global double *u,
               const long rowcol,
               const long Hnxt, const long Himin, const long Himax, const long Hnyt,
               const long Hnxyt, const int slices, const int Hnxystep) {
  long i = get_global_id(0);
  long s;

  if (i < Himin)
    return;
  if (i >= Himax)
    return;

  for (s = 0; s < slices; s++) {
    u[IHVWS(i, s, ID, Hnxyt, Hnxystep)] = uold[IHU(i, rowcol + s, ID, Hnxt, Hnyt)];
    u[IHVWS(i, s, IU, Hnxyt, Hnxystep)] = uold[IHU(i, rowcol + s, IU, Hnxt, Hnyt)];
    u[IHVWS(i, s, IV, Hnxyt, Hnxystep)] = uold[IHU(i, rowcol + s, IV, Hnxt, Hnyt)];
    u[IHVWS(i, s, IP, Hnxyt, Hnxystep)] = uold[IHU(i, rowcol + s, IP, Hnxt, Hnyt)];
  }
}

__kernel void
Loop2KcuGather(__global double *uold,
               __global double *u,
               const long rowcol, const long Hnxt, const long Himin, const long Himax, const long Hnyt,
               const long Hnxyt, const int slices, const int Hnxystep) {
  long i = get_global_id(0);
  int s;
  if (i < Himin)
    return;
  if (i >= Himax)
    return;

  for (s = 0; s < slices; s++) {
    size_t idxID = IHVWS(i, s, ID, Hnxyt, Hnxystep);
    size_t idxIP = IHVWS(i, s, IP, Hnxyt, Hnxystep);
    size_t idxIU = IHVWS(i, s, IU, Hnxyt, Hnxystep);
    size_t idxIV = IHVWS(i, s, IV, Hnxyt, Hnxystep);

    u[idxID] = uold[IHU(rowcol + s, i, ID, Hnxt, Hnyt)];
    u[idxIV] = uold[IHU(rowcol + s, i, IU, Hnxt, Hnyt)];
    u[idxIU] = uold[IHU(rowcol + s, i, IV, Hnxt, Hnyt)];
    u[idxIP] = uold[IHU(rowcol + s, i, IP, Hnxt, Hnyt)];
  }
}

__kernel void
Loop3KcuGather(__global double *uold,
               __global double *u,
               const long rowcol,
               const long Hnxt, const long Himin, const long Himax, const long Hnyt, const long Hnxyt, const long Hnvar,
               const int slices, const int Hnxystep) {
  long i = get_global_id(0);
  long ivar;
  int s;

  if (i < Himin)
    return;
  if (i >= Himax)
    return;

  for (ivar = IP + 1; ivar < Hnvar; ivar++) {
    for (s = 0; s < slices; s++) {
      u[IHVWS(i, s, ivar, Hnxyt, Hnxystep)] = uold[IHU(i, rowcol + s, ivar, Hnxt, Hnyt)];
    }
  }
}

__kernel void
Loop4KcuGather(__global double *uold,
               __global double *u,
               const long rowcol,
               const long Hnxt, const long Himin, const long Himax, const long Hnyt, const long Hnxyt, const long Hnvar,
               const int slices, const int Hnxystep) {
  long i = get_global_id(0);
  long ivar;

  if (i < Himin)
    return;
  if (i >= Himax)
    return;

  int s;
  for (ivar = IP + 1; ivar < Hnvar; ivar++) {
    for (s = 0; s < slices; s++) {
      u[IHVWS(i, s, ivar, Hnxyt, Hnxystep)] = uold[IHU(rowcol + s, i, ivar, Hnxt, Hnyt)];
    }
  }
}

__kernel void
Loop1KcuUpdate(const long rowcol, const double dtdx,
               __global double *uold,
               __global double *u,
               __global double *flux, const long Himin, const long Himax, const long Hnxt, const long Hnyt,
               const long Hnxyt, const int slices, const int Hnxystep) {
  long i, s;
  idx2d(&i, &s, Hnxyt);

  if (s >= slices)
    return;

  if (i < (Himin + ExtraLayer))
    return;
  if (i >= (Himax - ExtraLayer))
    return;

  size_t idxID = IHVWS(i, s, ID, Hnxyt, Hnxystep);
  size_t idxIP = IHVWS(i, s, IP, Hnxyt, Hnxystep);
  size_t idxIU = IHVWS(i, s, IU, Hnxyt, Hnxystep);
  size_t idxIV = IHVWS(i, s, IV, Hnxyt, Hnxystep);


  uold[IHU(i, rowcol + s, ID, Hnxt, Hnyt)] =
    u[idxID] + (flux[IHVWS(i - 2, s, ID, Hnxyt, Hnxystep)] - flux[IHVWS(i - 1, s, ID, Hnxyt, Hnxystep)]) * dtdx;
  uold[IHU(i, rowcol + s, IU, Hnxt, Hnyt)] =
    u[idxIU] + (flux[IHVWS(i - 2, s, IU, Hnxyt, Hnxystep)] - flux[IHVWS(i - 1, s, IU, Hnxyt, Hnxystep)]) * dtdx;
  uold[IHU(i, rowcol + s, IV, Hnxt, Hnyt)] =
    u[idxIV] + (flux[IHVWS(i - 2, s, IV, Hnxyt, Hnxystep)] - flux[IHVWS(i - 1, s, IV, Hnxyt, Hnxystep)]) * dtdx;
  uold[IHU(i, rowcol + s, IP, Hnxt, Hnyt)] =
    u[idxIP] + (flux[IHVWS(i - 2, s, IP, Hnxyt, Hnxystep)] - flux[IHVWS(i - 1, s, IP, Hnxyt, Hnxystep)]) * dtdx;
}

__kernel void
Loop2KcuUpdate(const long rowcol, const double dtdx,
               __global double *uold,
               __global double *u,
               __global double *flux,
               const long Himin, const long Himax, const long Hnvar, const long Hnxt, const long Hnyt, const long Hnxyt,
               const int slices, const int Hnxystep) {
  long ivar;
  long i, s;
  idx2d(&i, &s, Hnxyt);
  if (s >= slices)
    return;


  if (i < (Himin + ExtraLayer))
    return;
  if (i >= (Himax - ExtraLayer))
    return;

  for (ivar = IP + 1; ivar < Hnvar; ivar++) {
    uold[IHU(i, rowcol + s, ivar, Hnxt, Hnyt)] =
      u[IHVWS(i, s, ivar, Hnxyt, Hnxystep)] + (flux[IHVWS(i - 2, s, ivar, Hnxyt, Hnxystep)] -
                                               flux[IHVWS(i - 1, s, ivar, Hnxyt, Hnxystep)]) * dtdx;
  }
}

__kernel void
Loop3KcuUpdate(const long rowcol, const double dtdx,
               __global double *uold,
               __global double *u,
               __global double *flux, const long Hjmin, const long Hjmax, const long Hnxt, const long Hnyt,
               const long Hnxyt, const int slices, const int Hnxystep) {
  long ivar;
  long s, j;
  idx2d(&j, &s, Hnxyt);
  if (s >= slices)
    return;

  if (j < (Hjmin + ExtraLayer))
    return;
  if (j >= (Hjmax - ExtraLayer))
    return;

  uold[IHU(rowcol + s, j, ID, Hnxt, Hnyt)] =
    u[IHVWS(j, s, ID, Hnxyt, Hnxystep)] + (flux[IHVWS(j - 2, s, ID, Hnxyt, Hnxystep)] -
                                           flux[IHVWS(j - 1, s, ID, Hnxyt, Hnxystep)]) * dtdx;
  uold[IHU(rowcol + s, j, IP, Hnxt, Hnyt)] =
    u[IHVWS(j, s, IP, Hnxyt, Hnxystep)] + (flux[IHVWS(j - 2, s, IP, Hnxyt, Hnxystep)] -
                                           flux[IHVWS(j - 1, s, IP, Hnxyt, Hnxystep)]) * dtdx;
  uold[IHU(rowcol + s, j, IV, Hnxt, Hnyt)] =
    u[IHVWS(j, s, IU, Hnxyt, Hnxystep)] + (flux[IHVWS(j - 2, s, IU, Hnxyt, Hnxystep)] -
                                           flux[IHVWS(j - 1, s, IU, Hnxyt, Hnxystep)]) * dtdx;
  uold[IHU(rowcol + s, j, IU, Hnxt, Hnyt)] =
    u[IHVWS(j, s, IV, Hnxyt, Hnxystep)] + (flux[IHVWS(j - 2, s, IV, Hnxyt, Hnxystep)] -
                                           flux[IHVWS(j - 1, s, IV, Hnxyt, Hnxystep)]) * dtdx;
}

__kernel void
Loop4KcuUpdate(const long rowcol, const double dtdx,
               __global double *uold,
               __global double *u,
               __global double *flux,
               const long Hjmin, const long Hjmax, const long Hnvar, const long Hnxt, const long Hnyt, const long Hnxyt,
               const int slices, const int Hnxystep) {
  long ivar;
  long s, j;
  idx2d(&j, &s, Hnxyt);
  if (s >= slices)
    return;

  if (j < (Hjmin + ExtraLayer))
    return;
  if (j >= (Hjmax - ExtraLayer))
    return;

  for (ivar = IP + 1; ivar < Hnvar; ivar++) {
    uold[IHU(rowcol + s, j, ivar, Hnxt, Hnyt)] =
      u[IHVWS(j, s, ivar, Hnxyt, Hnxystep)] + (flux[IHVWS(j - 2, s, ivar, Hnxyt, Hnxystep)] -
                                               flux[IHVWS(j - 1, s, ivar, Hnxyt, Hnxystep)]) * dtdx;
  }
}


__kernel void
Loop1KcuConstoprim(const long n,
                   __global double *u, __global double *q, __global double *e,
                   const long Hnxyt, const double Hsmallr, const int slices, const int Hnxystep) {
  double eken;
  int idx = get_global_id(0);
  long i, s;

  s = idx / Hnxyt;
  i = idx % Hnxyt;

  idx2d(&i, &s, Hnxyt);

  if (s >= slices)
    return;

  if (i >= n)
    return;

  size_t idxID = IHVWS(i, s, ID, Hnxyt, Hnxystep);
  size_t idxIP = IHVWS(i, s, IP, Hnxyt, Hnxystep);
  size_t idxIU = IHVWS(i, s, IU, Hnxyt, Hnxystep);
  size_t idxIV = IHVWS(i, s, IV, Hnxyt, Hnxystep);

  q[idxID] = Max(u[idxID], Hsmallr);
  q[idxIU] = u[idxIU] / q[idxID];
  q[idxIV] = u[idxIV] / q[idxID];
  eken = demi * (Square(q[idxIU]) + Square(q[idxIV]));
  q[idxIP] = u[idxIP] / q[idxID] - eken;
  e[IHS(i, s, Hnxyt)] = q[idxIP];
}

__kernel void
Loop2KcuConstoprim(const long n, __global double *u, __global double *q,
                   const long Hnxyt, const long Hnvar, const int slices, const int Hnxystep) {
  long IN;
  long i, idx = get_global_id(0);
  int s;

  s = idx / Hnxyt;
  i = idx % Hnxyt;

  if (s >= slices)
    return;

  if (i >= n)
    return;

  for (IN = IP + 1; IN < Hnvar; IN++) {
    size_t idxIN = IHVWS(i, s, IN, Hnxyt, Hnxystep);
    q[idxIN] = u[idxIN] / q[idxIN];
  }
}

__kernel void
LoopEOS(__global double *q,
        __global double *eint,
        __global double *c,
        const long offsetIP, const long offsetID, const long imin, const long imax,
        const double Hsmallc, const double Hgamma, const int slices, const int Hnxyt) {
  double smallp;
  __global double *p = &q[offsetIP];
  __global double *rho = &q[offsetID];
  long s, k;

  idx2d(&k, &s, Hnxyt);
  if (s >= slices)
    return;
  if (k < imin)
    return;
  if (k >= imax)
    return;

  smallp = Square(Hsmallc) / Hgamma;
  int is = IHS(k, s, Hnxyt);
  p[is] = (Hgamma - one) * rho[is] * eint[is];
  p[is] = Max(p[is], (double) (rho[is] * smallp));
  c[is] = Sqrt(Hgamma * p[is] / rho[is]);
}

__kernel void
Loop1KcuMakeBoundary(const int i, const int i0, const double sign, const long Hjmin,
                     const long n, const long Hnxt, const long Hnyt, const long Hnvar, __global double *uold) {
  long j, ivar;
  double vsign = sign;

  idx2d(&j, &ivar, n);
  if (ivar >= Hnvar)
    return;

  if (ivar == IU)
    vsign = -1.0;

  j += (Hjmin + ExtraLayer);
  uold[IHV(i, j, ivar, Hnxt, Hnyt)] = uold[IHV(i0, j, ivar, Hnxt, Hnyt)] * vsign;
}

__kernel void
Loop2KcuMakeBoundary(const int j, const int j0, const double sign, const long Himin,
                     const long n, const long Hnxt, const long Hnyt, const long Hnvar, __global double *uold) {
  long i, ivar;
  double vsign = sign;

  idx2d(&i, &ivar, n);
  if (ivar >= Hnvar)
    return;

  if (ivar == IV)
    vsign = -1.0;

  i += (Himin + ExtraLayer);
  uold[IHV(i, j, ivar, Hnxt, Hnyt)] = uold[IHV(i, j0, ivar, Hnxt, Hnyt)] * vsign;
}


__kernel void
Loop1KcuQleftright(const long bmax, const long Hnvar, const long Hnxyt, const int slices, const int Hstep,
                   __global double *qxm, __global double *qxp, __global double *qleft, __global double *qright) {
  long nvar;
  long i, s;
  idx2d(&i, &s, Hnxyt);
  if (s >= slices)
    return;

  if (i >= bmax)
    return;

  for (nvar = 0; nvar < Hnvar; nvar++) {
    qleft[IHVWS(i, s, nvar, Hnxyt, Hstep)] = qxm[IHVWS(i + 1, s, nvar, Hnxyt, Hstep)];
    qright[IHVWS(i, s, nvar, Hnxyt, Hstep)] = qxp[IHVWS(i + 2, s, nvar, Hnxyt, Hstep)];
  }
}

__kernel void
LoopKcuSlope(__global double *q, __global double *dq,
             const long Hnvar, const long Hnxyt,
             const double slope_type, const long ijmin, const long ijmax, const int slices, const int Hnxystep) {
  int n;
  double dlft, drgt, dcen, dsgn, slop, dlim;
  long ihvwin, ihvwimn, ihvwipn;

  long i, s;
  idx2d(&i, &s, Hnxyt);

  if (s >= slices)
    return;

  i = i + ijmin + 1;
  if (i >= ijmax - 1)
    return;

  for (n = 0; n < Hnvar; n++) {
    ihvwin =  IHVWS(i,     s, n, Hnxyt, Hnxystep);
    ihvwimn = IHVWS(i - 1, s, n, Hnxyt, Hnxystep);
    ihvwipn = IHVWS(i + 1, s, n, Hnxyt, Hnxystep);
    dlft = slope_type * (q[ihvwin] - q[ihvwimn]);
    drgt = slope_type * (q[ihvwipn] - q[ihvwin]);
    dcen = demi * (dlft + drgt) / slope_type;
    dsgn = (dcen > 0) ? (double) 1.0 : (double) -1.0;   // sign(one, dcen);
    slop = (double) Min(Fabs(dlft), Fabs(drgt));
    dlim = ((dlft * drgt) <= zero) ? zero : slop;
    //         if ((dlft * drgt) <= zero) {
    //             dlim = zero;
    //         }
    dq[ihvwin] = dsgn * (double) Min(dlim, Fabs(dcen));
  }
}

__kernel void
Loop1KcuTrace(__global double *q, __global double *dq, __global double *c,
              __global double *qxm, __global double *qxp,
              const double dtdx, const long Hnxyt,
              const long imin, const long imax, const double zeror, const double zerol,
              const double project, const int slices, const int Hnxystep) {
  double cc, csq, r, u, v, p;
  double dr, du, dv, dp;
  double alpham, alphap, alpha0r, alpha0v;
  double spminus, spplus, spzero;
  double apright, amright, azrright, azv1right;
  double apleft, amleft, azrleft, azv1left;

  long idx = get_global_id(0);
  int i, s;

  s = idx / Hnxyt;
  i = idx % Hnxyt;

  if (s >= slices)
    return;

  if (i < imin)
    return;
  if (i >= imax)
    return;

  size_t idxIU = IHVWS(i, s, IU, Hnxyt, Hnxystep);
  size_t idxIV = IHVWS(i, s, IV, Hnxyt, Hnxystep);
  size_t idxIP = IHVWS(i, s, IP, Hnxyt, Hnxystep);
  size_t idxID = IHVWS(i, s, ID, Hnxyt, Hnxystep);
  size_t is = IHS(i, s, Hnxyt);

  cc = c[is];
  csq = Square(cc);
  r = q[idxID];
  u = q[idxIU];
  v = q[idxIV];
  p = q[idxIP];
  dr = dq[idxID];
  du = dq[idxIU];
  dv = dq[idxIV];
  dp = dq[idxIP];
  alpham = demi * (dp / (r * cc) - du) * r / cc;
  alphap = demi * (dp / (r * cc) + du) * r / cc;
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
  apright = -demi * spplus * alphap;
  amright = -demi * spminus * alpham;
  azrright = -demi * spzero * alpha0r;
  azv1right = -demi * spzero * alpha0v;
  qxp[idxID] = r + (apright + amright + azrright);
  qxp[idxIU] = u + (apright - amright) * cc / r;
  qxp[idxIV] = v + (azv1right);
  qxp[idxIP] = p + (apright + amright) * csq;

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
  apleft = -demi * spplus * alphap;
  amleft = -demi * spminus * alpham;
  azrleft = -demi * spzero * alpha0r;
  azv1left = -demi * spzero * alpha0v;
  qxm[idxID] = r + (apleft + amleft + azrleft);
  qxm[idxIU] = u + (apleft - amleft) * cc / r;
  qxm[idxIV] = v + (azv1left);
  qxm[idxIP] = p + (apleft + amleft) * csq;
}

__kernel void
Loop2KcuTrace(__global double *q, __global double *dq,
              __global double *qxm, __global double *qxp,
              const double dtdx, const long Hnvar, const long Hnxyt,
              const long imin, const long imax, const double zeror, const double zerol, const double project) {
  long IN;
  double u, a;
  double da;
  double spzero;
  double acmpright;
  double acmpleft;

  long i = get_global_id(0);
  if (i < imin)
    return;
  if (i >= imax)
    return;

  size_t idxIU = IHVW(i, IU, Hnxyt);
  size_t idxIV = IHVW(i, IV, Hnxyt);
  size_t idxIP = IHVW(i, IP, Hnxyt);
  size_t idxID = IHVW(i, ID, Hnxyt);

  for (IN = IP + 1; IN < Hnvar; IN++) {
    size_t idxIN = IHVW(i, IN, Hnxyt);
    u = q[idxIU];
    a = q[idxIN];
    da = dq[idxIN];

    // Right state
    spzero = u * dtdx + one;
    if (u >= zeror) {
      spzero = project;
    }
    acmpright = -demi * spzero * da;
    qxp[idxIN] = a + acmpright;

    // Left state
    spzero = u * dtdx - one;
    if (u <= zerol) {
      spzero = -project;
    }
    acmpleft = -demi * spzero * da;
    qxm[idxIN] = a + acmpleft;
  }
}

__kernel void
KernelMemset(__global double *a, int v, long lobj) {
  size_t gid = get_global_id(0);
  if (gid >= lobj)
    return;

  double dv = (double) v;
  a[gid] = dv;
}

__kernel void
KernelMemsetV4(__global int4 * a, int v, long lobj) {
  size_t gid = get_global_id(0) * 4;
  if (gid >= lobj)
    return;
  a[gid] = (int4) v;
}

__kernel void
Loop1KcuRiemann(__global double *qleft, __global double *qright,
                __global double *sgnm, __global double *qgdnv,
                long Hnxyt, long Knarray, double Hsmallc,
                double Hgamma, double Hsmallr, long Hniter_riemann, const int slices, const int HStep) {
  double smallp, gamma6, ql, qr, usr, usl, wwl, wwr, smallpp;
  long iter;
  double ulS = 0.0;
  double plS = 0.0;
  double clS = 0.0;
  double urS = 0.0;
  double prS = 0.0;
  double crS = 0.0;
  double uoS = 0.0;
  double delpS = 0.0;
  double poldS = 0.0;
  double Kroi = 0.0;
  double Kuoi = 0.0;
  double Kpoi = 0.0;
  double Kwoi = 0.0;
  double Kdelpi = 0.0;

  int s, i, j, idx = get_global_id(0);
  s = idx / Hnxyt;
  i = idx % Hnxyt;

  if (s >= slices)
    return;

  if (i >= Knarray)
    return;

  size_t idxIU = IHVWS(i, s, IU, Hnxyt, HStep);
  size_t idxIV = IHVWS(i, s, IV, Hnxyt, HStep);
  size_t idxIP = IHVWS(i, s, IP, Hnxyt, HStep);
  size_t idxID = IHVWS(i, s, ID, Hnxyt, HStep);
  size_t is = IHS(i, s, Hnxyt);

  smallp = Square(Hsmallc) / Hgamma;

  double Krli = Max(qleft[idxID], Hsmallr);
  double Kuli = qleft[idxIU];
  // opencl explose au dela de cette ligne si le code n'est pas en commentaire
  double Kpli = Max(qleft[idxIP], (double) (Krli * smallp));
  double Krri = Max(qright[idxID], Hsmallr);
  double Kuri = qright[idxIU];
  double Kpri = Max(qright[idxIP], (double) (Krri * smallp));
  // Lagrangian sound speed
  double Kcli = Hgamma * Kpli * Krli;
  double Kcri = Hgamma * Kpri * Krri;
  // First guess
  double Kwli = Sqrt(Kcli);
  double Kwri = Sqrt(Kcri);
  double Kpstari = ((Kwri * Kpli + Kwli * Kpri) + Kwli * Kwri * (Kuli - Kuri)) / (Kwli + Kwri);
  Kpstari = Max(Kpstari, 0.0);
  double Kpoldi = Kpstari;
  // indi is a mask for the newton
  long Kindi = 1;               // should we go on processing the cell 

  ulS = Kuli;
  plS = Kpli;
  clS = Kcli;
  urS = Kuri;
  prS = Kpri;
  crS = Kcri;
  uoS = Kuoi;
  delpS = Kdelpi;
  poldS = Kpoldi;

  smallp = Square(Hsmallc) / Hgamma;
  smallpp = Hsmallr * smallp;
  gamma6 = (Hgamma + one) / (two * Hgamma);

  long indi = Kindi;

  for (iter = 0; iter < Hniter_riemann; iter++) {
    double precision = 1.e-6;
    wwl = Sqrt(clS * (one + gamma6 * (poldS - plS) / plS));
    wwr = Sqrt(crS * (one + gamma6 * (poldS - prS) / prS));
    ql = two * wwl * Square(wwl) / (Square(wwl) + clS);
    qr = two * wwr * Square(wwr) / (Square(wwr) + crS);
    usl = ulS - (poldS - plS) / wwl;
    usr = urS + (poldS - prS) / wwr;
    double t1 = qr * ql / (qr + ql) * (usl - usr);
    double t2 = -poldS;
    delpS = Max(t1, t2);
    poldS = poldS + delpS;
    uoS = Fabs(delpS / (poldS + smallpp));
    indi = uoS > precision;
    if (!indi)
      break;
  }
  // barrier(CLK_LOCAL_MEM_FENCE);
  Kuoi = uoS;
  Kpoldi = poldS;

  gamma6 = (Hgamma + one) / (two * Hgamma);

  Kpstari = Kpoldi;
  Kwli = Sqrt(Kcli * (one + gamma6 * (Kpstari - Kpli) / Kpli));
  Kwri = Sqrt(Kcri * (one + gamma6 * (Kpstari - Kpri) / Kpri));

  double Kustari = demi * (Kuli + (Kpli - Kpstari) / Kwli + Kuri - (Kpri - Kpstari) / Kwri);
  sgnm[is] = (Kustari > 0) ? 1 : -1;
  if (sgnm[is] == 1) {
    Kroi = Krli;
    Kuoi = Kuli;
    Kpoi = Kpli;
    Kwoi = Kwli;
  } else {
    Kroi = Krri;
    Kuoi = Kuri;
    Kpoi = Kpri;
    Kwoi = Kwri;
  }
  double Kcoi = Max(Hsmallc, Sqrt(Fabs(Hgamma * Kpoi / Kroi)));
  double Krstari = Kroi / (one + Kroi * (Kpoi - Kpstari) / Square(Kwoi));
  Krstari = Max(Krstari, Hsmallr);
  double Kcstari = Max(Hsmallc, Sqrt(Fabs(Hgamma * Kpstari / Krstari)));
  double Kspouti = Kcoi - sgnm[is] * Kuoi;
  double Kspini = Kcstari - sgnm[is] * Kustari;
  double Kushocki = Kwoi / Kroi - sgnm[is] * Kuoi;
  if (Kpstari >= Kpoi) {
    Kspini = Kushocki;
    Kspouti = Kushocki;
  }
  double Kscri = Max((double) (Kspouti - Kspini), (double) (Hsmallc + Fabs(Kspouti + Kspini)));
  double Kfraci = (one + (Kspouti + Kspini) / Kscri) * demi;
  Kfraci = Max(zero, (double) (Min(one, Kfraci)));

  qgdnv[idxID] = Kfraci * Krstari + (one - Kfraci) * Kroi;
  qgdnv[idxIU] = Kfraci * Kustari + (one - Kfraci) * Kuoi;
  qgdnv[idxIP] = Kfraci * Kpstari + (one - Kfraci) * Kpoi;

  if (Kspouti < zero) {
    qgdnv[idxID] = Kroi;
    qgdnv[idxIU] = Kuoi;
    qgdnv[idxIP] = Kpoi;
  }
  if (Kspini > zero) {
    qgdnv[idxID] = Krstari;
    qgdnv[idxIU] = Kustari;
    qgdnv[idxIP] = Kpstari;
  }

  if (sgnm[is] == 1) {
    qgdnv[idxIV] = qleft[idxIV];
  } else {
    qgdnv[idxIV] = qright[idxIV];
  }
}

__kernel void
Loop10KcuRiemann(__global double *qleft, __global double *qright, __global double *sgnm,
                 __global double *qgdnv, long Knarray, long Knvar, long KHnxyt, const int slices, const int Hstep) {
  long invar;
  long s, i = get_global_id(0);
  long Hnxyt = KHnxyt;
  idx2d(&i, &s, Hnxyt);
  if (s >= slices)
    return;
  if (i >= Knarray)
    return;

  size_t is = IHS(i, s, Hnxyt);
  for (invar = IP + 1; invar < Knvar; invar++) {
    size_t idxIN = IHVWS(i, s, invar, Hnxyt, Hstep);
    if (sgnm[is] == 1) {
      qgdnv[idxIN] = qleft[idxIN];
    } else {
      qgdnv[idxIN] = qright[idxIN];
    }
  }
}

// kernel to pack/unpack arrays used in MPI exchanges
// #define IHv(i,j,v) ((i) + (Hnxt * (Hnyt * (v) + (j))))
#define IHv2v(i,j,v) ((i) + (ExtraLayer * (Hnyt * (v) + (j))))
#define IHv2h(i,j,v) ((i) + (Hnxt * (ExtraLayer * (v) + (j))))

__kernel void
kpack_arrayv(const int xmin, const long Hnxt, const long Hnyt, const long Hnvar, __global double *buffer,
             __global double *uold) {
  int ivar, i;
  int j = get_global_id(0);
  if (j >= Hnyt)
    return;

  for (ivar = 0; ivar < Hnvar; ivar++) {
    for (i = xmin; i < xmin + ExtraLayer; i++) {
      buffer[IHv2v(i - xmin, j, ivar)] = uold[IHV(i, j, ivar, Hnxt, Hnyt)];
    }
  }
}

__kernel void
kunpack_arrayv(const int xmin, const long Hnxt, const long Hnyt, const long Hnvar, __global double *buffer,
               __global double *uold) {
  int ivar, i;
  int j = get_global_id(0);
  if (j >= Hnyt)
    return;

  for (ivar = 0; ivar < Hnvar; ivar++) {
    for (i = xmin; i < xmin + ExtraLayer; i++) {
      uold[IHV(i, j, ivar, Hnxt, Hnyt)] = buffer[IHv2v(i - xmin, j, ivar)];
    }
  }
}

__kernel void
kpack_arrayh(const int ymin, const long Hnxt, const long Hnyt, const long Hnvar, __global double *buffer,
             __global double *uold) {
  int ivar, j;
  int i = get_global_id(0);
  if (i >= Hnxt)
    return;

  for (ivar = 0; ivar < Hnvar; ivar++) {
    for (j = ymin; j < ymin + ExtraLayer; j++) {
      buffer[IHv2h(i, j - ymin, ivar)] = uold[IHV(i, j, ivar, Hnxt, Hnyt)];
    }
  }
}

__kernel void
kunpack_arrayh(const int ymin, const long Hnxt, const long Hnyt, const long Hnvar, __global double *buffer,
               __global double *uold) {
  int ivar, j;
  int i = get_global_id(0);
  if (i >= Hnxt)
    return;

  for (ivar = 0; ivar < Hnvar; ivar++) {
    for (j = ymin; j < ymin + ExtraLayer; j++) {
      uold[IHV(i, j, ivar, Hnxt, Hnyt)] = buffer[IHv2h(i, j - ymin, ivar)];
    }
  }
}

#define REDUCALGO 1

#if REDUCALGO == 2
__kernel void
reduceMaxDble(__global double *buffer, __const long length, __global double *result, __local double *scratch) {
  int global_index = get_global_id(0);
  int i;
  double lmaxCourant;
  if (global_index == 0) {
    lmaxCourant = 0.;
    for (i = 0; i < length; i++) {
      lmaxCourant = fmax(lmaxCourant, buffer[i]);
    }
    
    result[get_group_id(0)] = lmaxCourant;
  }
}
#endif
#if REDUCALGO == 1
__kernel void
reduceMaxDble(__global double *buffer, 
	      __const long length, 
	      __global double *result, 
	      __local double *scratch) {
  int global_index = get_global_id(0);
  int local_index  = get_local_id(0);
  double accumulator = -DBL_MAX;
  // Pass 1
  // Loop sequentially over chunks of input vector

  while (global_index < length) {
    double element = buffer[global_index];
    accumulator = fmax(accumulator, element);
    global_index += get_local_size(0);  // to favor coalescing
  }

  // Pass 2
  // Perform parallel reduction
  scratch[local_index] = accumulator;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int offset = get_local_size(0) / 2; offset > 0; offset = offset / 2) {
    if (local_index < offset) {
      double other = scratch[local_index + offset];
      double mine = scratch[local_index];
      scratch[local_index] = fmax(mine, other);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (local_index == 0) {
    result[get_group_id(0)] = scratch[0];
  }
}
#endif

//EOF
#ifdef NOTDEF
#endif //NOTDEF
