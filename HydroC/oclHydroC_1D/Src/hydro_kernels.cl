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
idx2d(long *x, long *y, const long nx)
{
  long i1d = get_global_id(0) + (get_global_size(1) - 1) * (get_global_id(1) + (get_global_size(2) - 1) * get_global_id(2));
  *y = i1d / nx;
  *x = i1d - *y * nx;
}

inline double 
Square(const double x)
{
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

#if CPUVERSION == 0
#define Max max
#define Min min
#define Fabs fabs
#define Sqrt sqrt
#else

inline double
Max(const double a, const double b)
{
  return (a > b) ? a : b;
}
inline double
Min(const double a, const double b)
{
  return (a < b) ? a : b;
}
inline double
Fabs(const double a)
{
  return (a > 0) ? a : -a;
}
inline double
Sqrt(const double a)
{
  double v = 0;
  double vn = 0;
  float x0 = (float) a;
  double error = (double) 1.e-8;
  double prec = (double) 1.;
  
  // initial value: to speedup the process we take the float approximation
  x0 = sqrt(x0);
  vn = (double) x0;
  
  prec = Fabs((v - vn) / vn);
  
  if (prec > error)  {
    v = (double) 0.5 * (vn + a / vn);
    prec = Fabs((v - vn) / vn);
    if (prec > error)  {
      v = (double) 0.5 * (vn + a / vn);
      prec = Fabs((v - vn) / vn);
      if (prec > error)  {
	v = (double) 0.5 * (vn + a / vn);
	prec = Fabs((v - vn) / vn);
	if (prec > error)  {
	  v = (double) 0.5 * (vn + a / vn);
	  prec = Fabs((v - vn) / vn);
	  if (prec > error)  {
	    v = (double) 0.5 * (vn + a / vn);
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
IHVW(int i, int v, int Hnxyt)
{
  return (i) + (v) * Hnxyt;
}

inline size_t
IHU(int i, int j, int v, int Hnxt, int Hnyt)
{
  return (i) + (Hnxt * ((j) + Hnyt * (v)));
}

inline size_t
IHV(int i, int j, int v, int Hnxt, int Hnyt)
{
  return (i) + (Hnxt * ((j) + Hnyt * (v)));
}

__kernel void
Loop1KcuCmpflx(__global double *qgdnv, __global double *flux, const long narray, const long Hnxyt, const double Hgamma)
{
  double entho = 0, ekin = 0, etot = 0;
  long i = get_global_id(0);
  if (i >= narray)
    return;

  size_t idxID = IHVW(i, ID, Hnxyt);
  size_t idxIP = IHVW(i, IP, Hnxyt);
  size_t idxIU = IHVW(i, IU, Hnxyt);
  size_t idxIV = IHVW(i, IV, Hnxyt);

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
Loop2KcuCmpflx(__global double *qgdnv, __global double *flux, const long narray, const long Hnxyt, const long Hnvar)
{
  long IN, i = get_global_id(0);
  if (i >= narray)
    return;

  for (IN = IP + 1; IN < Hnvar; IN++) {
    size_t idxIN = IHVW(i, IN, Hnxyt);
    flux[idxIN] = flux[idxIN] * qgdnv[idxIN];
  }
}

__kernel void
LoopKQEforRow(const long j, __global double *uold, __global double *q, __global double *e, const double Hsmallr,
              const long Hnxt, const long Hnyt, const long Hnxyt, const long n)
{
  double eken;
  long i = get_global_id(0);

  if (i >= n)
    return;

  long idxuID = IHV(i + ExtraLayer, j, ID, Hnxt, Hnyt);
  long idxuIU = IHV(i + ExtraLayer, j, IU, Hnxt, Hnyt);
  long idxuIV = IHV(i + ExtraLayer, j, IV, Hnxt, Hnyt);
  long idxuIP = IHV(i + ExtraLayer, j, IP, Hnxt, Hnyt);

  size_t idxID = IHVW(i, ID, Hnxyt);
  size_t idxIP = IHVW(i, IP, Hnxyt);
  size_t idxIU = IHVW(i, IU, Hnxyt);
  size_t idxIV = IHVW(i, IV, Hnxyt);


  q[idxID] = Max(uold[idxuID], Hsmallr);
  q[idxIU] = uold[idxuIU] / q[idxID];
  q[idxIV] = uold[idxuIV] / q[idxID];
  eken = demi * (Square(q[idxIU]) + Square(q[idxIV]));
  q[idxIP] = uold[idxuIP] / q[idxID] - eken;
  e[i] = q[idxIP];
}

__kernel void
LoopKcourant(__global double *q, __global double *courant, const double Hsmallc, __global const double *c,
             const long Hnxyt, const long n)
{
  double cournox, cournoy, courantl;
  long i = get_global_id(0);

  if (i >= n)
    return;

  size_t idxID = IHVW(i, ID, Hnxyt);
  size_t idxIP = IHVW(i, IP, Hnxyt);
  size_t idxIU = IHVW(i, IU, Hnxyt);
  size_t idxIV = IHVW(i, IV, Hnxyt);

  cournox = cournoy = 0.;

  cournox = c[i] + Fabs(q[idxIU]);
  cournoy = c[i] + Fabs(q[idxIV]);
  courantl = Max(cournox, Max(cournoy, Hsmallc));
  courant[i] = Max(courant[i], courantl);
}


__kernel void
Loop1KcuGather(__global double *uold,
               __global double *u,
               const long rowcol, const long Hnxt, const long Himin, const long Himax, const long Hnyt, const long Hnxyt)
{
  long i = get_global_id(0);
  if (i < Himin)
    return;
  if (i >= Himax)
    return;

  size_t idxID = IHVW(i, ID, Hnxyt);
  size_t idxIP = IHVW(i, IP, Hnxyt);
  size_t idxIU = IHVW(i, IU, Hnxyt);
  size_t idxIV = IHVW(i, IV, Hnxyt);

  u[idxID] = uold[IHU(i, rowcol, ID, Hnxt, Hnyt)];
  u[idxIU] = uold[IHU(i, rowcol, IU, Hnxt, Hnyt)];
  u[idxIV] = uold[IHU(i, rowcol, IV, Hnxt, Hnyt)];
  u[idxIP] = uold[IHU(i, rowcol, IP, Hnxt, Hnyt)];
}

__kernel void
Loop2KcuGather(__global double *uold,
               __global double *u,
               const long rowcol, const long Hnxt, const long Himin, const long Himax, const long Hnyt, const long Hnxyt)
{
  long i = get_global_id(0);
  if (i < Himin)
    return;
  if (i >= Himax)
    return;

  size_t idxID = IHVW(i, ID, Hnxyt);
  size_t idxIP = IHVW(i, IP, Hnxyt);
  size_t idxIU = IHVW(i, IU, Hnxyt);
  size_t idxIV = IHVW(i, IV, Hnxyt);

  u[idxID] = uold[IHU(rowcol, i, ID, Hnxt, Hnyt)];
  u[idxIV] = uold[IHU(rowcol, i, IU, Hnxt, Hnyt)];
  u[idxIU] = uold[IHU(rowcol, i, IV, Hnxt, Hnyt)];
  u[idxIP] = uold[IHU(rowcol, i, IP, Hnxt, Hnyt)];
}

__kernel void
Loop3KcuGather(__global double *uold,
               __global double *u,
               const long rowcol,
               const long Hnxt, const long Himin, const long Himax, const long Hnyt, const long Hnxyt, const long Hnvar)
{
  long i = get_global_id(0);
  long ivar;

  if (i < Himin)
    return;
  if (i >= Himax)
    return;

  for (ivar = IP + 1; ivar < Hnvar; ivar++) {
    u[IHVW(i, ivar, Hnxyt)] = uold[IHU(i, rowcol, ivar, Hnxt, Hnyt)];
  }
}

__kernel void
Loop4KcuGather(__global double *uold,
               __global double *u,
               const long rowcol,
               const long Hnxt, const long Himin, const long Himax, const long Hnyt, const long Hnxyt, const long Hnvar)
{
  long i = get_global_id(0);
  long ivar;

  if (i < Himin)
    return;
  if (i >= Himax)
    return;

  // reconsiderer le calcul d'indices en supprimant la boucle sur ivar et
  // en la ventilant par thread
  for (ivar = IP + 1; ivar < Hnvar; ivar++) {
    u[IHVW(i, ivar, Hnxyt)] = uold[IHU(rowcol, i, ivar, Hnxt, Hnyt)];
  }
}

__kernel void
Loop1KcuUpdate(const long rowcol, const double dtdx,
               __global double *uold,
               __global double *u,
               __global double *flux, const long Himin, const long Himax, const long Hnxt, const long Hnyt, const long Hnxyt)
{
  long i = get_global_id(0);


  if (i < (Himin + ExtraLayer))
    return;
  if (i >= (Himax - ExtraLayer))
    return;

  size_t idxID = IHVW(i, ID, Hnxyt);
  size_t idxIP = IHVW(i, IP, Hnxyt);
  size_t idxIU = IHVW(i, IU, Hnxyt);
  size_t idxIV = IHVW(i, IV, Hnxyt);


  uold[IHU(i, rowcol, ID, Hnxt, Hnyt)] = u[idxID] + (flux[IHVW(i - 2, ID, Hnxyt)] - flux[IHVW(i - 1, ID, Hnxyt)]) * dtdx;
  uold[IHU(i, rowcol, IU, Hnxt, Hnyt)] = u[idxIU] + (flux[IHVW(i - 2, IU, Hnxyt)] - flux[IHVW(i - 1, IU, Hnxyt)]) * dtdx;
  uold[IHU(i, rowcol, IV, Hnxt, Hnyt)] = u[idxIV] + (flux[IHVW(i - 2, IV, Hnxyt)] - flux[IHVW(i - 1, IV, Hnxyt)]) * dtdx;
  uold[IHU(i, rowcol, IP, Hnxt, Hnyt)] = u[idxIP] + (flux[IHVW(i - 2, IP, Hnxyt)] - flux[IHVW(i - 1, IP, Hnxyt)]) * dtdx;
}

__kernel void
Loop2KcuUpdate(const long rowcol, const double dtdx,
               __global double *uold,
               __global double *u,
               __global double *flux,
               const long Himin, const long Himax, const long Hnvar, const long Hnxt, const long Hnyt, const long Hnxyt)
{
  long i = get_global_id(0);
  long ivar;

  if (i < (Himin + ExtraLayer))
    return;
  if (i >= (Himax - ExtraLayer))
    return;

  for (ivar = IP + 1; ivar < Hnvar; ivar++) {
    uold[IHU(i, rowcol, ivar, Hnxt, Hnyt)] =
      u[IHVW(i, ivar, Hnxyt)] + (flux[IHVW(i - 2, ivar, Hnxyt)] - flux[IHVW(i - 1, ivar, Hnxyt)]) * dtdx;
  }
}

__kernel void
Loop3KcuUpdate(const long rowcol, const double dtdx,
               __global double *uold,
               __global double *u,
               __global double *flux, const long Hjmin, const long Hjmax, const long Hnxt, const long Hnyt, const long Hnxyt)
{
  long j = get_global_id(0);

  if (j < (Hjmin + ExtraLayer))
    return;
  if (j >= (Hjmax - ExtraLayer))
    return;

  uold[IHU(rowcol, j, ID, Hnxt, Hnyt)] =
    u[IHVW(j, ID, Hnxyt)] + (flux[IHVW(j - 2, ID, Hnxyt)] - flux[IHVW(j - 1, ID, Hnxyt)]) * dtdx;
  uold[IHU(rowcol, j, IP, Hnxt, Hnyt)] =
    u[IHVW(j, IP, Hnxyt)] + (flux[IHVW(j - 2, IP, Hnxyt)] - flux[IHVW(j - 1, IP, Hnxyt)]) * dtdx;
  uold[IHU(rowcol, j, IV, Hnxt, Hnyt)] =
    u[IHVW(j, IU, Hnxyt)] + (flux[IHVW(j - 2, IU, Hnxyt)] - flux[IHVW(j - 1, IU, Hnxyt)]) * dtdx;
  uold[IHU(rowcol, j, IU, Hnxt, Hnyt)] =
    u[IHVW(j, IV, Hnxyt)] + (flux[IHVW(j - 2, IV, Hnxyt)] - flux[IHVW(j - 1, IV, Hnxyt)]) * dtdx;
}

__kernel void
Loop4KcuUpdate(const long rowcol, const double dtdx,
               __global double *uold,
               __global double *u,
               __global double *flux,
               const long Hjmin, const long Hjmax, const long Hnvar, const long Hnxt, const long Hnyt, const long Hnxyt)
{
  long j = get_global_id(0);
  long ivar;

  if (j < (Hjmin + ExtraLayer))
    return;
  if (j >= (Hjmax - ExtraLayer))
    return;

  for (ivar = IP + 1; ivar < Hnvar; ivar++) {
    uold[IHU(rowcol, j, ivar, Hnxt, Hnyt)] =
      u[IHVW(j, ivar, Hnxyt)] + (flux[IHVW(j - 2, ivar, Hnxyt)] - flux[IHVW(j - 1, ivar, Hnxyt)]) * dtdx;
  }
}


__kernel void
Loop1KcuConstoprim(const long n,
                   __global double *u, __global double *q, __global double *e, const long Hnxyt, const double Hsmallr)
{
  double eken;
  long i = get_global_id(0);
  if (i >= n)
    return;

  size_t idxID = IHVW(i, ID, Hnxyt);
  size_t idxIP = IHVW(i, IP, Hnxyt);
  size_t idxIU = IHVW(i, IU, Hnxyt);
  size_t idxIV = IHVW(i, IV, Hnxyt);

  q[idxID] = Max(u[idxID], Hsmallr);
  q[idxIU] = u[idxIU] / q[idxID];
  q[idxIV] = u[idxIV] / q[idxID];
  eken = demi * (Square(q[idxIU]) + Square(q[idxIV]));
  q[idxIP] = u[idxIP] / q[idxID] - eken;
  e[i] = q[idxIP];
}

__kernel void
Loop2KcuConstoprim(const long n, __global double *u, __global double *q, const long Hnxyt, const long Hnvar)
{
  long IN;
  long i = get_global_id(0);
  if (i >= n)
    return;

  for (IN = IP + 1; IN < Hnvar; IN++) {
    size_t idxIN = IHVW(i, IN, Hnxyt);
    q[idxIN] = u[idxIN] / q[idxIN];
  }
}

__kernel void
LoopEOS(__global double *q, __global double *eint,
        __global double *c,
        const long offsetIP, const long offsetID, const long imin, const long imax, const double Hsmallc, const double Hgamma)
{
  double smallp;
  __global double *p = &q[offsetIP];
  __global double *rho = &q[offsetID];
  long k = get_global_id(0);

  if (k < imin)
    return;
  if (k >= imax)
    return;

  smallp = Square(Hsmallc) / Hgamma;
  p[k] = (Hgamma - one) * rho[k] * eint[k];
  p[k] = Max(p[k], (double) (rho[k] * smallp));
  c[k] = Sqrt(Hgamma * p[k] / rho[k]);
}

__kernel void
Loop1KcuMakeBoundary(const long i, const long i0, const double sign, const long Hjmin,
                     const long n, const long Hnxt, const long Hnyt, const long Hnvar, __global double *uold)
{
  long j, ivar;
  double vsign = sign;

  idx2d(&j, &ivar, n);
  if (ivar >= Hnvar)
    return;

  // recuperation de la conditon qui etait dans la boucle
  if (ivar == IU)
    vsign = -1.0;

  j += (Hjmin + ExtraLayer);
  uold[IHV(i, j, ivar, Hnxt, Hnyt)] = uold[IHV(i0, j, ivar, Hnxt, Hnyt)] * vsign;
}

__kernel void
Loop2KcuMakeBoundary(const long j, const long j0, const double sign, const long Himin,
                     const long n, const long Hnxt, const long Hnyt, const long Hnvar, __global double *uold)
{
  long i, ivar;
  double vsign = sign;

  idx2d(&i, &ivar, n);
  if (ivar >= Hnvar)
    return;

  // recuperation de la conditon qui etait dans la boucle
  if (ivar == IV)
    vsign = -1.0;

  i += (Himin + ExtraLayer);
  uold[IHV(i, j, ivar, Hnxt, Hnyt)] = uold[IHV(i, j0, ivar, Hnxt, Hnyt)] * vsign;
}


__kernel void
Loop1KcuQleftright(const long bmax, const long Hnvar, const long Hnxyt,
                   __global double *qxm, __global double *qxp, __global double *qleft, __global double *qright)
{
  long nvar;
  long i = get_global_id(0);
  if (i >= bmax)
    return;

  for (nvar = 0; nvar < Hnvar; nvar++) {
    qleft[IHVW(i, nvar, Hnxyt)] = qxm[IHVW(i + 1, nvar, Hnxyt)];
    qright[IHVW(i, nvar, Hnxyt)] = qxp[IHVW(i + 2, nvar, Hnxyt)];
  }
}

typedef struct _Args {
  __global double *qleft;
  __global double *qright;
  __global double *qgdnv;
  __global double *rl;
  __global double *ul;
  __global double *pl;
  __global double *cl;
  __global double *wl;
  __global double *rr;
  __global double *ur;
  __global double *pr;
  __global double *cr;
  __global double *wr;
  __global double *ro;
  __global double *uo;
  __global double *po;
  __global double *co;
  __global double *wo;
  __global double *rstar;
  __global double *ustar;
  __global double *pstar;
  __global double *cstar;
  __global long *sgnm;
  __global double *spin;
  __global double *spout;
  __global double *ushock;
  __global double *frac;
  __global double *scr;
  __global double *delp;
  __global double *pold;
  __global long *ind;
  __global long *ind2;
  long narray;
  double Hsmallr;
  double Hsmallc;
  double Hgamma;
  long Hniter_riemann;
  long Hnvar;
  long Hnxyt;
} Args_t;

// memoire sur le device qui va contenir les arguments de riemann
// on les transmets en bloc en une fois et les differents kernels pourront y acceder.
//__constant Args_t K;

/* 
   pour contourner les limitations d'OpenCL, nous allons utiliser 4
   kernels d'affectation des pointeurs de la structure. Methode
   bestiale et peu efficace mais qui permet de garder le code intact.

   Remarque: je suis conscient de l'inutilite des tableaux
   intermediaires qui auraient pu etre des scalaires, mais l'objectif
   est de rester (pour l'instant) au plus pres du code d'origine.
*/

__kernel void
Init1KcuRiemann(__global Args_t * K,
                __global double *qleft,
                __global double *qright,
                __global double *qgdnv,
                __global double *rl,
                __global double *ul,
                __global double *pl, __global double *cl, __global double *wl, __global double *rr, __global double *ur)
{
  long tid = get_global_id(0);
  if (tid != 0)
    return;

  K->qleft = qleft;
  K->qright = qright;
  K->qgdnv = qgdnv;
  K->rl = rl;
  K->ul = ul;
  K->pl = pl;
  K->cl = cl;
  K->wl = wl;
  K->rr = rr;
  K->ur = ur;
}
__kernel void
Init2KcuRiemann(__global Args_t * K,
                __global double *pr,
                __global double *cr,
                __global double *wr,
                __global double *ro,
                __global double *uo,
                __global double *po, __global double *co, __global double *wo, __global double *rstar, __global double *ustar)
{
  long tid = get_global_id(0);
  if (tid != 0)
    return;
  K->pr = pr;
  K->cr = cr;
  K->wr = wr;
  K->ro = ro;
  K->uo = uo;
  K->po = po;
  K->co = co;
  K->wo = wo;
  K->rstar = rstar;
  K->ustar = ustar;
}
__kernel void
Init3KcuRiemann(__global Args_t * K,
                __global double *pstar,
                __global double *cstar,
                __global long *sgnm,
                __global double *spin,
                __global double *spout,
                __global double *ushock,
                __global double *frac, __global double *scr, __global double *delp, __global double *pold)
{
  long tid = get_global_id(0);
  if (tid != 0)
    return;
  K->pstar = pstar;
  K->cstar = cstar;
  K->spin = spin;
  K->spout = spout;
  K->ushock = ushock;
  K->frac = frac;
  K->scr = scr;
  K->delp = delp;
  K->pold = pold;
  K->sgnm = sgnm;
}
__kernel void
Init4KcuRiemann(__global Args_t * K, __global long *ind, __global long *ind2)
{
  long tid = get_global_id(0);
  if (tid != 0)
    return;
  K->ind = ind;
  K->ind2 = ind2;
}

__kernel void
LoopKcuSlope(__global double *q, __global double *dq,
             const long Hnvar, const long Hnxyt, const double slope_type, const long ijmin, const long ijmax)
{
  long n;
  double dlft, drgt, dcen, dsgn, slop, dlim;
  long ihvwin, ihvwimn, ihvwipn;

  long i;
  idx2d(&i, &n, (ijmax - ijmin));

  if (n >= Hnvar)
    return;

  i = i + ijmin + 1;
  if (i >= ijmax - 1)
    return;

  ihvwin = IHVW(i, n, Hnxyt);
  ihvwimn = IHVW(i - 1, n, Hnxyt);
  ihvwipn = IHVW(i + 1, n, Hnxyt);
  dlft = slope_type * (q[ihvwin] - q[ihvwimn]);
  drgt = slope_type * (q[ihvwipn] - q[ihvwin]);
  dcen = demi * (dlft + drgt) / slope_type;
  dsgn = (dcen > 0) ? (double) 1.0 : (double) -1.0;     // sign(one, dcen);
  slop = (double) Min(Fabs(dlft), Fabs(drgt));
  dlim = ((dlft * drgt) <= zero) ? zero : slop;
  //         if ((dlft * drgt) <= zero) {
  //             dlim = zero;
  //         }
  dq[ihvwin] = dsgn * (double) Min(dlim, Fabs(dcen));
}

__kernel void
Loop1KcuTrace(__global double *q, __global double *dq, __global double *c,
              __global double *qxm, __global double *qxp,
              const double dtdx, const long Hnxyt,
              const long imin, const long imax, const double zeror, const double zerol, const double project)
{
  double cc, csq, r, u, v, p;
  double dr, du, dv, dp;
  double alpham, alphap, alpha0r, alpha0v;
  double spminus, spplus, spzero;
  double apright, amright, azrright, azv1right;
  double apleft, amleft, azrleft, azv1left;

  long i = get_global_id(0);
  if (i < imin)
    return;
  if (i >= imax)
    return;

  size_t idxIU = IHVW(i, IU, Hnxyt);
  size_t idxIV = IHVW(i, IV, Hnxyt);
  size_t idxIP = IHVW(i, IP, Hnxyt);
  size_t idxID = IHVW(i, ID, Hnxyt);

  cc = c[i];
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
              const long imin, const long imax, const double zeror, const double zerol, const double project)
{
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
reduceMaxDble(__global double *buffer, 
	      __const long length, 
	      __global double *result, 
	      __local double *scratch) 
{
  int global_index = get_global_id(0);
  int local_index  = get_local_id(0);
  double accumulator = -DBL_MAX;
  // Pass 1
  // Loop sequentially over chunks of input vector

  // if (global_index >= length) return;
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

__kernel void
KernelMemset(__global double *a, int v, long lobj)
{
  size_t gid = get_global_id(0);
  if (gid >= lobj)
    return;

  double dv = (double) 0.0 + (double) v;
  a[gid] = dv;
}

__kernel void
KernelMemsetV4(__global int4 * a, int v, long lobj)
{
  size_t gid = get_global_id(0) * 4;
  if (gid >= lobj)
    return;
  a[gid] = (int4) v;
}

__kernel void
Loop1KcuRiemann(__global double *qleft, __global double *qright, __global double *sgnm, __global double *qgdnv, long Hnxyt,
                long Knarray, double Hsmallc, double Hgamma, double Hsmallr, long Hniter_riemann)
{
  double smallp, gamma6, ql, qr, usr, usl, wwl, wwr, smallpp;
  long iter;
  long tid = get_local_id(0);
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

  long i = get_global_id(0);
  if (i >= Knarray)
    return;

  size_t idxIU = IHVW(i, IU, Hnxyt);
  size_t idxIV = IHVW(i, IV, Hnxyt);
  size_t idxIP = IHVW(i, IP, Hnxyt);
  size_t idxID = IHVW(i, ID, Hnxyt);

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
  // ind est un masque de traitement pour le newton
  long Kindi = 1;               // toutes les cellules sont a traiter

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
  sgnm[i] = (Kustari > 0) ? 1 : -1;
  if (sgnm[i] == 1) {
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
  double Kspouti = Kcoi - sgnm[i] * Kuoi;
  double Kspini = Kcstari - sgnm[i] * Kustari;
  double Kushocki = Kwoi / Kroi - sgnm[i] * Kuoi;
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

  if (sgnm[i] == 1) {
    qgdnv[idxIV] = qleft[idxIV];
  } else {
    qgdnv[idxIV] = qright[idxIV];
  }
}

__kernel void
Loop10KcuRiemann(__global Args_t * K, __global double *qleft, __global double *qright, __global double *sgnm,
                 __global double *qgdnv, long Knarray, long Knvar, long KHnxyt)
{
  long invar;
  long i = get_global_id(0);
  long Hnxyt = KHnxyt;
  if (i >= Knarray)
    return;

  for (invar = IP + 1; invar < Knvar; invar++) {
    size_t idxIN = IHVW(i, invar, Hnxyt);
    if (sgnm[i] == 1) {
      qgdnv[idxIN] = qleft[idxIN];
    }
    if (sgnm[i] != 1) {
      qgdnv[idxIN] = qright[idxIN];
    }
  }
}

//EOF
#ifdef NOTDEF
#endif //NOTDEF
