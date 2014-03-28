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
#include <string.h>
#include <malloc.h>
// #include <unistd.h>
#include <math.h>

#ifdef HMPP
#undef HMPP
#endif

#include "parametres.h"
#include "compute_deltat.h"
#include "utils.h"
#include "perfcnt.h"
#include "equation_of_state.h"

#define DABS(x) (real_t) fabs((x))

inline void
ComputeQEforRow(const int j,
                const real_t Hsmallr,
                const int Hnx,
                const int Hnxt,
                const int Hnyt,
                const int Hnxyt,
                const int Hnvar,
                const int slices, const int Hstep, 
		real_t * uold, 
		real_t q[Hnvar][Hstep][Hnxyt], real_t e[Hstep][Hnxyt]
		) {
  int i, s = slices;

#define IHV(i, j, v)  ((i) + Hnxt * ((j) + Hnyt * (v)))

#pragma simd
  for (i = 0; i < Hnx; i++) {
    real_t eken;
    real_t tmp;
    int idxuID = IHV(i + ExtraLayer, j, ID);
    int idxuIU = IHV(i + ExtraLayer, j, IU);
    int idxuIV = IHV(i + ExtraLayer, j, IV);
    int idxuIP = IHV(i + ExtraLayer, j, IP);
    q[ID][s][i] = MAX(uold[idxuID], Hsmallr);
    q[IU][s][i] = uold[idxuIU] / q[ID][s][i];
    q[IV][s][i] = uold[idxuIV] / q[ID][s][i];
    eken = half * (Square(q[IU][s][i]) + Square(q[IV][s][i]));
    tmp = uold[idxuIP] / q[ID][s][i] - eken;
    q[IP][s][i] = tmp;
    e[s][i] = tmp;
  }
  { 
    int nops = slices * Hnx;
    FLOPS(5 * nops, 3 * nops, 1 * nops, 0 * nops);
  }
#undef IHV
#undef IHVW
}

// to force a parallel reduction with OpenMP
inline void
courantOnXY(real_t *cournox,
            real_t *cournoy,
            const int Hnx,
            const int Hnxyt,
            const int Hnvar, const int slices, const int Hstep, real_t c[Hstep][Hnxyt], real_t q[Hnvar][Hstep][Hnxyt],
	    real_t *tmpm1,
	    real_t *tmpm2
	    ) 
{
  int s = slices, i;
  // real_t maxValC = zero;
  real_t tmp1, tmp2;

#pragma omp critical
  tmp1 = *cournox;
#pragma omp critical
  tmp2 = *cournoy;

  for (i = 0; i < Hnx; i++) {
    tmp1 = MAX(tmp1, c[s][i] + DABS(q[IU][s][i]));
    tmp2 = MAX(tmp2, c[s][i] + DABS(q[IV][s][i]));
  }
#pragma omp critical
  *cournox = tmp1;
#pragma omp critical
  *cournoy = tmp2;
  { 
    int nops = (slices) * Hnx;
    FLOPS(2 * nops, 0 * nops, 2 * nops, 0 * nops);
  }
#undef IHVW
}

void compute_deltat_init_mem(const hydroparam_t H, hydrowork_t * Hw, hydrovarwork_t * Hvw)
{
  Hvw->q = (real_t (*)) DMalloc(H.nvar * H.nxyt * H.nxystep);
  Hw->e = (real_t (*))  DMalloc(         H.nxyt * H.nxystep);
  Hw->c = (real_t (*))  DMalloc(         H.nxyt * H.nxystep);
  Hw->tmpm1 = (real_t *) DMalloc(H.nxystep);
  Hw->tmpm2 = (real_t *) DMalloc(H.nxystep);

}

void compute_deltat_clean_mem(const hydroparam_t H, hydrowork_t * Hw, hydrovarwork_t * Hvw)
{
  DFree(&Hvw->q, H.nvar * H.nxyt * H.nxystep);
  DFree(&Hw->e, H.nxyt * H.nxystep);
  DFree(&Hw->c, H.nxyt * H.nxystep);
  DFree(&Hw->tmpm1, H.nxystep);
  DFree(&Hw->tmpm2, H.nxystep);
}


void
compute_deltat(real_t *dt, const hydroparam_t H, hydrowork_t * Hw, hydrovar_t * Hv, hydrovarwork_t * Hvw) {
  real_t cournox, cournoy;
  int j, jend, Hstep, Hmin, Hmax;
  real_t (*e)[H.nxyt];
  real_t (*c)[H.nxystep];
  real_t (*q)[H.nxystep][H.nxyt];
  WHERE("compute_deltat");
  int maxTh = omp_get_max_threads();
  real_t tcx[maxTh];
  real_t tcy[maxTh];


  memset(tcx, 0, sizeof(real_t) * maxTh);
  memset(tcy, 0, sizeof(real_t) * maxTh);

    //   compute time step on grid interior
  cournox = zero;
  cournoy = zero;
  c = (real_t (*)[H.nxystep]) Hw->c;
  e = (real_t (*)[H.nxystep]) Hw->e;
  q = (real_t (*)[H.nxystep][H.nxyt]) Hvw->q;

  Hstep = H.nxystep;
  Hmin = H.jmin + ExtraLayer;
  Hmax = H.jmax - ExtraLayer;

  #pragma omp parallel for private(j) shared(cournox, cournoy) schedule(static, 1)
  for (j = Hmin; j < Hmax; j++) {
    int myThread = omp_get_thread_num();
    int slice = myThread;      
    ComputeQEforRow(j, H.smallr, H.nx, H.nxt, H.nyt, H.nxyt, H.nvar, slice, Hstep, Hv->uold, q, e);
    equation_of_state(0, H.nx, H.nxyt, H.nvar, H.smallc, H.gamma, slice, Hstep, e, q, c);
    courantOnXY(&tcx[slice], &tcy[slice], H.nx, H.nxyt, H.nvar, slice, Hstep, c, q, Hw->tmpm1, Hw->tmpm2);
    // fprintf(stdout, "[%2d]\t%g %g %g %g\n", H.mype, cournox, cournoy, H.smallc, H.courant_factor);
  }

  for (j = 0; j < maxTh; j++) {
    cournox = MAX(cournox, tcx[j]);
    cournoy = MAX(cournoy, tcy[j]);
  }

  *dt = H.courant_factor * H.dx / MAX(cournox, MAX(cournoy, H.smallc));

  FLOPS(1, 1, 2, 0);
  // fprintf(stdout, "[%2d]\t%g %g %g %g %g %g\n", H.mype, cournox, cournoy, H.smallc, H.courant_factor, H.dx, *dt);
}                               // compute_deltat

//EOF
