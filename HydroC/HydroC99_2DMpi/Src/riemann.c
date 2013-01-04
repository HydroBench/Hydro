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
#include "riemann.h"

#ifdef HMPP
#undef HMPP
#include "constoprim.c"
#include "equation_of_state.c"
#include "slope.c"
#include "trace.c"
#include "qleftright.c"
#include "cmpflx.c"
#include "conservar.c"
#define HMPP
#endif

#define PRECISION 1e-6

void
Dmemset(size_t nbr, double t[nbr], double motif) {
  int i;
  for (i = 0; i < nbr; i++) {
    t[i] = motif;
  }
}


#define DABS(x) (double) fabs((x))
#ifdef HMPP
#define MAX(x,y) fmax(x,y)
#endif

void
riemann(int narray, const double Hsmallr, 
	const double Hsmallc, const double Hgamma, 
	const int Hniter_riemann, const int Hnvar, 
	const int Hnxyt, const int slices, 
	const int Hstep, 
	double qleft[Hnvar][Hstep][Hnxyt], 
	double qright[Hnvar][Hstep][Hnxyt],      //
	double qgdnv[Hnvar][Hstep][Hnxyt],      //
	int sgnm[Hstep][Hnxyt], 
	hydrowork_t * Hw) 
{
  int i, s, ii, iimx;
  double smallp_ = Square(Hsmallc) / Hgamma;
  double gamma6_ = (Hgamma + one) / (two * Hgamma);
  double smallpp_ = Hsmallr * smallp_;

  FLOPS(4, 2, 0, 0);
  // __declspec(align(256)) thevariable

  int *Fgoon = Hw->goon;
  double *Fpstar = Hw->pstar;
  double *Frl = Hw->rl;
  double *Ful = Hw->ul;
  double *Fpl = Hw->pl;
  double *Fur = Hw->ur;
  double *Fpr = Hw->pr;
  double *Fcl = Hw->cl;
  double *Fcr = Hw->cr;
  double *Frr = Hw->rr;

  double smallp = smallp_;
  double gamma6 = gamma6_;
  double smallpp = smallpp_;

  // fprintf(stderr, "%d\n", __ICC );
#warning "active pragma simd " __ICC
#define SIMDNEEDED 1
#if __ICC < 1300
#define SIMD ivdep
#else
#define SIMD simd
#endif
  // #define SIMD novector

  // Pressure, density and velocity
#pragma omp parallel for  schedule(auto) private(s, i), shared(qgdnv, sgnm) reduction(+:flopsAri), reduction(+:flopsSqr), reduction(+:flopsMin), reduction(+:flopsTra)
  for (s = 0; s < slices; s++) {
    int ii, iimx;
    int *goon;
    double *pstar, *rl, *ul, *pl, *rr, *ur, *pr, *cl, *cr;
    int iter;
    pstar = &Fpstar[s * narray];
    rl = &Frl[s * narray];
    ul = &Ful[s * narray];
    pl = &Fpl[s * narray];
    rr = &Frr[s * narray];
    ur = &Fur[s * narray];
    pr = &Fpr[s * narray];
    cl = &Fcl[s * narray];
    cr = &Fcr[s * narray];
    goon = &Fgoon[s * narray];

    // Precompute values for this slice

#ifdef SIMDNEEDED
#if __ICC < 1300
#pragma ivdep
#else
#pragma SIMD
#endif
#endif
    for (i = 0; i < narray; i++) {
      rl[i] = fmax(qleft[ID][s][i], Hsmallr);
      ul[i] = qleft[IU][s][i];
      pl[i] = fmax(qleft[IP][s][i], (double) (rl[i] * smallp));
      rr[i] = fmax(qright[ID][s][i], Hsmallr);
      ur[i] = qright[IU][s][i];
      pr[i] = fmax(qright[IP][s][i], (double) (rr[i] * smallp));

      // Lagrangian sound speed
      cl[i] = Hgamma * pl[i] * rl[i];
      cr[i] = Hgamma * pr[i] * rr[i];
      // First guess

      double wl_i = sqrt(cl[i]);
      double wr_i = sqrt(cr[i]);
      pstar[i] = fmax(((wr_i * pl[i] + wl_i * pr[i]) + wl_i * wr_i * (ul[i] - ur[i])) / (wl_i + wr_i), 0.0);
      goon[i] = 1;
    }

#define Fmax(a,b) (((a) > (b)) ? (a): (b))
#define Fabs(a) (((a) > 0) ? (a): -(a))

    // solve the riemann problem on the interfaces of this slice
    for (iter = 0; iter < Hniter_riemann; iter++) {
#ifdef SIMDNEEDED
#if __ICC < 1300
#pragma simd
#else
#pragma SIMD
#endif
#endif
      for (i = 0; i < narray; i++) {
	if (goon[i]) {
	  double pst = pstar[i];
	  // Newton-Raphson iterations to find pstar at the required accuracy
	  double wwl = sqrt(cl[i] * (one + gamma6 * (pst - pl[i]) / pl[i]));
	  double wwr = sqrt(cr[i] * (one + gamma6 * (pst - pr[i]) / pr[i]));
	  double ql = two * wwl * Square(wwl) / (Square(wwl) + cl[i]);
	  double qr = two * wwr * Square(wwr) / (Square(wwr) + cr[i]);
	  double usl = ul[i] - (pst - pl[i]) / wwl;
	  double usr = ur[i] + (pst - pr[i]) / wwr;
	  double tmp = (qr * ql / (qr + ql) * (usl - usr));
	  double delp_i = Fmax(tmp, (-pst));
	  // pstar[i] = pstar[i] + delp_i;
	  pst += delp_i;
	  // Convergence indicator
	  double tmp2 = delp_i / (pst + smallpp);
	  double uo_i = Fabs(tmp2);
	  goon[i] = uo_i > PRECISION;
	  // FLOPS(29, 10, 2, 0);
	  pstar[i] = pst;
	}
      }
    }                           // iter_riemann

#ifdef SIMDNEEDED
#pragma SIMD
#endif
    for (i = 0; i < narray; i++) {
      double wl_i = sqrt(cl[i]);
      double wr_i = sqrt(cr[i]);

      wr_i = sqrt(cr[i] * (one + gamma6 * (pstar[i] - pr[i]) / pr[i]));
      wl_i = sqrt(cl[i] * (one + gamma6 * (pstar[i] - pl[i]) / pl[i]));

      double ustar_i = half * (ul[i] + (pl[i] - pstar[i]) / wl_i + ur[i] - (pr[i] - pstar[i]) / wr_i);

      int left = ustar_i > 0;

      double ro_i, uo_i, po_i, wo_i;

      if (left) {
	sgnm[s][i] = 1;
	ro_i = rl[i];
	uo_i = ul[i];
	po_i = pl[i];
	wo_i = wl_i;
      } else {
	sgnm[s][i] = -1;
	ro_i = rr[i];
	uo_i = ur[i];
	po_i = pr[i];
	wo_i = wr_i;
      }

      double co_i = sqrt(fabs(Hgamma * po_i / ro_i));
      co_i = fmax(Hsmallc, co_i);

      double rstar_i = ro_i / (one + ro_i * (po_i - pstar[i]) / Square(wo_i));
      rstar_i = fmax(rstar_i, Hsmallr);

      double cstar_i = sqrt(fabs(Hgamma * pstar[i] / rstar_i));
      cstar_i = fmax(Hsmallc, cstar_i);

      double spout_i = co_i - sgnm[s][i] * uo_i;
      double spin_i = cstar_i - sgnm[s][i] * ustar_i;
      double ushock_i = wo_i / ro_i - sgnm[s][i] * uo_i;

      if (pstar[i] >= po_i) {
	spin_i = ushock_i;
	spout_i = ushock_i;
      }

      double scr_i = fmax((double) (spout_i - spin_i), (double) (Hsmallc + fabs(spout_i + spin_i)));

      double frac_i = (one + (spout_i + spin_i) / scr_i) * half;
      frac_i = fmax(zero, (double) (fmin(one, frac_i)));

      int addSpout = spout_i < zero;
      int addSpin = spin_i > zero;
      // double originalQgdnv = !addSpout & !addSpin;
      double qgdnv_ID, qgdnv_IU, qgdnv_IP;

      if (addSpout) {
	qgdnv_ID = ro_i;
	qgdnv_IU = uo_i;
	qgdnv_IP = po_i;
      } else if (addSpin) {
	qgdnv_ID = rstar_i;
	qgdnv_IU = ustar_i;
	qgdnv_IP = pstar[i];
      } else {
	qgdnv_ID = (frac_i * rstar_i + (one - frac_i) * ro_i);
	qgdnv_IU = (frac_i * ustar_i + (one - frac_i) * uo_i);
	qgdnv_IP = (frac_i * pstar[i] + (one - frac_i) * po_i);
      }

      qgdnv[ID][s][i] = qgdnv_ID;
      qgdnv[IU][s][i] = qgdnv_IU;
      qgdnv[IP][s][i] = qgdnv_IP;

      // transverse velocity
      if (left) {
	qgdnv[IV][s][i] = qleft[IV][s][i];
      } else {
	qgdnv[IV][s][i] = qright[IV][s][i];
      }
    }
  }
  {
    int nops = slices * narray;
    FLOPS(57 * nops, 17 * nops, 14 * nops, 0 * nops);
  }

  // other passive variables
  if (Hnvar > IP) {
    int invar;
    for (invar = IP + 1; invar < Hnvar; invar++) {
      for (s = 0; s < slices; s++) {
#ifdef SIMDNEEDED
#pragma SIMD
#endif
	for (i = 0; i < narray; i++) {
	  int left = (sgnm[s][i] == 1);
	  qgdnv[invar][s][i] = qleft[invar][s][i] * left + qright[invar][s][i] * !left;
	}
      }
    }
  }
}                               // riemann_vec

//EOF
