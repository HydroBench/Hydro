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

void Dmemset(size_t nbr, real_t t[nbr], real_t motif)
{
    int i;
    for (i = 0; i < nbr; i++) {
	t[i] = motif;
    }
}

#define DABS(x) (real_t) fabs((x))
#define Fmax(a,b) (((a) > (b)) ? (a): (b))
#define Fabs(a) (((a) > 0) ? (a): -(a))

#define MYSQRT sqrt
#ifdef TARGETON
#define WITHTARGET
#endif

void riemann(int narray, const real_t Hsmallr, const real_t Hsmallc, const real_t Hgamma, const int Hniter_riemann, const int Hnvar, const int Hnxyt, const int slices, const int Hstep, real_t qleft[Hnvar][Hstep][Hnxyt], real_t qright[Hnvar][Hstep][Hnxyt],	//
	     real_t qgdnv[Hnvar][Hstep][Hnxyt],	//
	     int sgnm[Hstep][Hnxyt], hydrowork_t * Hw)
{
    int i, s, ii, iimx;
    int iter;
    real_t smallp_ = Square(Hsmallc) / Hgamma;
    real_t gamma6_ = (Hgamma + one) / (two * Hgamma);
    real_t smallpp_ = Hsmallr * smallp_;

    FLOPS(4, 2, 0, 0);
    // __declspec(align(256)) thevariable

    int *goon = Hw->goon;
    real_t *pstar = Hw->pstar;
    real_t *rl = Hw->rl;
    real_t *ul = Hw->ul;
    real_t *pl = Hw->pl;
    real_t *ur = Hw->ur;
    real_t *pr = Hw->pr;
    real_t *cl = Hw->cl;
    real_t *cr = Hw->cr;
    real_t *rr = Hw->rr;

    real_t smallp = smallp_;
    real_t gamma6 = gamma6_;
    real_t smallpp = smallpp_;

    long tmpsiz = slices * narray;

    // Pressure, density and velocity
#ifdef WITHTARGET
    // fprintf(stderr, "riemann IN\n");
#pragma omp target teams distribute parallel for default(none)		\
	private(s, i),							\
	shared(qgdnv, sgnm, qleft, qright, pstar, rl, ul, pl, rr, ur, cl, pr, cr, goon) \
	map(tofrom: qleft[0:Hnvar][0:Hstep][0:narray])			\
	map(tofrom: qright[0:Hnvar][0:Hstep][0:narray])			\
	map(tofrom: qgdnv[0:Hnvar][0:Hstep][0:narray])			\
	map(tofrom: sgnm[0:Hstep][0:narray])				\
	map(tofrom: pstar[0:tmpsiz])					\
	map(tofrom: rl[0:tmpsiz])					\
	map(tofrom: ul[0:tmpsiz])					\
	map(tofrom: pl[0:tmpsiz])					\
	map(tofrom: rr[0:tmpsiz])					\
	map(tofrom: ur[0:tmpsiz])					\
	map(tofrom: cl[0:tmpsiz])					\
	map(tofrom: pr[0:tmpsiz])					\
	map(tofrom: cr[0:tmpsiz])					\
	map(tofrom: goon[0:tmpsiz]) collapse(2)
#else
#pragma omp parallel for private(s, i),					\
	firstprivate(Hsmallr, Hgamma, slices, narray, smallp)		\
	default(none),							\
	shared(qgdnv, sgnm, qleft, qright, pstar, ul, rl, pl, rr, ur, pr, cl, cr, goon)
#endif
    for (s = 0; s < slices; s++) {
	// Precompute values for this slice
	for (i = 0; i < narray; i++) {
	    int ii = i + s * narray;
	    rl[ii] = fmax(qleft[ID][s][i], Hsmallr);
	    ul[ii] = qleft[IU][s][i];
	    pl[ii] = fmax(qleft[IP][s][i], (real_t) (rl[ii] * smallp));
	    rr[ii] = fmax(qright[ID][s][i], Hsmallr);
	    ur[ii] = qright[IU][s][i];
	    pr[ii] = fmax(qright[IP][s][i], (real_t) (rr[ii] * smallp));

	    // Lagrangian sound speed
	    cl[ii] = Hgamma * pl[ii] * rl[ii];
	    cr[ii] = Hgamma * pr[ii] * rr[ii];
	    // First guess

	    real_t wl_i = MYSQRT(cl[ii]);
	    real_t wr_i = MYSQRT(cr[ii]);
	    pstar[ii] =
		fmax(((wr_i * pl[ii] + wl_i * pr[ii]) +
		      wl_i * wr_i * (ul[ii] - ur[ii])) / (wl_i + wr_i), 0.0);
	    goon[ii] = 1;
	}
    }

#ifdef WITHTARGET
    // fprintf(stderr, "riemann between loop 1 and 2\n");
#pragma omp target teams distribute parallel for default(none)		\
	private(s, i, iter),							\
	shared(qgdnv, sgnm, qleft, qright, pstar, rl, ul, pl, rr, ur, cl, pr, cr, goon) \
	map(tofrom: qleft[0:Hnvar][0:Hstep][0:narray])			\
	map(tofrom: qright[0:Hnvar][0:Hstep][0:narray])			\
	map(tofrom: qgdnv[0:Hnvar][0:Hstep][0:narray])			\
	map(tofrom: sgnm[0:Hstep][0:narray])				\
	map(tofrom: pstar[0:tmpsiz])					\
	map(tofrom: rl[0:tmpsiz])					\
	map(tofrom: ul[0:tmpsiz])					\
	map(tofrom: pl[0:tmpsiz])					\
	map(tofrom: rr[0:tmpsiz])					\
	map(tofrom: ur[0:tmpsiz])					\
	map(tofrom: cl[0:tmpsiz])					\
	map(tofrom: pr[0:tmpsiz])					\
	map(tofrom: cr[0:tmpsiz])					\
	map(tofrom: goon[0:tmpsiz]) collapse(2)
#else
#pragma omp parallel for private(s, i, iter),				\
	firstprivate(Hsmallr, Hgamma, Hniter_riemann, slices, narray, smallp, smallpp, gamma6) \
	default(none),							\
	shared(qgdnv, sgnm, qleft, qright, pstar, ul, rl, pl, rr, ur, pr, cl, cr, goon)
#endif

    for (s = 0; s < slices; s++) {
	// solve the riemann problem on the interfaces of this slice
	for (i = 0; i < narray; i++) {
	    // Warning: this loop is not collapsable since it iterates until convergence on the boubaries off cells
	    for (iter = 0; iter < Hniter_riemann; iter++) {
		int ii = i + s * narray;
		if (goon[ii]) {
		    real_t pst = pstar[ii];
		    // Newton-Raphson iterations to find pstar at the required accuracy
		    real_t wwl =
			MYSQRT(cl[ii] *
			       (one + gamma6 * (pst - pl[ii]) / pl[ii]));
		    real_t wwr =
			MYSQRT(cr[ii] *
			       (one + gamma6 * (pst - pr[ii]) / pr[ii]));
		    real_t swwl = Square(wwl);
		    real_t ql = two * wwl * swwl / (swwl + cl[ii]);
		    real_t qr =
			two * wwr * Square(wwr) / (Square(wwr) + cr[ii]);
		    real_t usl = ul[ii] - (pst - pl[ii]) / wwl;
		    real_t usr = ur[ii] + (pst - pr[ii]) / wwr;
		    real_t tmp = (qr * ql / (qr + ql) * (usl - usr));
		    real_t delp_i = Fmax(tmp, (-pst));
		    // pstar[ii] = pstar[ii] + delp_i;
		    pst += delp_i;
		    // Convergence indicator
		    real_t tmp2 = delp_i / (pst + smallpp);
		    real_t uo_i = Fabs(tmp2);
		    goon[ii] = uo_i > PRECISION;
		    // FLOPS(29, 10, 2, 0);
		    pstar[ii] = pst;
		}
	    } // iter_riemann
	}			
    }

#ifdef WITHTARGET
    // fprintf(stderr, "riemann between loop 2 and 3\n");
#pragma omp target teams distribute parallel for default(none)		\
	private(s, i),							\
	shared(qgdnv, sgnm, qleft, qright, pstar)			\
	shared(rl, ul, pl, rr, ur, cl, pr, cr, goon)			\
	map(tofrom: qleft[0:Hnvar][0:Hstep][0:narray])			\
	map(tofrom: qright[0:Hnvar][0:Hstep][0:narray])			\
	map(tofrom: qgdnv[0:Hnvar][0:Hstep][0:narray])			\
	map(tofrom: sgnm[0:Hstep][0:narray])				\
	map(tofrom: pstar[0:tmpsiz])					\
	map(tofrom: rl[0:tmpsiz])					\
	map(tofrom: ul[0:tmpsiz])					\
	map(tofrom: pl[0:tmpsiz])					\
	map(tofrom: rr[0:tmpsiz])					\
	map(tofrom: ur[0:tmpsiz])					\
	map(tofrom: cl[0:tmpsiz])					\
	map(tofrom: pr[0:tmpsiz])					\
	map(tofrom: cr[0:tmpsiz])					\
	map(tofrom: goon[0:tmpsiz]) collapse(2)	//

#else
#pragma omp parallel for private(s, i),					\
	firstprivate(Hsmallr, Hsmallc, Hgamma, slices, narray, smallp, gamma6) \
	default(none),							\
	shared(qgdnv, sgnm, qleft, qright, pstar, ul, rl, pl, rr, ur, pr, cl, cr, goon)
#endif
    for (s = 0; s < slices; s++) {
	for (i = 0; i < narray; i++) {
	    int ii = i + s * narray;
	    real_t wl_i = MYSQRT(cl[ii]);
	    real_t wr_i = MYSQRT(cr[ii]);

	    wr_i =
		MYSQRT(cr[ii] * (one + gamma6 * (pstar[ii] - pr[ii]) / pr[ii]));
	    wl_i =
		MYSQRT(cl[ii] * (one + gamma6 * (pstar[ii] - pl[ii]) / pl[ii]));

	    real_t ustar_i =
		half * (ul[ii] + (pl[ii] - pstar[ii]) / wl_i + ur[ii] -
			(pr[ii] - pstar[ii]) / wr_i);

	    int left = ustar_i > 0;

	    real_t ro_i, uo_i, po_i, wo_i;

	    if (left) {
		sgnm[s][ii] = 1;
		ro_i = rl[ii];
		uo_i = ul[ii];
		po_i = pl[ii];
		wo_i = wl_i;
	    } else {
		sgnm[s][ii] = -1;
		ro_i = rr[ii];
		uo_i = ur[ii];
		po_i = pr[ii];
		wo_i = wr_i;
	    }

	    real_t co_i = MYSQRT(fabs(Hgamma * po_i / ro_i));
	    co_i = fmax(Hsmallc, co_i);

	    real_t rstar_i =
		ro_i / (one + ro_i * (po_i - pstar[ii]) / Square(wo_i));
	    rstar_i = fmax(rstar_i, Hsmallr);

	    real_t cstar_i = MYSQRT(fabs(Hgamma * pstar[ii] / rstar_i));
	    cstar_i = fmax(Hsmallc, cstar_i);

	    real_t spout_i = co_i - sgnm[s][i] * uo_i;
	    real_t spin_i = cstar_i - sgnm[s][i] * ustar_i;
	    real_t ushock_i = wo_i / ro_i - sgnm[s][i] * uo_i;

	    if (pstar[ii] >= po_i) {
		spin_i = ushock_i;
		spout_i = ushock_i;
	    }

	    real_t scr_i = fmax((real_t) (spout_i - spin_i),
				(real_t) (Hsmallc + fabs(spout_i + spin_i)));

	    real_t frac_i = (one + (spout_i + spin_i) / scr_i) * half;
	    frac_i = fmax(zero, (real_t) (fmin(one, frac_i)));

	    int addSpout = spout_i < zero;
	    int addSpin = spin_i > zero;
	    // real_t originalQgdnv = !addSpout & !addSpin;
	    real_t qgdnv_ID, qgdnv_IU, qgdnv_IP;

	    if (addSpout) {
		qgdnv_ID = ro_i;
		qgdnv_IU = uo_i;
		qgdnv_IP = po_i;
	    } else if (addSpin) {
		qgdnv_ID = rstar_i;
		qgdnv_IU = ustar_i;
		qgdnv_IP = pstar[ii];
	    } else {
		qgdnv_ID = (frac_i * rstar_i + (one - frac_i) * ro_i);
		qgdnv_IU = (frac_i * ustar_i + (one - frac_i) * uo_i);
		qgdnv_IP = (frac_i * pstar[ii] + (one - frac_i) * po_i);
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
    int nops = slices * narray;
    FLOPS(57 * nops, 17 * nops, 14 * nops, 0 * nops);
#ifdef WITHTARGET
    // fprintf(stderr, "riemann OUT\n");
#endif

    // other passive variables
    if (Hnvar > IP) {
	int invar;
	for (invar = IP + 1; invar < Hnvar; invar++) {
	    for (s = 0; s < slices; s++) {
		for (i = 0; i < narray; i++) {
		    int left = (sgnm[s][i] == 1);
		    qgdnv[invar][s][i] =
			qleft[invar][s][i] * left + qright[invar][s][i] * !left;
		}
	    }
	}
    }
}				// riemann_vec

//EOF
