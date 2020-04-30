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
Dmemset(size_t nbr, real_t t[nbr], real_t motif) {
  int i;
  for (i = 0; i < nbr; i++) {
    t[i] = motif;
  }
}


#define DABS(x) (real_t) fabs((x))
#ifdef HMPP
#define MAX(x,y) fmax(x,y)
#endif

#define MYSQRT sqrt


#define printvf64(v, gvl) do {\
	real_t buf[gvl];\
	int i;\
	printf("%s ", #v);\
	__builtin_epi_vstore_1xf64(buf, v, gvl);\
	for (i = 0 ; i < gvl; i ++) {\
		printf("(%d)%e ", i, buf[i]);\
	}\
	printf("\n");\
	}while(0)
#define printvi32(v, gvl) do {\
	 int buf[gvl];\
	int i;\
	printf("%s ", #v);\
	__builtin_epi_vstore_2xi32(buf, v, gvl);\
	for (i = 0 ; i < gvl; i ++) {\
		printf("(%d)%d ", i, buf[i]);\
	}\
	printf("\n");\
	}while(0)
#define printvmaski2(v, gvl) do {\
	unsigned int buf[512];\
	int i;\
	printf("%s ", #v);\
	__builtin_epi_vstore_2xi1(buf, v);\
	for (i = 0 ; i < gvl; i ++) {\
		printf("(%d)%d ", i, buf[i]?1:0);	\
	}\
	printf("\n");\
	}while(0)
#define printvmaski1(v, gvl) do {\
	unsigned long buf[512];\
	int i;\
	printf("%s ", #v);\
	__builtin_epi_vstore_1xi1(buf, v);\
	for (i = 0 ; i < gvl; i ++) {\
		printf("(%d)%d ", i, buf[i]?1:0);	\
	}\
	printf("\n");\
	}while(0)

void solve_all_masking_rvv(const int s, // for debugging
			   const int narray,
			   real_t *restrict pstar,
			   const real_t * const restrict ul,
			   const real_t * const restrict pl,
			   const real_t * const restrict ur,
			   const real_t * const restrict pr,
			   const real_t * const restrict cl,
			   const real_t * const restrict cr,
			  int * restrict goon,
			   const real_t gamma6,
			   const real_t smallpp) {
	//fprintf(stderr, "%s start\n", __PRETTY_FUNCTION__);
	int i = 0, j;
	
	while (i < narray) {
		unsigned long gvl = __builtin_epi_vsetvl(narray - i, __epi_e64, __epi_m1);
		unsigned long long goon64[gvl];
		//fprintf(stderr, "%s: gvl = %lu for %d / %d (= %d)\n", __PRETTY_FUNCTION__, gvl, i, narray, narray - i);
		__epi_1xf64 vgamma6 = __builtin_epi_vfmv_v_f_1xf64(gamma6, gvl);
		__epi_1xf64 vsmallpp = __builtin_epi_vfmv_v_f_1xf64(smallpp, gvl);
		for (j = 0 ; j < gvl ; j++) goon64[j] = goon[i + j]; // i've yet to find a better way
		__epi_1xi64 vgoon = __builtin_epi_vload_1xi64(goon64, gvl);
		__epi_1xi1 mask = __builtin_epi_vmseq_1xi64(vgoon, __builtin_epi_vmv_v_x_1xi64(1, gvl), gvl);
		/* currently crashes in vehave */
/* 		if (__builtin_epi_vfirst_1xi1(mask, gvl) < 0) */
/* 			return; // early abort when nothing to do. */
		/* mask = __builtin_epi_vmxnor_1xi1(mask, mask, gvl); */
		/* assume real_t == double */
		const __epi_1xf64 vone = __builtin_epi_vfmv_v_f_1xf64(1., gvl);
		const __epi_1xf64 vtwo = __builtin_epi_vfmv_v_f_1xf64(2., gvl);
		const __epi_1xf64 vzero = __builtin_epi_vfmv_v_f_1xf64(0., gvl);
#if 0 // EPI compiler doesn't support masked load/store yet, no big deal, do it by hand; loads are good - data exist
		__epi_1xf64 vpst = __builtin_epi_vload_1xf64_mask(vzero, pstar + i, mask, gvl);
		__epi_1xf64 vul = __builtin_epi_vload_1xf64_mask(vzero, ul + i, mask, gvl);
		__epi_1xf64 vpl = __builtin_epi_vload_1xf64_mask(vzero, pl + i, mask, gvl);
		__epi_1xf64 vur = __builtin_epi_vload_1xf64_mask(vzero, ur + i, mask, gvl);
		__epi_1xf64 vpr = __builtin_epi_vload_1xf64_mask(vzero, pr + i, mask, gvl);
		__epi_1xf64 vcl = __builtin_epi_vload_1xf64_mask(vzero, cl + i, mask, gvl);
		__epi_1xf64 vcr = __builtin_epi_vload_1xf64_mask(vzero, cr + i, mask, gvl);
#else
		__epi_1xf64 vpst = __builtin_epi_vload_1xf64(pstar + i, gvl);
		__epi_1xf64 vul = __builtin_epi_vload_1xf64(ul + i, gvl);
		__epi_1xf64 vpl = __builtin_epi_vload_1xf64(pl + i, gvl);
		__epi_1xf64 vur = __builtin_epi_vload_1xf64(ur + i, gvl);
		__epi_1xf64 vpr = __builtin_epi_vload_1xf64(pr + i, gvl);
		__epi_1xf64 vcl = __builtin_epi_vload_1xf64(cl + i, gvl);
		__epi_1xf64 vcr = __builtin_epi_vload_1xf64(cr + i, gvl);
		__epi_1xf64 orig_vpst = vpst; // to merge before store
#endif
		// Newton-Raphson iterations to find pstar at the required accuracy
		__epi_1xf64 vwwl = __builtin_epi_vfsqrt_1xf64_mask(vzero,
								   __builtin_epi_vfmul_1xf64_mask(vzero,
												  vcl,
												  __builtin_epi_vfadd_1xf64_mask(vzero,
																 vone,
																 __builtin_epi_vfmul_1xf64_mask(vzero,
																				vgamma6,
																				__builtin_epi_vfdiv_1xf64_mask(vzero,
																							       __builtin_epi_vfsub_1xf64_mask(vzero, vpst, vpl, mask, gvl),
																							       vpl, mask, gvl), mask, gvl), mask, gvl), mask, gvl), mask, gvl);
		__epi_1xf64 vwwr = __builtin_epi_vfsqrt_1xf64_mask(vzero,
								   __builtin_epi_vfmul_1xf64_mask(vzero,
												  vcr,
												  __builtin_epi_vfadd_1xf64_mask(vzero,
																 vone,
																 __builtin_epi_vfmul_1xf64_mask(vzero,
																				vgamma6,
																				__builtin_epi_vfdiv_1xf64_mask(vzero,
																							       __builtin_epi_vfsub_1xf64_mask(vzero, vpst, vpr, mask, gvl),
																							       vpr, mask, gvl), mask, gvl), mask, gvl), mask, gvl), mask, gvl);
		__epi_1xf64 vswwl = __builtin_epi_vfmul_1xf64_mask(vzero, vwwl, vwwl, mask, gvl);
		__epi_1xf64 vswwr = __builtin_epi_vfmul_1xf64_mask(vzero, vwwr, vwwr, mask, gvl);
		__epi_1xf64 vql = __builtin_epi_vfdiv_1xf64_mask(vzero,
								 __builtin_epi_vfmul_1xf64_mask(vzero, vtwo, __builtin_epi_vfmul_1xf64_mask(vzero, vwwl, vswwl, mask, gvl), mask, gvl),
								 __builtin_epi_vfadd_1xf64_mask(vzero, vswwl, vcl, mask, gvl), mask, gvl);
		__epi_1xf64 vqr = __builtin_epi_vfdiv_1xf64_mask(vzero,
								 __builtin_epi_vfmul_1xf64_mask(vzero, vtwo, __builtin_epi_vfmul_1xf64_mask(vzero, vwwr, vswwr, mask, gvl), mask, gvl),
								 __builtin_epi_vfadd_1xf64_mask(vzero, vswwr, vcr, mask, gvl), mask, gvl);
		__epi_1xf64 vusl = __builtin_epi_vfsub_1xf64_mask(vzero,
								  vul,
								  __builtin_epi_vfdiv_1xf64_mask(vzero,
												 __builtin_epi_vfsub_1xf64_mask(vzero, vpst, vpl, mask, gvl),
												 vwwl, mask, gvl), mask, gvl);
/* 		if (s == 32) printvf64(vusl, gvl); */
		__epi_1xf64 vusr = __builtin_epi_vfadd_1xf64_mask(vzero,
								  vur,
								  __builtin_epi_vfdiv_1xf64_mask(vzero,
												 __builtin_epi_vfsub_1xf64_mask(vzero, vpst, vpr, mask, gvl),
												 vwwr, mask, gvl), mask, gvl);
/* 		if (s == 32) printvf64(vusr, gvl); */
		__epi_1xf64 vtmpmiddle = __builtin_epi_vfdiv_1xf64_mask(vzero,
									__builtin_epi_vfmul_1xf64_mask(vzero,
												       vqr,
												       vql, mask, gvl),
									__builtin_epi_vfadd_1xf64_mask(vzero, vqr, vql, mask, gvl), mask, gvl);
		__epi_1xf64 vtmp = __builtin_epi_vfmul_1xf64_mask(vzero,
								  vtmpmiddle,
								  __builtin_epi_vfsub_1xf64_mask(vzero, vusl, vusr, mask, gvl), mask, gvl);
/* 		if (s == 32) printvf64(vtmp, gvl); */
		__epi_1xf64 vdelp_i = __builtin_epi_vfmax_1xf64_mask(vzero, vtmp, __builtin_epi_vfsgnjn_1xf64_mask(vzero, vpst, vpst, mask, gvl), mask, gvl); // fsgnjn for fneg
		vpst = __builtin_epi_vfadd_1xf64_mask(vzero, vpst, vdelp_i, mask, gvl);
/* 		if (s == 32) printvf64(vpst, gvl); */
		// Convergence indicator
		__epi_1xf64 vtmp2 = __builtin_epi_vfdiv_1xf64_mask(vzero,
								   vdelp_i,
								   __builtin_epi_vfadd_1xf64_mask(vzero, vpst, vsmallpp, mask, gvl), mask, gvl);
		__epi_1xf64 vuo_i = __builtin_epi_vfsgnjx_1xf64_mask(vzero, vtmp2, vtmp2, mask, gvl); // fsgnjx for fabs
		__epi_1xi1 converged = __builtin_epi_vmfle_1xf64(vuo_i, __builtin_epi_vfmv_v_f_1xf64(PRECISION, gvl), gvl); // no masked version ?
		converged = __builtin_epi_vmand_1xi1(mask, converged, gvl);

		/* again masjked version is not yet implemented
		   ... though the performance limitations is probably from the big copy ... */
		vgoon = __builtin_epi_vmerge_1xi64(vgoon, __builtin_epi_vmv_v_x_1xi64(0, gvl), converged, gvl);
		__builtin_epi_vstore_1xi64(goon64, vgoon, gvl);
		for (j = 0 ; j < gvl ; j++) goon[i + j] = goon64[j];
	
#if 0
		__builtin_epi_vstore_1xf64_mask(pstar + i, vpst, mask, gvl);
#else
		vpst = __builtin_epi_vfmerge_1xf64(orig_vpst, vpst, mask, gvl);
		__builtin_epi_vstore_1xf64(pstar + i, vpst, gvl);
#endif

		i += gvl;
	}
	//fprintf(stderr, "%s stop\n", __PRETTY_FUNCTION__);
}

void
riemann(int narray, const real_t Hsmallr, 
	const real_t Hsmallc, const real_t Hgamma, 
	const int Hniter_riemann, const int Hnvar, 
	const int Hnxyt, const int slices, 
	const int Hstep, 
	real_t qleft[Hnvar][Hstep][Hnxyt], 
	real_t qright[Hnvar][Hstep][Hnxyt],      //
	real_t qgdnv[Hnvar][Hstep][Hnxyt],      //
	int sgnm[Hstep][Hnxyt], 
	hydrowork_t * Hw) 
{
  int i, s, ii, iimx;
  real_t smallp_ = Square(Hsmallc) / Hgamma;
  real_t gamma6_ = (Hgamma + one) / (two * Hgamma);
  real_t smallpp_ = Hsmallr * smallp_;

  FLOPS(4, 2, 0, 0);
  // __declspec(align(256)) thevariable

  int *Fgoon = Hw->goon;
  real_t *Fpstar = Hw->pstar;
  real_t *Frl = Hw->rl;
  real_t *Ful = Hw->ul;
  real_t *Fpl = Hw->pl;
  real_t *Fur = Hw->ur;
  real_t *Fpr = Hw->pr;
  real_t *Fcl = Hw->cl;
  real_t *Fcr = Hw->cr;
  real_t *Frr = Hw->rr;

  real_t smallp = smallp_;
  real_t gamma6 = gamma6_;
  real_t smallpp = smallpp_;

  // fprintf(stderr, "%d\n", __ICC );
#pragma message "active pragma simd "
#define SIMDNEEDED 1
#if __ICC < 1300
#define SIMD ivdep
#else
#define SIMD simd
#endif
  // #define SIMD novector

  // Pressure, density and velocity
#pragma omp parallel for private(s, i), shared(qgdnv, sgnm) reduction(+:flopsAri), reduction(+:flopsSqr), reduction(+:flopsMin), reduction(+:flopsTra)
  for (s = 0; s < slices; s++) {
    int ii, iimx;
    int *goon;
    real_t *pstar, *rl, *ul, *pl, *rr, *ur, *pr, *cl, *cr;
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
      pl[i] = fmax(qleft[IP][s][i], (real_t) (rl[i] * smallp));
      rr[i] = fmax(qright[ID][s][i], Hsmallr);
      ur[i] = qright[IU][s][i];
      pr[i] = fmax(qright[IP][s][i], (real_t) (rr[i] * smallp));

      // Lagrangian sound speed
      cl[i] = Hgamma * pl[i] * rl[i];
      cr[i] = Hgamma * pr[i] * rr[i];
      // First guess

      real_t wl_i = MYSQRT(cl[i]);
      real_t wr_i = MYSQRT(cr[i]);
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
#if 0
      for (i = 0; i < narray; i++) {
	if (goon[i]) {
	  real_t pst = pstar[i];
	  // Newton-Raphson iterations to find pstar at the required accuracy
	  real_t wwl = MYSQRT(cl[i] * (one + gamma6 * (pst - pl[i]) / pl[i]));
	  real_t wwr = MYSQRT(cr[i] * (one + gamma6 * (pst - pr[i]) / pr[i]));
	  real_t swwl = Square(wwl);
	  real_t ql = two * wwl * swwl / (swwl + cl[i]);
	  real_t qr = two * wwr * Square(wwr) / (Square(wwr) + cr[i]);
	  real_t usl = ul[i] - (pst - pl[i]) / wwl;
	  real_t usr = ur[i] + (pst - pr[i]) / wwr;
	  real_t tmp = (qr * ql / (qr + ql) * (usl - usr));
	  real_t delp_i = Fmax(tmp, (-pst));
	  // pstar[i] = pstar[i] + delp_i;
	  pst += delp_i;
	  // Convergence indicator
	  real_t tmp2 = delp_i / (pst + smallpp);
	  real_t uo_i = Fabs(tmp2);
	  goon[i] = uo_i > PRECISION;
	  // FLOPS(29, 10, 2, 0);
	  pstar[i] = pst;
	}
      }
#else
      solve_all_masking_rvv(s,
			    narray,
			    pstar,
			    ul,
			    pl,
			    ur,
			    pr,
			    cl,
			    cr,
			    goon,
			    gamma6,
			    smallpp);
#endif
    }                           // iter_riemann

#ifdef SIMDNEEDED
#pragma SIMD
#endif
    for (i = 0; i < narray; i++) {
      real_t wl_i = MYSQRT(cl[i]);
      real_t wr_i = MYSQRT(cr[i]);

      wr_i = MYSQRT(cr[i] * (one + gamma6 * (pstar[i] - pr[i]) / pr[i]));
      wl_i = MYSQRT(cl[i] * (one + gamma6 * (pstar[i] - pl[i]) / pl[i]));

      real_t ustar_i = half * (ul[i] + (pl[i] - pstar[i]) / wl_i + ur[i] - (pr[i] - pstar[i]) / wr_i);

      int left = ustar_i > 0;

      real_t ro_i, uo_i, po_i, wo_i;

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

      real_t co_i = MYSQRT(fabs(Hgamma * po_i / ro_i));
      co_i = fmax(Hsmallc, co_i);

      real_t rstar_i = ro_i / (one + ro_i * (po_i - pstar[i]) / Square(wo_i));
      rstar_i = fmax(rstar_i, Hsmallr);

      real_t cstar_i = MYSQRT(fabs(Hgamma * pstar[i] / rstar_i));
      cstar_i = fmax(Hsmallc, cstar_i);

      real_t spout_i = co_i - sgnm[s][i] * uo_i;
      real_t spin_i = cstar_i - sgnm[s][i] * ustar_i;
      real_t ushock_i = wo_i / ro_i - sgnm[s][i] * uo_i;

      if (pstar[i] >= po_i) {
	spin_i = ushock_i;
	spout_i = ushock_i;
      }

      real_t scr_i = fmax((real_t) (spout_i - spin_i), (real_t) (Hsmallc + fabs(spout_i + spin_i)));

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
