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

#if defined(__aarch64__)

#include <arm_sve.h>

//#define SOLVE_ALL_FUNC

#ifdef SOLVE_ALL_FUNC

#define printvbool(vb, so)			\
  {						\
    unsigned long vc = svcntb();		\
    unsigned char buf[vc];			\
    int i;					\
    printf("%s: ", #vb);			\
    for (i = 0 ; i < vc ; i++) buf[i] = 0;	\
    svst1_u8(vb, buf, svdup_n_u8(1));		\
    for (i = 0 ; i < vc ; i+=so) {		\
      printf("0x%02x ", buf[i]);		\
    }						\
    printf("\n");				\
  }

void solve_all_masking_sve(const int s, // for debugging
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
  const unsigned long vc = svcntb();
  int i = 0, j;
  while (i < narray) {
#if 0
    long long goon64[vc/sizeof(real_t)];
    svbool_t tailmask = svcmplt_s64(svptrue_b64(), svindex_s64(0, 1), svdup_n_s64(narray - i));
    for (j = 0 ; j < vc/sizeof(real_t) ; j++) goon64[j] = goon[i + j];
    svbool_t mask = svcmpeq_s64(tailmask, svld1_s64(tailmask, goon64), svdup_n_s64(1));
#else
    svbool_t tailmask32 = svcmplt_s32(svptrue_b32(), svindex_s32(0, 1), svdup_n_s32(narray - i));
    svint32_t vgoon32 = svld1_s32(tailmask32, goon + i);
    svbool_t mask32 = svcmpeq_s32(tailmask32, vgoon32, svdup_n_s32(1));
    if (!svptest_any(mask32, mask32)) {
      i += vc/sizeof(real_t);
      continue;
    }
    svbool_t mask = svunpklo_b(mask32); // unpack to convert the mask to the wider type
#endif
    /* assume real_t == double */
    svfloat64_t vgamma6 = svdup_n_f64(gamma6);
    svfloat64_t vsmallpp = svdup_n_f64(smallpp);
    svfloat64_t vone = svdup_n_f64(1.);
    svfloat64_t vtwo = svdup_n_f64(2.);
    svfloat64_t vpst = svld1_f64(mask, pstar + i);
    svfloat64_t vul = svld1_f64(mask, ul + i);
    svfloat64_t vpl = svld1_f64(mask, pl + i);
    svfloat64_t vur = svld1_f64(mask, ur + i);
    svfloat64_t vpr = svld1_f64(mask, pr + i);
    svfloat64_t vcl = svld1_f64(mask, cl + i);
    svfloat64_t vcr = svld1_f64(mask, cr + i);
    // Newton-Raphson iterations to find pstar at the required accuracy
    svfloat64_t vwwl = svsqrt_f64_z(mask,
				    svmul_f64_z(mask,
						vcl,
						svadd_f64_z(mask,
							    vone,
							    svmul_f64_z(mask,
									vgamma6,
									svdiv_f64_z(mask,
										    svsub_f64_z(mask, vpst, vpl),
										    vpl)))));
    svfloat64_t vwwr = svsqrt_f64_z(mask,
				    svmul_f64_z(mask,
						vcr,
						svadd_f64_z(mask,
							    vone,
							    svmul_f64_z(mask,
									vgamma6,
									svdiv_f64_z(mask,
										    svsub_f64_z(mask, vpst, vpr),
										    vpr)))));
    svfloat64_t vswwl = svmul_f64_z(mask, vwwl, vwwl);
    svfloat64_t vswwr = svmul_f64_z(mask, vwwr, vwwr);
    svfloat64_t vql = svdiv_f64_z(mask,
				  svmul_f64_z(mask, vtwo, svmul_f64_z(mask, vwwl, vswwl)),
				  svadd_f64_z(mask, vswwl, vcl));
    svfloat64_t vqr = svdiv_f64_z(mask,
				  svmul_f64_z(mask, vtwo, svmul_f64_z(mask, vwwr, vswwr)),
				  svadd_f64_z(mask, vswwr, vcr));
    svfloat64_t vusl = svsub_f64_z(mask,
				   vul,
				   svdiv_f64_z(mask,
					       svsub_f64_z(mask, vpst, vpl),
					       vwwl));
    svfloat64_t vusr = svadd_f64_z(mask,
				   vur,
				   svdiv_f64_z(mask,
					       svsub_f64_z(mask, vpst, vpr),
					       vwwr));
    svfloat64_t vtmpmiddle = svdiv_f64_z(mask,
					 vql,
					 svadd_f64_z(mask, vqr, vql));
    svfloat64_t vtmpfront = svmul_f64_z(mask,
					vqr,
					vtmpmiddle);
    svfloat64_t vtmp = svmul_f64_z(mask,
				   vtmpfront,
				   svsub_f64_z(mask, vusl, vusr));
    svfloat64_t vdelp_i = svmax_f64_z(mask, vtmp, svneg_f64_z(mask, vpst));
    vpst = svadd_f64_z(mask, vpst, vdelp_i);
    // Convergence indicator
    svfloat64_t vtmp2 = svdiv_f64_z(mask,
				    vdelp_i,
				    svadd_f64_z(mask, vpst, vsmallpp));
    svfloat64_t vuo_i = svabs_f64_z(mask, vtmp2);

    //svbool_t notconverged = svcmpgt_f64(mask, vuo_i, svdup_n_f64(PRECISION));
    svbool_t converged = svcmple_f64(mask, vuo_i, svdup_n_f64(PRECISION));
#if 0
    svst1_s64(converged, goon64, svdup_n_s64(0));
    for (j = 0 ; j < vc/sizeof(real_t) ; j++) goon[i + j] = goon64[j];
#else
    svbool_t converged32 = svuzp1_b32(converged, svpfalse_b()); // zip to convert the mask to the narrower type
    svst1_s32(converged32, goon + i, svdup_n_s32(0));
#endif

    // FLOPS(29, 10, 2, 0);
    svst1_f64(mask, pstar + i, vpst);

    i += vc/sizeof(real_t);
  }
}

#else

//#define SOLVE_USE_FUNC

#ifdef SOLVE_USE_FUNC

#define EMBED_MASKING

#ifdef EMBED_MASKING
void solve_one_masking_sve(real_t *restrict pstar,
			  const real_t * const restrict ul,
			  const real_t * const restrict pl,
			  const real_t * const restrict ur,
			  const real_t * const restrict pr,
			  const real_t * const restrict cl,
			  const real_t * const restrict cr,
			  int * restrict goon,
			  const svfloat64_t vgamma6,
			  const svfloat64_t vsmallpp,
			  svbool_t mandatory) {
  //printf("%s\n", __PRETTY_FUNCTION__);
  /* FIXME: won't work, see solve_all_masking_sve  */
  svbool_t mask = svcmpeq_s32(mandatory, svld1_s32(svptrue_b32(), goon), svdup_n_s32(1)); // overshoot ...
  /* assume real_t == double */
  svfloat64_t vone = svdup_n_f64(1.);
  svfloat64_t vtwo = svdup_n_f64(2.);
  svfloat64_t vpst = svld1_f64(mask, pstar);
  svfloat64_t vul = svld1_f64(mask, ul);
  svfloat64_t vpl = svld1_f64(mask, pl);
  svfloat64_t vur = svld1_f64(mask, ur);
  svfloat64_t vpr = svld1_f64(mask, pr);
  svfloat64_t vcl = svld1_f64(mask, cl);
  svfloat64_t vcr = svld1_f64(mask, cr);
  // Newton-Raphson iterations to find pstar at the required accuracy
  svfloat64_t vwwl = svsqrt_f64_z(mask,
				  svmul_f64_z(mask,
					      vcl,
					      svadd_f64_z(mask,
							  vone,
							  svmul_f64_z(mask,
								      vgamma6,
								      svdiv_f64_z(mask,
										  svsub_f64_z(mask, vpst, vpl),
										  vpl)))));
  svfloat64_t vwwr = svsqrt_f64_z(mask,
				  svmul_f64_z(mask,
					      vcr,
					      svadd_f64_z(mask,
							  vone,
							  svmul_f64_z(mask,
								      vgamma6,
								      svdiv_f64_z(mask,
										  svsub_f64_z(mask, vpst, vpr),
										  vpr)))));
  svfloat64_t vswwl = svmul_f64_z(mask, vwwl, vwwl);
  svfloat64_t vswwr = svmul_f64_z(mask, vwwr, vwwr);
  svfloat64_t vql = svdiv_f64_z(mask,
				svmul_f64_z(mask, vtwo, svmul_f64_z(mask, vwwl, vswwl)),
				svadd_f64_z(mask, vswwl, vcl));
  svfloat64_t vqr = svdiv_f64_z(mask,
				svmul_f64_z(mask, vtwo, svmul_f64_z(mask, vwwr, vswwr)),
				svadd_f64_z(mask, vswwr, vcr));
  svfloat64_t vusl = svsub_f64_z(mask,
			       vul,
			       svdiv_f64_z(mask,
					   svsub_f64_z(mask, vpst, vpl),
					   vwwl));
  svfloat64_t vusr = svadd_f64_z(mask,
			       vur,
			       svdiv_f64_z(mask,
					   svsub_f64_z(mask, vpst, vpr),
					   vwwr));
  svfloat64_t vtmpmiddle = svdiv_f64_z(mask,
				     vql,
				     svadd_f64_z(mask, vqr, vql));
  svfloat64_t vtmpfront = svmul_f64_z(mask,
				    vqr,
				    vtmpmiddle);
  svfloat64_t vtmp = svmul_f64_z(mask,
			       vtmpfront,
			       svsub_f64_z(mask, vusl, vusr));
  svfloat64_t vdelp_i = svmax_f64_z(mask, vtmp, svneg_f64_z(mask, vpst));
  vpst = svadd_f64_z(mask, vpst, vdelp_i);
  // Convergence indicator
  svfloat64_t vtmp2 = svdiv_f64_z(mask,
				  vdelp_i,
				  svadd_f64_z(mask, vpst, vsmallpp));
  svfloat64_t vuo_i = svabs_f64_z(mask, vtmp2);

  //svbool_t notconverged = svcmpgt_f64(mask, vuo_i, svdup_n_f64(PRECISION));
  svbool_t converged = svcmple_f64(mask, vuo_i, svdup_n_f64(PRECISION));
  /* FIXME: won't work, see solve_all_masking_sve  */
  svst1_s32(converged, goon, svdup_n_s32(0)); // hopefully doesn't overshoot the array...

  // FLOPS(29, 10, 2, 0);
  svst1_f64(mask, pstar, vpst);
}

/* #pragma omp declare variant(solve_one_masking_sve)	\ */
/*   match(construct = {simd(notinbranch,linear(pstar,ul,pl,ur,pr,cl,cr,goon))}, \ */
/* 	device = {isa("sve")},				\ */
/* 	implementation = {extension("scalable")}) */
#pragma omp declare variant(solve_one_masking_sve)	\
  match(construct = {simd(notinbranch,linear(pstar),linear(ul),linear(pl),linear(ur),linear(pr),linear(cl),linear(cr),linear(goon))}, \
	device = {isa("sve")},				\
	implementation = {extension("scalable")})
#endif // EMBED_MASKING
#pragma omp declare simd linear(pstar,ul,pl,ur,pr,cl,cr,goon) inbranch
void solve_one_masking(real_t *restrict pstar,
				    const real_t * const restrict ul,
				    const real_t * const restrict pl,
				    const real_t * const restrict ur,
				    const real_t * const restrict pr,
				    const real_t * const restrict cl,
				    const real_t * const restrict cr,
		      	   	    int * restrict goon,
		      	   	    const real_t gamma6,
		       const real_t smallpp) {
  if (!(*goon)) return;
  real_t pst = *pstar;
  // Newton-Raphson iterations to find pstar at the required accuracy
  real_t wwl = MYSQRT(*cl * (one + gamma6 * (pst - *pl) / *pl));
  real_t wwr = MYSQRT(*cr * (one + gamma6 * (pst - *pr) / *pr));
  real_t swwl = Square(wwl);
  real_t ql = two * wwl * swwl / (swwl + *cl);
  real_t qr = two * wwr * Square(wwr) / (Square(wwr) + *cr);
  real_t usl = *ul - (pst - *pl) / wwl;
  real_t usr = *ur + (pst - *pr) / wwr;
  real_t tmp = (qr * ql / (qr + ql) * (usl - usr));
  real_t delp_i = fmax(tmp, (-pst));
  // *pstar = *pstar + delp_i;
  pst += delp_i;
  // Convergence indicator
  real_t tmp2 = delp_i / (pst + smallpp);
  real_t uo_i = fabs(tmp2);
  *goon = uo_i > PRECISION;
  // FLOPS(29, 10, 2, 0);
  *pstar = pst;
}

#else 

//#define UNIFORM_IS_VECTOR
//#define UNIFORM_IS_GLOBAL

#ifdef  UNIFORM_IS_VECTOR
void solve_one_direct_sve(real_t *restrict pstar,
			  const real_t * const restrict ul,
			  const real_t * const restrict pl,
			  const real_t * const restrict ur,
			  const real_t * const restrict pr,
			  const real_t * const restrict cl,
			  const real_t * const restrict cr,
			  int * restrict goon,
			  const svfloat64_t vgamma6,
			  const svfloat64_t vsmallpp,
			  svbool_t mask) {
  printf("%s\n", __PRETTY_FUNCTION__);
  /* assume real_t == double */
  svfloat64_t vone = svdup_n_f64(1.);
  svfloat64_t vtwo = svdup_n_f64(2.);
  svfloat64_t vpst = svld1_f64(mask, pstar);
  svfloat64_t vul = svld1_f64(mask, ul);
  svfloat64_t vpl = svld1_f64(mask, pl);
  svfloat64_t vur = svld1_f64(mask, ur);
  svfloat64_t vpr = svld1_f64(mask, pr);
  svfloat64_t vcl = svld1_f64(mask, cl);
  svfloat64_t vcr = svld1_f64(mask, cr);
  // Newton-Raphson iterations to find pstar at the required accuracy
  svfloat64_t vwwl = svsqrt_f64_z(mask,
				  svmul_f64_z(mask,
					      vcl,
					      svadd_f64_z(mask,
							  vone,
							  svmul_f64_z(mask,
								      vgamma6,
								      svdiv_f64_z(mask,
										  svsub_f64_z(mask, vpst, vpl),
										  vpl)))));
  svfloat64_t vwwr = svsqrt_f64_z(mask,
				  svmul_f64_z(mask,
					      vcr,
					      svadd_f64_z(mask,
							  vone,
							  svmul_f64_z(mask,
								      vgamma6,
								      svdiv_f64_z(mask,
										  svsub_f64_z(mask, vpst, vpr),
										  vpr)))));
  svfloat64_t vswwl = svmul_f64_z(mask, vwwl, vwwl);
  svfloat64_t vswwr = svmul_f64_z(mask, vwwr, vwwr);
  svfloat64_t vql = svdiv_f64_z(mask,
				svmul_f64_z(mask, vtwo, svmul_f64_z(mask, vwwl, vswwl)),
				svadd_f64_z(mask, vswwl, vcl));
  svfloat64_t vqr = svdiv_f64_z(mask,
				svmul_f64_z(mask, vtwo, svmul_f64_z(mask, vwwr, vswwr)),
				svadd_f64_z(mask, vswwr, vcr));
  svfloat64_t vusl = svsub_f64_z(mask,
			       vul,
			       svdiv_f64_z(mask,
					   svsub_f64_z(mask, vpst, vpl),
					   vwwl));
  svfloat64_t vusr = svadd_f64_z(mask,
			       vur,
			       svdiv_f64_z(mask,
					   svsub_f64_z(mask, vpst, vpr),
					   vwwr));
  svfloat64_t vtmpmiddle = svdiv_f64_z(mask,
				     vql,
				     svadd_f64_z(mask, vqr, vql));
  svfloat64_t vtmpfront = svmul_f64_z(mask,
				    vqr,
				    vtmpmiddle);
  svfloat64_t vtmp = svmul_f64_z(mask,
			       vtmpfront,
			       svsub_f64_z(mask, vusl, vusr));
  svfloat64_t vdelp_i = svmax_f64_z(mask, vtmp, svneg_f64_z(mask, vpst));
  vpst = svadd_f64_z(mask, vpst, vdelp_i);
  // Convergence indicator
  svfloat64_t vtmp2 = svdiv_f64_z(mask,
				  vdelp_i,
				  svadd_f64_z(mask, vpst, vsmallpp));
  svfloat64_t vuo_i = svabs_f64_z(mask, vtmp2);

  //svbool_t notconverged = svcmpgt_f64(mask, vuo_i, svdup_n_f64(PRECISION));
  /* FIXME: won't work, see solve_all_masking_sve  */
  svbool_t converged = svcmple_f64(mask, vuo_i, svdup_n_f64(PRECISION));
  svst1_s32(converged, goon, svdup_n_s32(0)); // hopefully doesn't overshoot the array...

  // FLOPS(29, 10, 2, 0);
  svst1_f64(mask, pstar, vpst);
}

#pragma omp declare variant(solve_one_direct_sve)	\
  match(construct = {simd(inbranch,linear(pstar,ul,pl,ur,pr,cl,cr,goon))}, \
	device = {isa("sve")},				\
	implementation = {extension("scalable")})
//#pragma omp declare simd linear(i,pstar,ul,pl,ur,pr,cl,cr,goon) inbranch
void solve_one_direct(real_t *restrict pstar,
				    const real_t * const restrict ul,
				    const real_t * const restrict pl,
				    const real_t * const restrict ur,
				    const real_t * const restrict pr,
				    const real_t * const restrict cl,
				    const real_t * const restrict cr,
		      	   	    int * restrict goon,
		      	   	    const real_t gamma6,
			  	    const real_t smallpp) {
  real_t pst = *pstar;
  // Newton-Raphson iterations to find pstar at the required accuracy
  real_t wwl = MYSQRT(*cl * (one + gamma6 * (pst - *pl) / *pl));
  real_t wwr = MYSQRT(*cr * (one + gamma6 * (pst - *pr) / *pr));
  real_t swwl = Square(wwl);
  real_t ql = two * wwl * swwl / (swwl + *cl);
  real_t qr = two * wwr * Square(wwr) / (Square(wwr) + *cr);
  real_t usl = *ul - (pst - *pl) / wwl;
  real_t usr = *ur + (pst - *pr) / wwr;
  real_t tmp = (qr * ql / (qr + ql) * (usl - usr));
  real_t delp_i = fmax(tmp, (-pst));
  // *pstar = *pstar + delp_i;
  pst += delp_i;
  // Convergence indicator
  real_t tmp2 = delp_i / (pst + smallpp);
  real_t uo_i = fabs(tmp2);
  *goon = uo_i > PRECISION;
  // FLOPS(29, 10, 2, 0);
  *pstar = pst;
}
#endif // UNIFORM_IS_VECTOR

#ifdef UNIFORM_IS_GLOBAL
/* super-ugly, non-thread safe workaround for lack of 'uniform'... use a couple of global variables for loop invariant */
static real_t ggamma6;
static real_t gsmallpp;

void solve_one_direct_sve(real_t *restrict pstar,
			  const real_t * const restrict ul,
			  const real_t * const restrict pl,
			  const real_t * const restrict ur,
			  const real_t * const restrict pr,
			  const real_t * const restrict cl,
			  const real_t * const restrict cr,
			  int * restrict goon,
			  svbool_t mask) {
  /* assume real_t == double */
  svfloat64_t vone = svdup_n_f64(1.);
  svfloat64_t vtwo = svdup_n_f64(2.);
  svfloat64_t vpst = svld1_f64(mask, pstar);
  svfloat64_t vul = svld1_f64(mask, ul);
  svfloat64_t vpl = svld1_f64(mask, pl);
  svfloat64_t vur = svld1_f64(mask, ur);
  svfloat64_t vpr = svld1_f64(mask, pr);
  svfloat64_t vcl = svld1_f64(mask, cl);
  svfloat64_t vcr = svld1_f64(mask, cr);
  svfloat64_t vgamma6 = svdup_n_f64(ggamma6);
  svfloat64_t vsmallpp = svdup_n_f64(gsmallpp);
  // Newton-Raphson iterations to find pstar at the required accuracy
  svfloat64_t vwwl = svsqrt_f64_z(mask,
				  svmul_f64_z(mask,
					      vcl,
					      svadd_f64_z(mask,
							  vone,
							  svmul_f64_z(mask,
								      vgamma6,
								      svdiv_f64_z(mask,
										  svsub_f64_z(mask, vpst, vpl),
										  vpl)))));
  svfloat64_t vwwr = svsqrt_f64_z(mask,
				  svmul_f64_z(mask,
					      vcr,
					      svadd_f64_z(mask,
							  vone,
							  svmul_f64_z(mask,
								      vgamma6,
								      svdiv_f64_z(mask,
										  svsub_f64_z(mask, vpst, vpr),
										  vpr)))));
  svfloat64_t vswwl = svmul_f64_z(mask, vwwl, vwwl);
  svfloat64_t vswwr = svmul_f64_z(mask, vwwr, vwwr);
  svfloat64_t vql = svdiv_f64_z(mask,
				svmul_f64_z(mask, vtwo, svmul_f64_z(mask, vwwl, vswwl)),
				svadd_f64_z(mask, vswwl, vcl));
  svfloat64_t vqr = svdiv_f64_z(mask,
				svmul_f64_z(mask, vtwo, svmul_f64_z(mask, vwwr, vswwr)),
				svadd_f64_z(mask, vswwr, vcr));
  svfloat64_t vusl = svsub_f64_z(mask,
			       vul,
			       svdiv_f64_z(mask,
					   svsub_f64_z(mask, vpst, vpl),
					   vwwl));
  svfloat64_t vusr = svadd_f64_z(mask,
			       vur,
			       svdiv_f64_z(mask,
					   svsub_f64_z(mask, vpst, vpr),
					   vwwr));
  svfloat64_t vtmpmiddle = svdiv_f64_z(mask,
				     vql,
				     svadd_f64_z(mask, vqr, vql));
  svfloat64_t vtmpfront = svmul_f64_z(mask,
				    vqr,
				    vtmpmiddle);
  svfloat64_t vtmp = svmul_f64_z(mask,
			       vtmpfront,
			       svsub_f64_z(mask, vusl, vusr));
  svfloat64_t vdelp_i = svmax_f64_z(mask, vtmp, svneg_f64_z(mask, vpst));
  vpst = svadd_f64_z(mask, vpst, vdelp_i);
  // Convergence indicator
  svfloat64_t vtmp2 = svdiv_f64_z(mask,
				  vdelp_i,
				  svadd_f64_z(mask, vpst, vsmallpp));
  svfloat64_t vuo_i = svabs_f64_z(mask, vtmp2);

  //svbool_t notconverged = svcmpgt_f64(mask, vuo_i, svdup_n_f64(PRECISION));
  /* FIXME: won't work, see solve_all_masking_sve  */
  svbool_t converged = svcmple_f64(mask, vuo_i, svdup_n_f64(PRECISION));
  svst1_s32(converged, goon, svdup_n_s32(0)); // hopefully doesn't overshoot the array...

  // FLOPS(29, 10, 2, 0);
  svst1_f64(mask, pstar, vpst);
}

#pragma omp declare variant(solve_one_direct_sve)	\
  match(construct = {simd(inbranch,linear(pstar,ul,pl,ur,pr,cl,cr,goon))}, \
	device = {isa("sve")},				\
	implementation = {extension("scalable")})
//#pragma omp declare simd linear(i,pstar,ul,pl,ur,pr,cl,cr,goon) inbranch
void solve_one_direct(real_t *restrict pstar,
				    const real_t * const restrict ul,
				    const real_t * const restrict pl,
				    const real_t * const restrict ur,
				    const real_t * const restrict pr,
				    const real_t * const restrict cl,
				    const real_t * const restrict cr,
				    int * restrict goon) {
  real_t pst = *pstar;
  // Newton-Raphson iterations to find pstar at the required accuracy
  real_t wwl = MYSQRT(*cl * (one + ggamma6 * (pst - *pl) / *pl));
  real_t wwr = MYSQRT(*cr * (one + ggamma6 * (pst - *pr) / *pr));
  real_t swwl = Square(wwl);
  real_t ql = two * wwl * swwl / (swwl + *cl);
  real_t qr = two * wwr * Square(wwr) / (Square(wwr) + *cr);
  real_t usl = *ul - (pst - *pl) / wwl;
  real_t usr = *ur + (pst - *pr) / wwr;
  real_t tmp = (qr * ql / (qr + ql) * (usl - usr));
  real_t delp_i = fmax(tmp, (-pst));
  // *pstar = *pstar + delp_i;
  pst += delp_i;
  // Convergence indicator
  real_t tmp2 = delp_i / (pst + gsmallpp);
  real_t uo_i = fabs(tmp2);
  *goon = uo_i > PRECISION;
  // FLOPS(29, 10, 2, 0);
  *pstar = pst;
}
#endif // UNIFORM_IS_GLOBAL

#if !defined(UNIFORM_IS_VECTOR) && !defined(UNIFORM_IS_GLOBAL)
// not yet doable this way, because the current Arm Compiler (20.0) doesn't support 'uniform' <https://developer.arm.com/docs/101458/2000/vector-math-routines/interface-user-vector-functions-with-serial-code>
// For 20.1, see <https://developer.arm.com/docs/101458/2010/vector-routines-support/support-level-for-declare-simd>
#pragma omp declare simd linear(i) uniform(pstar,ul,pl,ur,pr,cl,cr,goon,gamma6,smallpp) inbranch
/* static inline  */ void solve_one_clean(real_t *restrict pstar,
			     const real_t * const restrict ul,
			     const real_t * const restrict pl,
			     const real_t * const restrict ur,
			     const real_t * const restrict pr,
			     const real_t * const restrict cl,
			     const real_t * const restrict cr,
			     int * restrict goon,
			     const int i,
			     const real_t gamma6,
			     const real_t smallpp) {
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
  real_t delp_i = fmax(tmp, (-pst));
  // pstar[i] = pstar[i] + delp_i;
  pst += delp_i;
  // Convergence indicator
  real_t tmp2 = delp_i / (pst + smallpp);
  real_t uo_i = fabs(tmp2);
  goon[i] = uo_i > PRECISION;
  // FLOPS(29, 10, 2, 0);
  pstar[i] = pst;
}

#endif // EMBED_MASKING

#endif // SOLVE_USE_FUNC

#endif // SOLVE_ALL_FUNC

#endif // __aarch64__

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
	hydrowork_t * restrict Hw) 
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
//#pragma message "active pragma simd "
#define SIMDNEEDED 1
#define __ICC 200000
#if defined(__ICC) && __ICC < 1300
#define SIMD ivdep
#else
#define SIMD omp simd
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
#pragma omp simd
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
#ifdef SOLVE_ALL_FUNC
      solve_all_masking_sve(s,
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
#else
#ifdef SIMDNEEDED
#if __ICC < 1300
#pragma simd
#else
//#pragma omp simd
#endif
#endif
#pragma omp simd
      for (i = 0; i < narray; i++) {
#if defined (EMBED_MASKING)
	  solve_one_masking(&pstar[i],&ul[i],&pl[i],&ur[i],&pr[i],&cl[i],&cr[i],&goon[i], gamma6, smallpp);
#else
	/* if (goon[i]) */ {
#ifndef SOLVE_USE_FUNC
	  real_t pst = pstar[i];
real_t oldpst = pst;
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
pst = goon[i] ? pst : oldpst;
	  goon[i] = uo_i > PRECISION;
	  // FLOPS(29, 10, 2, 0);
	  pstar[i] = pst;
#else
#if !defined(UNIFORM_IS_VECTOR) && !defined(UNIFORM_IS_GLOBAL)
	  solve_one_clean(pstar,ul,pl,ur,pr,cl,cr,goon,i,gamma6,smallpp);
#elif defined(UNIFORM_IS_VECTOR)
	  solve_one_direct(&pstar[i],&ul[i],&pl[i],&ur[i],&pr[i],&cl[i],&cr[i],&goon[i], gamma6, smallpp);
#elif defined(UNIFORM_IS_GLOBAL)
	  ggamma6 = gamma6;
	  gsmallpp = smallpp;
	  solve_one_direct(&pstar[i],&ul[i],&pl[i],&ur[i],&pr[i],&cl[i],&cr[i],&goon[i]);
#else
#error "Uh?"
#endif // UNIFORM_IS_*
#endif // SOLVE_USE_FUNC
	}
#endif // EMBED_MASKING
      }
#endif // SOLVE_ALL_FUNC
    }                           // iter_riemann

#ifdef SIMDNEEDED
//#pragma omp simd
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
#pragma omp simd
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
