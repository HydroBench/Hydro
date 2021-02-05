#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <stdio.h>

#include "parametres.h"
#include "utils.h"
#include "trace.h"
#include "perfcnt.h"
#include "cclock.h"

void
trace(const real_t dtdx,
      const int n,
      const int Hscheme,
      const int Hnvar,
      const int Hnxyt,
      const int slices, const int Hstep,
      real_t q[Hnvar][Hstep][Hnxyt],
      real_t dq[Hnvar][Hstep][Hnxyt],
      real_t c[Hstep][Hnxyt],
      real_t qxm[Hnvar][Hstep][Hnxyt], real_t qxp[Hnvar][Hstep][Hnxyt])
{
    int ijmin, ijmax;
    int i, IN, s;
    real_t zerol = 0.0, zeror = 0.0, project = 0.;
    struct timespec start, end;

    WHERE("trace");
#ifdef TRACKDATA
    fprintf(stderr, "Moving trace IN\n");
#endif
    start = cclock();
    ijmin = 0;
    ijmax = n;

    // if (strcmp(Hscheme, "muscl") == 0) {       // MUSCL-Hancock method
    if (Hscheme == HSCHEME_MUSCL) {	// MUSCL-Hancock method
	zerol = -hundred / dtdx;
	zeror = hundred / dtdx;
	project = one;
    }
    // if (strcmp(Hscheme, "plmde") == 0) {       // standard PLMDE
    if (Hscheme == HSCHEME_PLMDE) {	// standard PLMDE
	zerol = zero;
	zeror = zero;
	project = one;
    }
    // if (strcmp(Hscheme, "collela") == 0) {     // Collela's method
    if (Hscheme == HSCHEME_COLLELA) {	// Collela's method
	zerol = zero;
	zeror = zero;
	project = zero;
    }
#ifdef TARGETON
    // 
#pragma omp target \
 	map(c[0:Hstep][0:Hnxyt]) \
 	map(q[0:Hnvar][0:Hstep][0:Hnxyt])		\
 	map(dq[0:Hnvar][0:Hstep][0:Hnxyt])		\
 	map(qxp[0:Hnvar][0:Hstep][0:Hnxyt])		\
 	map(qxm[0:Hnvar][0:Hstep][0:Hnxyt])
#endif
#pragma omp TEAMSDIS parallel for default(none), private(s,i), 	\
	firstprivate(dtdx, slices, ijmin,ijmax,zeror,zerol, Hnxyt, Hstep, project) \
	shared(qxp, qxm, c, q, dq) collapse(2)
    for (s = 0; s < slices; s++) {
	for (i = ijmin + 1; i < ijmax - 1; i++) {
	    real_t cc, csq, r, u, v, p;
	    real_t dr, du, dv, dp;
	    real_t alpham, alphap, alpha0r, alpha0v;
	    real_t spminus, spplus, spzero;
	    real_t apright, amright, azrright, azv1right;
	    real_t apleft, amleft, azrleft, azv1left;

	    real_t upcc, umcc, upccx, umccx, ux;
	    real_t rOcc, OrOcc, dprcc;

	    cc = c[s][i];
	    csq = Square(cc);
	    r = q[ID][s][i];
	    u = q[IU][s][i];
	    v = q[IV][s][i];
	    p = q[IP][s][i];
	    dr = dq[ID][s][i];
	    du = dq[IU][s][i];
	    dv = dq[IV][s][i];
	    dp = dq[IP][s][i];
	    rOcc = r / cc;
	    OrOcc = cc / r;
	    dprcc = dp / (r * cc);
	    alpham = half * (dprcc - du) * rOcc;
	    alphap = half * (dprcc + du) * rOcc;
	    alpha0r = dr - dp / csq;
	    alpha0v = dv;
	    upcc = u + cc;
	    umcc = u - cc;
	    upccx = upcc * dtdx;
	    umccx = umcc * dtdx;
	    ux = u * dtdx;

	    // Right state
	    spminus = (umcc >= zeror) ? (project) : umccx + one;
	    spplus = (upcc >= zeror) ? (project) : upccx + one;
	    spzero = (u >= zeror) ? (project) : ux + one;
	    apright = -half * spplus * alphap;
	    amright = -half * spminus * alpham;
	    azrright = -half * spzero * alpha0r;
	    azv1right = -half * spzero * alpha0v;
	    qxp[ID][s][i] = r + (apright + amright + azrright);
	    qxp[IU][s][i] = u + (apright - amright) * OrOcc;
	    qxp[IV][s][i] = v + (azv1right);
	    qxp[IP][s][i] = p + (apright + amright) * csq;

	    // Left state
	    spminus = (umcc <= zerol) ? (-project) : umccx - one;
	    spplus = (upcc <= zerol) ? (-project) : upccx - one;
	    spzero = (u <= zerol) ? (-project) : ux - one;
	    apleft = -half * spplus * alphap;
	    amleft = -half * spminus * alpham;
	    azrleft = -half * spzero * alpha0r;
	    azv1left = -half * spzero * alpha0v;
	    qxm[ID][s][i] = r + (apleft + amleft + azrleft);
	    qxm[IU][s][i] = u + (apleft - amleft) * OrOcc;
	    qxm[IV][s][i] = v + (azv1left);
	    qxm[IP][s][i] = p + (apleft + amleft) * csq;
	}
    }

    {
	int nops = slices * ((ijmax - 1) - (ijmin + 1));
	FLOPS(77 * nops, 7 * nops, 0 * nops, 0 * nops);
    }

    if (Hnvar > IP) {
#ifdef TARGETON
	// 
#pragma omp target \
 	map(c[0:Hstep][0:Hnxyt]) \
 	map(q[0:Hnvar][0:Hstep][0:Hnxyt])		\
 	map(dq[0:Hnvar][0:Hstep][0:Hnxyt])		\
 	map(qxp[0:Hnvar][0:Hstep][0:Hnxyt])		\
 	map(qxm[0:Hnvar][0:Hstep][0:Hnxyt])
#endif
#pragma omp TEAMSDIS parallel for default(none), private(s,i), 	\
	firstprivate(dtdx, slices, Hnvar, ijmin,ijmax,zeror,zerol, Hnxyt, Hstep, project) \
	shared(qxp, qxm, c, q, dq) collapse(3)
	for (IN = IP + 1; IN < Hnvar; IN++) {
	    for (s = 0; s < slices; s++) {
		for (i = ijmin + 1; i < ijmax - 1; i++) {
		    real_t u, a;
		    real_t da;
		    real_t spzero;
		    real_t acmpright;
		    real_t acmpleft;
		    u = q[IU][s][i];
		    a = q[IN][s][i];
		    da = dq[IN][s][i];

		    // Right state
		    spzero = u * dtdx + one;
		    if (u >= zeror) {
			spzero = project;
		    }
		    acmpright = -half * spzero * da;
		    qxp[IN][s][i] = a + acmpright;

		    // Left state
		    spzero = u * dtdx - one;
		    if (u <= zerol) {
			spzero = -project;
		    }
		    acmpleft = -half * spzero * da;
		    qxm[IN][s][i] = a + acmpleft;
		}
	    }
	}
    }
    end = cclock();
    functim[TIM_TRACE] += ccelaps(start, end);
#ifdef TRACKDATA
    fprintf(stderr, "Moving trace OUT\n");
#endif
}				// trace

//EOF
