#include <stdio.h>
#include <string.h>
#include <malloc.h>
// #include <unistd.h>
#include <math.h>

#include "parametres.h"
#include "compute_deltat.h"
#include "utils.h"
#include "perfcnt.h"
#include "equation_of_state.h"

#define DABS(x) (real_t) fabs((x))

void
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
    )
{
    int i, s;

#define IHV(i, j, v)  ((i) + Hnxt * ((j) + Hnyt * (v)))

#pragma omp parallel for shared(q, e) private(s, i) COLLAPSE
    for (s = 0; s < slices; s++) {
	for (i = 0; i < Hnx; i++) {
	    real_t eken;
	    real_t tmp;
	    int idxuID = IHV(i + ExtraLayer, j + s, ID);
	    int idxuIU = IHV(i + ExtraLayer, j + s, IU);
	    int idxuIV = IHV(i + ExtraLayer, j + s, IV);
	    int idxuIP = IHV(i + ExtraLayer, j + s, IP);
	    q[ID][s][i] = MAX(uold[idxuID], Hsmallr);
	    q[IU][s][i] = uold[idxuIU] / q[ID][s][i];
	    q[IV][s][i] = uold[idxuIV] / q[ID][s][i];
	    eken = half * (Square(q[IU][s][i]) + Square(q[IV][s][i]));
	    tmp = uold[idxuIP] / q[ID][s][i] - eken;
	    q[IP][s][i] = tmp;
	    e[s][i] = tmp;
	}
    }
    {
	int nops = slices * Hnx;
	FLOPS(5 * nops, 3 * nops, 1 * nops, 0 * nops);
    }
#undef IHV
#undef IHVW
}

// to force a parallel reduction with OpenMP
#define WOMP

void
courantOnXY(real_t * cournox,
	    real_t * cournoy,
	    const int Hnx,
	    const int Hnxyt,
	    const int Hnvar, const int slices, const int Hstep,
	    real_t c[Hstep][Hnxyt], real_t q[Hnvar][Hstep][Hnxyt],
	    real_t * tmpm1, real_t * tmpm2)
{
#ifdef WOMP
    int s, i;
    // real_t maxValC = zero;
    real_t tmp1 = *cournox, tmp2 = *cournoy;

#pragma omp parallel for shared(tmpm1, tmpm2) private(s,i) reduction(max:tmp1) reduction(max:tmp2)
    for (s = 0; s < slices; s++) {
	for (i = 0; i < Hnx; i++) {
	    tmp1 = MAX(tmp1, c[s][i] + DABS(q[IU][s][i]));
	    tmp2 = MAX(tmp2, c[s][i] + DABS(q[IV][s][i]));
	}
    }
    *cournox = tmp1;
    *cournoy = tmp2;
    {
	int nops = (slices) * Hnx;
	FLOPS(2 * nops, 0 * nops, 2 * nops, 0 * nops);
    }
#else
    int i, s;
    real_t tmp1, tmp2;
    for (s = 0; s < slices; s++) {
	for (i = 0; i < Hnx; i++) {
	    tmp1 = c[s][i] + DABS(q[IU][s][i]);
	    tmp2 = c[s][i] + DABS(q[IV][s][i]);
	    *cournox = MAX(*cournox, tmp1);
	    *cournoy = MAX(*cournoy, tmp2);
	}
    }
    {
	int nops = (slices) * Hnx;
	FLOPS(2 * nops, 0 * nops, 5 * nops, 0 * nops);
    }
#endif
#undef IHVW
}

void compute_deltat_init_mem(const hydroparam_t H, hydrowork_t * Hw,
			     hydrovarwork_t * Hvw)
{
    Hvw->q = (real_t(*))DMalloc(H.nvar * H.nxyt * H.nxystep);
    Hw->e = (real_t(*))DMalloc(H.nxyt * H.nxystep);
    Hw->c = (real_t(*))DMalloc(H.nxyt * H.nxystep);
    Hw->tmpm1 = (real_t *) DMalloc(H.nxystep);
    Hw->tmpm2 = (real_t *) DMalloc(H.nxystep);

}

void compute_deltat_clean_mem(const hydroparam_t H, hydrowork_t * Hw,
			      hydrovarwork_t * Hvw)
{
    DFree(&Hvw->q, H.nvar * H.nxyt * H.nxystep);
    DFree(&Hw->e, H.nxyt * H.nxystep);
    DFree(&Hw->c, H.nxyt * H.nxystep);
    DFree(&Hw->tmpm1, H.nxystep);
    DFree(&Hw->tmpm2, H.nxystep);
}

void
compute_deltat(real_t * dt, const hydroparam_t H, hydrowork_t * Hw,
	       hydrovar_t * Hv, hydrovarwork_t * Hvw)
{
    real_t cournox, cournoy;
    int j, jend, slices, Hstep, Hmin, Hmax;
    real_t(*e)[H.nxyt];
    real_t(*c)[H.nxystep];
    real_t(*q)[H.nxystep][H.nxyt];
    WHERE("compute_deltat");

    //   compute time step on grid interior
    cournox = zero;
    cournoy = zero;

    c = (real_t(*)[H.nxystep]) Hw->c;
    e = (real_t(*)[H.nxystep]) Hw->e;
    q = (real_t(*)[H.nxystep][H.nxyt]) Hvw->q;

    Hstep = H.nxystep;
    Hmin = H.jmin + ExtraLayer;
    Hmax = H.jmax - ExtraLayer;
    for (j = Hmin; j < Hmax; j += Hstep) {
	jend = j + Hstep;
	if (jend >= Hmax)
	    jend = Hmax;
	slices = jend - j;	// numbre of slices to compute
	ComputeQEforRow(j, H.smallr, H.nx, H.nxt, H.nyt, H.nxyt, H.nvar, slices,
			Hstep, Hv->uold, q, e);
	equation_of_state(0, H.nx, H.nxyt, H.nvar, H.smallc, H.gamma, slices,
			  Hstep, e, q, c);
	courantOnXY(&cournox, &cournoy, H.nx, H.nxyt, H.nvar, slices, Hstep, c,
		    q, Hw->tmpm1, Hw->tmpm2);
	// fprintf(stdout, "[%2d]\t%g %g %g %g\n", H.mype, cournox, cournoy, H.smallc, H.courant_factor);
    }

    *dt = H.courant_factor * H.dx / MAX(cournox, MAX(cournoy, H.smallc));
    FLOPS(1, 1, 2, 0);
    // fprintf(stdout, "[%2d]\t%g %g %g %g %g %g\n", H.mype, cournox, cournoy, H.smallc, H.courant_factor, H.dx, *dt);
}				// compute_deltat

//EOF
