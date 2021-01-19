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
#include "courantOnXY.h"
#include "ComputeQEforRow.h"

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
#ifdef TRACKDATA
	fprintf(stderr, "Moving compute_deltat IN\n");
#endif
	
    //   compute time step on grid interior
    cournox = zero;
    cournoy = zero;

    c = (real_t(*)[H.nxystep]) Hw->c;
    e = (real_t(*)[H.nxystep]) Hw->e;
    q = (real_t(*)[H.nxystep][H.nxyt]) Hvw->q;

    Hstep = H.nxystep;
    Hmin = H.jmin + ExtraLayer;
    Hmax = H.jmax - ExtraLayer;
// #ifdef TARGETON
// #pragma omp target data	\
// 	map(tofrom: Hv->uold) \
// 	map(tofrom: q)	\
// 	map(tofrom: c)\
// 	map(tofrom: e)
// #endif
    {
	for (j = Hmin; j < Hmax; j += Hstep) {
	    jend = j + Hstep;
	    if (jend >= Hmax)
		jend = Hmax;
	    slices = jend - j;	// numbre of slices to compute
	    {
		ComputeQEforRow(j, H.smallr, H.nx, H.nxt, H.nyt, H.nxyt, H.nvar,
				slices, Hstep, Hv->uold, q, e);
		equation_of_state(0, H.nx, H.nxyt, H.nvar, H.smallc, H.gamma,
				  slices, Hstep, e, q, c);
		courantOnXY(&cournox, &cournoy, H.nx, H.nxyt, H.nvar, slices,
			    Hstep, c, q, Hw->tmpm1, Hw->tmpm2);
	    }
	    // fprintf(stdout, "[%2d]\t%g %g %g %g\n", H.mype, cournox, cournoy, H.smallc, H.courant_factor);
	}
    }
#ifdef TRACKDATA
    fprintf(stderr, "Moving compute_deltat OUT\n");
#endif
    
    *dt = H.courant_factor * H.dx / MAX(cournox, MAX(cournoy, H.smallc));
    FLOPS(1, 1, 2, 0);
    // fprintf(stdout, "[%2d]\t%g %g %g %g %g %g\n", H.mype, cournox, cournoy, H.smallc, H.courant_factor, H.dx, *dt);
}				// compute_deltat

//EOF
