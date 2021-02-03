#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <stdio.h>
#include <strings.h>
#include <string.h>

#include "parametres.h"
#include "hydro_godunov.h"
#include "hydro_funcs.h"
#include "cclock.h"
#include "utils.h"
#include "make_boundary.h"

#include "riemann.h"
#include "qleftright.h"
#include "trace.h"
#include "slope.h"
#include "equation_of_state.h"
#include "constoprim.h"

#include "cmpflx.h"
#include "conservar.h"
#include "perfcnt.h"

void
hydro_godunov(int idimStart, real_t dt, const hydroparam_t H, hydrovar_t * Hv,
	      hydrowork_t * Hw, hydrovarwork_t * Hvw)
{
    // Local variables
    struct timespec start, end;
    int j;
    real_t dtdx;
    int clear = 0;

    real_t(*e)[H.nxyt];
    real_t(*flux)[H.nxystep][H.nxyt];
    real_t(*qleft)[H.nxystep][H.nxyt];
    real_t(*qright)[H.nxystep][H.nxyt];
    real_t(*c)[H.nxyt];
    real_t *uold;
    int (*sgnm)[H.nxyt];
    real_t(*qgdnv)[H.nxystep][H.nxyt];
    real_t(*u)[H.nxystep][H.nxyt];
    real_t(*qxm)[H.nxystep][H.nxyt];
    real_t(*qxp)[H.nxystep][H.nxyt];
    real_t(*q)[H.nxystep][H.nxyt];
    real_t(*dq)[H.nxystep][H.nxyt];
    int idimIndex = 0;

    static FILE *fic = NULL;

    WHERE("hydro_godunov");

#ifdef TRACKDATA
    fprintf(stderr, "Moving hydro_godunov IN\n");
#endif

    uold = Hv->uold;
    qgdnv = (real_t(*)[H.nxystep][H.nxyt]) Hvw->qgdnv;
    flux = (real_t(*)[H.nxystep][H.nxyt]) Hvw->flux;
    c = (real_t(*)[H.nxyt]) Hw->c;
    e = (real_t(*)[H.nxyt]) Hw->e;
    qleft = (real_t(*)[H.nxystep][H.nxyt]) Hvw->qleft;
    qright = (real_t(*)[H.nxystep][H.nxyt]) Hvw->qright;
    sgnm = (int (*)[H.nxyt])Hw->sgnm;
    q = (real_t(*)[H.nxystep][H.nxyt]) Hvw->q;
    dq = (real_t(*)[H.nxystep][H.nxyt]) Hvw->dq;
    u = (real_t(*)[H.nxystep][H.nxyt]) Hvw->u;
    qxm = (real_t(*)[H.nxystep][H.nxyt]) Hvw->qxm;
    qxp = (real_t(*)[H.nxystep][H.nxyt]) Hvw->qxp;

    for (idimIndex = 0; idimIndex < 2; idimIndex++) {
	int idim = (idimStart - 1 + idimIndex) % 2 + 1;
	int Hmin, Hmax, Hstep;
	int Hdimsize;
	int Hndim_1;
	int tmpsiz;

	// constant
	dtdx = dt / H.dx;

	// Update boundary conditions

#ifdef TARGETON
// #pragma omp target update from (uold [0:H.nvar *H.nxt * H.nyt])
#endif
	make_boundary(idim, H, Hv);
#ifdef TARGETON
// #pragma omp target update to (uold [0:H.nvar *H.nxt * H.nyt])
#endif

	if (idim == 1) {
	    Hmin = H.jmin + ExtraLayer;
	    Hmax = H.jmax - ExtraLayer;
	    Hdimsize = H.nxt;
	    Hndim_1 = H.nx + 1;
	    Hstep = H.nxystep;
	} else {
	    Hmin = H.imin + ExtraLayer;
	    Hmax = H.imax - ExtraLayer;
	    Hdimsize = H.nyt;
	    Hndim_1 = H.ny + 1;
	    Hstep = H.nxystep;
	}
	tmpsiz = H.nxyt * Hstep;

	for (j = Hmin; j < Hmax; j += Hstep) {
	    // we try to compute many slices each pass
	    int jend = j + Hstep;
	    if (jend >= Hmax)
		jend = Hmax;
	    int slices = jend - j;	// numbre of slices to compute

	    {
		gatherConservativeVars(idim, j, H.imin, H.imax, H.jmin, H.jmax,
				       H.nvar, H.nxt, H.nyt, H.nxyt, slices,
				       Hstep, uold, u);
		// Convert to primitive variables
		constoprim(Hdimsize, H.nxyt, H.nvar, H.smallr, slices, Hstep, u,
			   q, e);
		equation_of_state(0, Hdimsize, H.nxyt, H.nvar, H.smallc,
				  H.gamma, slices, Hstep, e, q, c);
		// Characteristic tracing
		if (H.iorder != 1) {
		    slope(Hdimsize, H.nvar, H.nxyt, H.slope_type, slices, Hstep,
			  q, dq);
		}
		trace(dtdx, Hdimsize, H.scheme, H.nvar, H.nxyt, slices, Hstep,
		      q, dq, c, qxm, qxp);
//
		start = cclock();
		qleftright(idim, H.nx, H.ny, H.nxyt, H.nvar, slices, Hstep, qxm,
			   qxp, qleft, qright);
		end = cclock();
//      
		functim[TIM_QLEFTR] += ccelaps(start, cclock());
		riemann(Hndim_1, H.smallr, H.smallc, H.gamma, H.niter_riemann,
			H.nvar, H.nxyt, slices, Hstep, qleft, qright, qgdnv,
			sgnm, Hw);
		cmpflx(Hdimsize, H.nxyt, H.nvar, H.gamma, slices, Hstep, qgdnv,
		       flux);
		updateConservativeVars(idim, j, dtdx, H.imin, H.imax, H.jmin,
				       H.jmax, H.nvar, H.nxt, H.nyt, H.nxyt,
				       slices, Hstep, uold, u, flux);
	    }
	}			// for j
    }
#ifdef TRACKDATA
    fprintf(stderr, "Moving hydro_godunov OUT\n");
#endif

}				// hydro_godunov

// EOF
