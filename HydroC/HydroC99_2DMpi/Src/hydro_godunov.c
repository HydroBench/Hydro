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
#include <strings.h>
#include <string.h>

#include "hmpp.h"
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

// variables auxiliaires pour mettre en place le mode resident de HMPP
void
hydro_godunov(int idimStart, real_t dt, const hydroparam_t H, hydrovar_t * Hv, hydrowork_t * Hw, hydrovarwork_t * Hvw) {
  // Local variables
  struct timespec start, end;
  int j;
  real_t dtdx;
  int clear=0;

  real_t (*e)[H.nxyt];
  real_t (*flux)[H.nxystep][H.nxyt];
  real_t (*qleft)[H.nxystep][H.nxyt];
  real_t (*qright)[H.nxystep][H.nxyt];
  real_t (*c)[H.nxyt];
  real_t *uold;
  int (*sgnm)[H.nxyt];
  real_t (*qgdnv)[H.nxystep][H.nxyt];
  real_t (*u)[H.nxystep][H.nxyt];
  real_t (*qxm)[H.nxystep][H.nxyt];
  real_t (*qxp)[H.nxystep][H.nxyt];
  real_t (*q)[H.nxystep][H.nxyt];
  real_t (*dq)[H.nxystep][H.nxyt];

  static FILE *fic = NULL;

  if (fic == NULL && H.prt == 1) {
    char logname[256];
    sprintf(logname, "TRACE.%04d_%04d.txt", H.nproc, H.mype);
    fic = fopen(logname, "w");
  }

  WHERE("hydro_godunov");

  // int hmppGuard = 1;
  int idimIndex = 0;

  for (idimIndex = 0; idimIndex < 2; idimIndex++) {
    int idim = (idimStart - 1 + idimIndex) % 2 + 1;
    // constant
    dtdx = dt / H.dx;

    // Update boundary conditions
    if (H.prt) {
      fprintf(fic, "godunov %d %le %le\n", idim, dt, H.t);
      PRINTUOLD(fic, H, Hv);
    }
    // if (H.mype == 1) fprintf(fic, "Hydro makes boundary.\n");
    start = cclock();
    make_boundary(idim, H, Hv);
    end = cclock();
    functim[TIM_MAKBOU] += ccelaps(start, end);

    if (H.prt) {fprintf(fic, "MakeBoundary\n");}
    PRINTUOLD(fic, H, Hv);

    uold = Hv->uold;
    qgdnv = (real_t (*)[H.nxystep][H.nxyt]) Hvw->qgdnv;
    flux = (real_t (*)[H.nxystep][H.nxyt]) Hvw->flux;
    c = (real_t (*)[H.nxyt]) Hw->c;
    e = (real_t (*)[H.nxyt]) Hw->e;
    qleft = (real_t (*)[H.nxystep][H.nxyt]) Hvw->qleft;
    qright = (real_t (*)[H.nxystep][H.nxyt]) Hvw->qright;
    sgnm = (int (*)[H.nxyt]) Hw->sgnm;
    q = (real_t (*)[H.nxystep][H.nxyt]) Hvw->q;
    dq = (real_t (*)[H.nxystep][H.nxyt]) Hvw->dq;
    u = (real_t (*)[H.nxystep][H.nxyt]) Hvw->u;
    qxm = (real_t (*)[H.nxystep][H.nxyt]) Hvw->qxm;
    qxp = (real_t (*)[H.nxystep][H.nxyt]) Hvw->qxp;

    int Hmin, Hmax, Hstep;
    int Hdimsize;
    int Hndim_1;

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

    if (!H.nstep && idim == 1) {
      /* LM -- HERE a more secure implementation should be used: a new parameter ? */
    }
    // if (H.mype == 1) fprintf(fic, "Hydro computes slices.\n");
    for (j = Hmin; j < Hmax; j += Hstep) {
      // we try to compute many slices each pass
      int jend = j + Hstep;
      if (jend >= Hmax)
        jend = Hmax;
      int slices = jend - j;    // numbre of slices to compute
      // fprintf(stderr, "Godunov idim=%d, j=%d %d \n", idim, j, slices);

      if (clear) Dmemset((H.nxyt) * H.nxystep * H.nvar, (real_t *) dq, 0);
      start = cclock();
      gatherConservativeVars(idim, j, H.imin, H.imax, H.jmin, H.jmax, H.nvar, H.nxt, H.nyt, H.nxyt, slices, Hstep, uold,
                             u);
      end = cclock();
      functim[TIM_GATCON] += ccelaps(start, end);
      if (H.prt) {fprintf(fic, "ConservativeVars %d %d %d %d %d %d\n", H.nvar, H.nxt, H.nyt, H.nxyt, slices, Hstep);}
      PRINTARRAYV2(fic, u, Hdimsize, "u", H);

      if (clear) Dmemset((H.nxyt) * H.nxystep * H.nvar, (real_t *) dq, 0);

      // Convert to primitive variables
      start = cclock();
      constoprim(Hdimsize, H.nxyt, H.nvar, H.smallr, slices, Hstep, u, q, e);
      end = cclock();
      functim[TIM_CONPRI] += ccelaps(start, end);
      PRINTARRAY(fic, e, Hdimsize, "e", H);
      PRINTARRAYV2(fic, q, Hdimsize, "q", H);

      start = cclock();
      equation_of_state(0, Hdimsize, H.nxyt, H.nvar, H.smallc, H.gamma, slices, Hstep, e, q, c);
      end = cclock();
      functim[TIM_EOS] += ccelaps(start, end);
      PRINTARRAY(fic, c, Hdimsize, "c", H);
      PRINTARRAYV2(fic, q, Hdimsize, "q", H);

      // Characteristic tracing
      if (H.iorder != 1) {
	start = cclock();
        slope(Hdimsize, H.nvar, H.nxyt, H.slope_type, slices, Hstep, q, dq);
	end = cclock();
	functim[TIM_SLOPE] += ccelaps(start, end);
        PRINTARRAYV2(fic, dq, Hdimsize, "dq", H);
      }

      if (clear) Dmemset(H.nxyt * H.nxystep * H.nvar, (real_t *) qxm, 0);
      if (clear) Dmemset(H.nxyt * H.nxystep * H.nvar, (real_t *) qxp, 0);
      if (clear) Dmemset(H.nxyt * H.nxystep * H.nvar, (real_t *) qleft, 0);
      if (clear) Dmemset(H.nxyt * H.nxystep * H.nvar, (real_t *) qright, 0);
      if (clear) Dmemset(H.nxyt * H.nxystep * H.nvar, (real_t *) flux, 0);
      if (clear) Dmemset(H.nxyt * H.nxystep * H.nvar, (real_t *) qgdnv, 0);
      start = cclock();
      trace(dtdx, Hdimsize, H.scheme, H.nvar, H.nxyt, slices, Hstep, q, dq, c, qxm, qxp);
      end = cclock();
      functim[TIM_TRACE] += ccelaps(start, end);
      PRINTARRAYV2(fic, qxm, Hdimsize, "qxm", H);
      PRINTARRAYV2(fic, qxp, Hdimsize, "qxp", H);

      start = cclock();
      qleftright(idim, H.nx, H.ny, H.nxyt, H.nvar, slices, Hstep, qxm, qxp, qleft, qright);
      end = cclock();
      functim[TIM_QLEFTR] += ccelaps(start, end);
      PRINTARRAYV2(fic, qleft, Hdimsize, "qleft", H);
      PRINTARRAYV2(fic, qright, Hdimsize, "qright", H);

      start = cclock();
      riemann(Hndim_1, H.smallr, H.smallc, H.gamma, H.niter_riemann, H.nvar, H.nxyt, slices, Hstep, qleft, qright, qgdnv, sgnm, Hw);
      end = cclock();
      functim[TIM_RIEMAN] += ccelaps(start, end);
      PRINTARRAYV2(fic, qgdnv, Hdimsize, "qgdnv", H);

      start = cclock();
      cmpflx(Hdimsize, H.nxyt, H.nvar, H.gamma, slices, Hstep, qgdnv, flux);
      end = cclock();
      functim[TIM_CMPFLX] += ccelaps(start, end);
      PRINTARRAYV2(fic, flux, Hdimsize, "flux", H);
      PRINTARRAYV2(fic, u, Hdimsize, "u", H);

      start = cclock();
      updateConservativeVars(idim, j, dtdx, H.imin, H.imax, H.jmin, H.jmax, H.nvar, H.nxt, H.nyt, H.nxyt, slices, Hstep,
                             uold, u, flux);
      end = cclock();
      functim[TIM_UPDCON] += ccelaps(start, end);
      PRINTUOLD(fic, H, Hv);
    }                           // for j

    if (H.prt) {
      fprintf(fic, "[%d] After pass %d\n", H.mype, idim);
      PRINTUOLD(fic, H, Hv);
    }
  }

  if ((H.t + dt >= H.tend) || (H.nstep + 1 >= H.nstepmax)) {
    /* LM -- HERE a more secure implementation should be used: a new parameter ? */
  }

}                               // hydro_godunov


// EOF
