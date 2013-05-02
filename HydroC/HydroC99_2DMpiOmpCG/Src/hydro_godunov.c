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
#include <omp.h>

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
      fprintf(fic, "godunov %d\n", idim);
      PRINTUOLD(fic, H, Hv);
    }
    // if (H.mype == 1) fprintf(fic, "Hydro makes boundary.\n");
    start = cclock();
    // fprintf(stderr, "Hydro makes boundary %d.\n", myThread);
    make_boundary(idim, H, Hv);
    // fprintf(stderr, "Hydro barrier %d.\n", myThread);

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

#pragma omp parallel for private(j) shared(uold, u) 
    // schedule(static, 1) 
    for (j = Hmin; j < Hmax; j++) {
      int myThread = omp_get_thread_num();
      int slice = myThread;
      // fprintf(stderr, "Godunov slice=%d\n", slice);
      // fprintf(stderr, "Godunov idim=%d, j=%d %d \n", idim, j, slice);

      gatherConservativeVars(idim, j, H.imin, H.imax, H.jmin, H.jmax, H.nvar, H.nxt, H.nyt, H.nxyt, slice, Hstep, uold, u); // done

      // Convert to primitive variables
      constoprim(Hdimsize, H.nxyt, H.nvar, H.smallr, slice, Hstep, u, q, e); // done

      equation_of_state(0, Hdimsize, H.nxyt, H.nvar, H.smallc, H.gamma, slice, Hstep, e, q, c);

      // Characteristic tracing
      if (H.iorder != 1) {
        slope(Hdimsize, H.nvar, H.nxyt, H.slope_type, slice, Hstep, q, dq); // done
      }

      trace(dtdx, Hdimsize, H.scheme, H.nvar, H.nxyt, slice, Hstep, q, dq, c, qxm, qxp); // done

      qleftright(idim, H.nx, H.ny, H.nxyt, H.nvar, slice, Hstep, qxm, qxp, qleft, qright); // done

      riemann(Hndim_1, H.smallr, H.smallc, H.gamma, H.niter_riemann, H.nvar, H.nxyt, slice, Hstep, qleft, qright, qgdnv, sgnm, Hw); // done

      cmpflx(Hdimsize, H.nxyt, H.nvar, H.gamma, slice, Hstep, qgdnv, flux); // done

      updateConservativeVars(idim, j, dtdx, H.imin, H.imax, H.jmin, H.jmax, H.nvar, H.nxt, H.nyt, H.nxyt, slice, Hstep, uold, u, flux); // done
    }                           // for j
  }
  // fprintf(stderr, "Godunov end myThread=%d\n", myThread);
}                               // hydro_godunov


// EOF
