/*
  A simple 2D hydro code
  (C) Romain Teyssier : CEA/IRFU           -- original F90 code
  (C) Pierre-Francois Lavallee : IDRIS      -- original F90 code
  (C) Guillaume Colin de Verdiere : CEA/DAM -- for the C version
  (C) Jeffrey Poznanovic : CSCS             -- for the OpenACC version
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

#ifdef CRAYPAT
#include <pat_api.h>
#endif

// variables auxiliaires pour mettre en place le mode resident de HMPP
void
hydro_godunov(int idimStart, double dt, const hydroparam_t H, hydrovar_t * Hv, hydrowork_t * Hw, hydrovarwork_t * Hvw) {
  // Local variables
  int j;
  double dtdx;
  int clear=0;

  double *restrict e;
  double *restrict flux;
  double *restrict qleft;
  double *restrict qright;
  double *restrict c;
  double *restrict uold;
  int *restrict sgnm;
  double *restrict qgdnv;
  double *restrict u;
  double *restrict qxm;
  double *restrict qxp;
  double *restrict q;
  double *restrict dq;

  static FILE *fic = NULL;

#ifdef CRAYPAT
  PAT_record(PAT_STATE_ON);
#endif

  if (fic == NULL && H.prt == 1) {
    char logname[256];
    sprintf(logname, "TRACE.%04d_%04d.txt", H.nproc, H.mype);
    fic = fopen(logname, "w");
  }

  WHERE("hydro_godunov");

  uold = Hv->uold;
#pragma acc data present(uold[0:H.nvar*H.nxt*H.nyt])
  {

  // int hmppGuard = 1;
  int idimIndex = 0;
  // Allocate work space for 1D sweeps
  allocate_work_space(H.nxyt, H, Hw, Hvw);

  qgdnv = Hvw->qgdnv;
  flux = Hvw->flux;
  c = Hw->c;
  e = Hw->e;
  qleft = Hvw->qleft;
  qright = Hvw->qright;
  sgnm = Hw->sgnm;
  q = Hvw->q;
  dq = Hvw->dq;
  u = Hvw->u;
  qxm = Hvw->qxm;
  qxp = Hvw->qxp;

#pragma acc data \
  create(qleft[0:H.nvar*H.nxystep*H.nxyt], qright[0:H.nvar*H.nxystep*H.nxyt], \
         q[0:H.nvar*H.nxystep*H.nxyt], qgdnv[0:H.nvar*H.nxystep*H.nxyt], \
         flux[0:H.nvar*H.nxystep*H.nxyt], u[0:H.nvar*H.nxystep*H.nxyt], \
         dq[0:H.nvar*H.nxystep*H.nxyt], e[0:H.nxystep*H.nxyt], c[0:H.nxystep*H.nxyt], \
         sgnm[0:H.nxystep*H.nxyt], qxm[0:H.nvar*H.nxystep*H.nxyt], qxp[0:H.nvar*H.nxystep*H.nxyt])

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
#if 0
    make_boundary(idim, H, Hv);
#else
    #pragma acc host_data use_device(uold)
    cuMakeBoundary(idim, H, Hv, uold);
#endif
    if (H.prt) {fprintf(fic, "MakeBoundary\n");}
    PRINTUOLD(fic, H, Hv);

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

    //#pragma acc update device(uold[0:H.nvar*H.nxt*H.nyt])

    for (j = Hmin; j < Hmax; j += Hstep) {
      // we try to compute many slices each pass
      int jend = j + Hstep;
      if (jend >= Hmax)
        jend = Hmax;
      int slices = jend - j;    // numbre of slices to compute
      //printf("Godunov idim=%d, j=%d %d \n", idim, j, slices);

      if (clear) Dmemset((H.nxyt) * H.nxystep * H.nvar, (double *) dq, 0);
      gatherConservativeVars(idim, j, H.imin, H.imax, H.jmin, H.jmax, H.nvar, H.nxt, H.nyt, H.nxyt, slices, Hstep, uold,
                             u);
      if (H.prt) {fprintf(fic, "ConservativeVars %d %d %d %d %d %d\n", H.nvar, H.nxt, H.nyt, H.nxyt, slices, Hstep);}
      PRINTARRAYV2(fic, u, Hdimsize, "u", H);

      if (clear) Dmemset((H.nxyt) * H.nxystep * H.nvar, (double *) dq, 0);

      // Convert to primitive variables
      constoprim(Hdimsize, H.nxyt, H.nvar, H.smallr, slices, Hstep, u, q, e);
      PRINTARRAY(fic, e, Hdimsize, "e", H);
      PRINTARRAYV2(fic, q, Hdimsize, "q", H);

      equation_of_state(0, Hdimsize, H.nxyt, H.nvar, H.smallc, H.gamma, slices, Hstep, e, q, c);
      PRINTARRAY(fic, c, Hdimsize, "c", H);
      PRINTARRAYV2(fic, q, Hdimsize, "q", H);

      // Characteristic tracing
      if (H.iorder != 1) {
        slope(Hdimsize, H.nvar, H.nxyt, H.slope_type, slices, Hstep, q, dq);
        PRINTARRAYV2(fic, dq, Hdimsize, "dq", H);
      }

      if (clear) Dmemset((H.nxyt + 2) * H.nxystep * H.nvar, (double *) qxm, 0);
      if (clear) Dmemset((H.nxyt + 2) * H.nxystep * H.nvar, (double *) qxp, 0);
      if (clear) Dmemset((H.nxyt + 2) * H.nxystep * H.nvar, (double *) qleft, 0);
      if (clear) Dmemset((H.nxyt + 2) * H.nxystep * H.nvar, (double *) qright, 0);
      if (clear) Dmemset((H.nxyt + 2) * H.nxystep * H.nvar, (double *) flux, 0);
      if (clear) Dmemset((H.nxyt + 2) * H.nxystep * H.nvar, (double *) qgdnv, 0);

      trace(dtdx, Hdimsize, H.scheme, H.nvar, H.nxyt, slices, Hstep, q, dq, c, qxm, qxp);
      PRINTARRAYV2(fic, qxm, Hdimsize, "qxm", H);
      PRINTARRAYV2(fic, qxp, Hdimsize, "qxp", H);

      qleftright(idim, H.nx, H.ny, H.nxyt, H.nvar, slices, Hstep, qxm, qxp, qleft, qright);
      PRINTARRAYV2(fic, qleft, Hdimsize, "qleft", H);
      PRINTARRAYV2(fic, qright, Hdimsize, "qright", H);

      riemann(Hndim_1, H.smallr, H.smallc, H.gamma, H.niter_riemann, H.nvar, H.nxyt, slices, Hstep, qleft, qright,
              qgdnv, sgnm);
      PRINTARRAYV2(fic, qgdnv, Hdimsize, "qgdnv", H);

      cmpflx(Hdimsize, H.nxyt, H.nvar, H.gamma, slices, Hstep, qgdnv, flux);
      PRINTARRAYV2(fic, flux, Hdimsize, "flux", H);
      PRINTARRAYV2(fic, u, Hdimsize, "u", H);

      updateConservativeVars(idim, j, dtdx, H.imin, H.imax, H.jmin, H.jmax, H.nvar, H.nxt, H.nyt, H.nxyt, slices, Hstep,
                             uold, u, flux);
      PRINTUOLD(fic, H, Hv);
    }                           // for j

    if (H.prt) {
      // printf("[%d] After pass %d\n", H.mype, idim);
      PRINTUOLD(fic, H, Hv);
    }
    //#pragma acc update host(uold[0:H.nvar*H.nxt*H.nyt])
  }
  // Deallocate work space
  deallocate_work_space(H, Hw, Hvw);

  if ((H.t + dt >= H.tend) || (H.nstep + 1 >= H.nstepmax)) {
    /* LM -- HERE a more secure implementation should be used: a new parameter ? */
  }

  }

}                               // hydro_godunov


// EOF
