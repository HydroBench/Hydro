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
#include <assert.h>
#include <hip/hip_runtime.h>

extern "C" {
#include "parametres.h"
#include "hydro_funcs.h"
#include "utils.h"
#include "make_boundary.h"

#include "cmpflx.h"
#include "conservar.h"
#include "equation_of_state.h"
#include "qleftright.h"
#include "constoprim.h"
#include "riemann.h"
#include "trace.h"
#include "slope.h"
};

#include "cuHydroGodunov.h"
#include "cuConservar.h"
#include "cuConstoprim.h"
#include "cuSlope.h"
#include "cuTrace.h"
#include "cuEquationOfState.h"
#include "cuQleftright.h"
#include "cuRiemann.h"
#include "cuCmpflx.h"
#include "cuMakeBoundary.h"
#include "gridfuncs.h"

#define VERIF(x, ou) if ((x) != hipSuccess)  { CheckErr((ou)); }

// Variables DEVice 
double *uoldDEV = 0, *uDEV = 0, *eDEV = 0, *qDEV = 0, *dqDEV = 0, *cDEV = 0, *qxpDEV = 0, *qxmDEV = 0;
double *qleftDEV = 0, *qrightDEV = 0, *qgdnvDEV = 0, *fluxDEV = 0;
long *sgnmDEV = 0;

// double *rlDEV = 0, *ulDEV = 0, *plDEV = 0, *clDEV = 0, *wlDEV = 0, *rrDEV = 0, *urDEV = 0;
// double *prDEV = 0, *crDEV = 0, *wrDEV = 0, *roDEV = 0, *uoDEV = 0, *poDEV = 0, *coDEV = 0, *woDEV = 0;
// double *rstarDEV = 0, *ustarDEV = 0, *pstarDEV = 0, *cstarDEV = 0;
// double *spinDEV = 0, *spoutDEV = 0, *ushockDEV = 0, *fracDEV = 0, *scrDEV = 0, *delpDEV = 0, *poldDEV = 0;
// long *indDEV = 0, *ind2DEV = 0;

void
cuGetUoldQECDevicePtr(double **uoldDEV_p, double **qDEV_p, double **eDEV_p, double **cDEV_p) {
  *uoldDEV_p = uoldDEV;
  *qDEV_p = qDEV;
  *eDEV_p = eDEV;
  *cDEV_p = cDEV;
}

extern "C" void
cuPutUoldOnDevice(const hydroparam_t H, hydrovar_t * Hv) {
  hipError_t status;
  status = hipMemcpy(uoldDEV, Hv->uold, H.arUoldSz * sizeof(double), hipMemcpyHostToDevice);
  VERIF(status, "cmcpy H2D uoldDEV");
}

extern "C" void
cuGetUoldFromDevice(const hydroparam_t H, hydrovar_t * Hv) {
  hipError_t status;
  status = hipMemcpy(Hv->uold, uoldDEV, H.arUoldSz * sizeof(double), hipMemcpyDeviceToHost);
  VERIF(status, "cmcpy D2H uoldDEV");
}

extern "C" void
cuAllocOnDevice(const hydroparam_t H) {
  hipError_t status;
  status = hipMalloc((void **) &uoldDEV, H.arUoldSz * sizeof(double));
  VERIF(status, "malloc uoldDEV");
  status = hipMalloc((void **) &uDEV, H.nxystep * H.arVarSz * sizeof(double));
  VERIF(status, "malloc uDEV");
  status = hipMalloc((void **) &qDEV, H.nxystep * H.arVarSz * sizeof(double));
  VERIF(status, "malloc qDEV");
  status = hipMalloc((void **) &dqDEV, H.nxystep * H.arVarSz * sizeof(double));
  VERIF(status, "malloc dqDEV");
  status = hipMalloc((void **) &qxpDEV, H.nxystep * H.arVarSz * sizeof(double));
  VERIF(status, "malloc qxpDEV");
  status = hipMalloc((void **) &qleftDEV, H.nxystep * H.arVarSz * sizeof(double));
  VERIF(status, "malloc qleftDEV");
  status = hipMalloc((void **) &qrightDEV, H.nxystep * H.arVarSz * sizeof(double));
  VERIF(status, "malloc qrightDEV");
  status = hipMalloc((void **) &qxmDEV, H.nxystep * H.arVarSz * sizeof(double));
  VERIF(status, "malloc qxmDEV");
  status = hipMalloc((void **) &qgdnvDEV, H.nxystep * H.arVarSz * sizeof(double));
  VERIF(status, "malloc qgdnvDEV");
  status = hipMalloc((void **) &fluxDEV, H.nxystep * H.arVarSz * sizeof(double));
  VERIF(status, "malloc fluxDEV");
  status = hipMalloc((void **) &eDEV, H.nxystep * H.arSz * sizeof(double));
  VERIF(status, "malloc eDEV");
  status = hipMalloc((void **) &cDEV, H.nxystep * H.arSz * sizeof(double));
  VERIF(status, "malloc cDEV");
  status = hipMalloc((void **) &sgnmDEV, H.nxystep * H.arSz * sizeof(long));
  VERIF(status, "malloc sgnmDEV");

//   status = hipMalloc((void **) &rlDEV, H.nxystep * H.arSz * sizeof(double));
//   VERIF(status, "malloc rlDEV");
//   status = hipMalloc((void **) &ulDEV, H.nxystep * H.arSz * sizeof(double));
//   VERIF(status, "malloc ulDEV");
//   status = hipMalloc((void **) &plDEV, H.nxystep * H.arSz * sizeof(double));
//   VERIF(status, "malloc plDEV");
//   status = hipMalloc((void **) &clDEV, H.nxystep * H.arSz * sizeof(double));
//   VERIF(status, "malloc clDEV");
//   status = hipMalloc((void **) &wlDEV, H.nxystep * H.arSz * sizeof(double));
//   VERIF(status, "malloc wlDEV");

//   status = hipMalloc((void **) &rrDEV, H.nxystep * H.arSz * sizeof(double));
//   VERIF(status, "malloc rrDEV");
//   status = hipMalloc((void **) &urDEV, H.nxystep * H.arSz * sizeof(double));
//   VERIF(status, "malloc urDEV");
//   status = hipMalloc((void **) &prDEV, H.nxystep * H.arSz * sizeof(double));
//   VERIF(status, "malloc prDEV");
//   status = hipMalloc((void **) &crDEV, H.nxystep * H.arSz * sizeof(double));
//   VERIF(status, "malloc crDEV");
//   status = hipMalloc((void **) &wrDEV, H.nxystep * H.arSz * sizeof(double));
//   VERIF(status, "malloc wrDEV");

//   status = hipMalloc((void **) &roDEV, H.nxystep * H.arSz * sizeof(double));
//   VERIF(status, "malloc roDEV");
//   status = hipMalloc((void **) &uoDEV, H.nxystep * H.arSz * sizeof(double));
//   VERIF(status, "malloc uoDEV");
//   status = hipMalloc((void **) &poDEV, H.nxystep * H.arSz * sizeof(double));
//   VERIF(status, "malloc poDEV");
//   status = hipMalloc((void **) &coDEV, H.nxystep * H.arSz * sizeof(double));
//   VERIF(status, "malloc coDEV");
//   status = hipMalloc((void **) &woDEV, H.nxystep * H.arSz * sizeof(double));
//   VERIF(status, "malloc woDEV");

//   status = hipMalloc((void **) &rstarDEV, H.nxystep * H.arSz * sizeof(double));
//   VERIF(status, "malloc rstarDEV");
//   status = hipMalloc((void **) &ustarDEV, H.nxystep * H.arSz * sizeof(double));
//   VERIF(status, "malloc ustarDEV");
//   status = hipMalloc((void **) &pstarDEV, H.nxystep * H.arSz * sizeof(double));
//   VERIF(status, "malloc pstarDEV");
//   status = hipMalloc((void **) &cstarDEV, H.nxystep * H.arSz * sizeof(double));
//   VERIF(status, "malloc cstarDEV");

//   status = hipMalloc((void **) &spinDEV, H.nxystep * H.arSz * sizeof(double));
//   VERIF(status, "malloc spinDEV");
//   status = hipMalloc((void **) &spoutDEV, H.nxystep * H.arSz * sizeof(double));
//   VERIF(status, "malloc spoutDEV");

//   status = hipMalloc((void **) &ushockDEV, H.nxystep * H.arSz * sizeof(double));
//   VERIF(status, "malloc ushockDEV");
//   status = hipMalloc((void **) &fracDEV, H.nxystep * H.arSz * sizeof(double));
//   VERIF(status, "malloc fracDEV");
//   status = hipMalloc((void **) &scrDEV, H.nxystep * H.arSz * sizeof(double));
//   VERIF(status, "malloc scrDEV");
//   status = hipMalloc((void **) &delpDEV, H.nxystep * H.arSz * sizeof(double));
//   VERIF(status, "malloc delpDEV");
//   status = hipMalloc((void **) &poldDEV, H.nxystep * H.arSz * sizeof(double));
//   VERIF(status, "malloc poldDEV");

//   status = hipMalloc((void **) &indDEV, H.nxystep * H.arSz * sizeof(long));
//   VERIF(status, "malloc indDEV");
//   status = hipMalloc((void **) &ind2DEV, H.nxystep * H.arSz * sizeof(long));
//   VERIF(status, "malloc ind2DEV");
}

extern "C" void
cuFreeOnDevice() {
  hipError_t status;
  // liberation de la memoire sur le device (en attendant de la remonter dans le main).
  status = hipFree(uoldDEV);
  VERIF(status, "free uoldDEV");
  status = hipFree(uDEV);
  VERIF(status, "free uDEV");
  status = hipFree(qDEV);
  VERIF(status, "free qDEV");
  status = hipFree(dqDEV);
  VERIF(status, "free dqDEV");
  status = hipFree(qxpDEV);
  VERIF(status, "free qxpDEV");
  status = hipFree(qxmDEV);
  VERIF(status, "free qxmDEV");
  status = hipFree(eDEV);
  VERIF(status, "free eDEV");
  status = hipFree(cDEV);
  VERIF(status, "free cDEV");
  status = hipFree(qleftDEV);
  VERIF(status, "free qleftDEV");
  status = hipFree(qrightDEV);
  VERIF(status, "free qrightDEV");
  status = hipFree(qgdnvDEV);
  VERIF(status, "free qgndvDEV");
  status = hipFree(fluxDEV);
  VERIF(status, "free fluxDEV");
  status = hipFree(sgnmDEV);
  VERIF(status, "free sgnmDEV");

//   status = hipFree(rlDEV);
//   VERIF(status, "free rlDEV");
//   status = hipFree(ulDEV);
//   VERIF(status, "free ulDEV");
//   status = hipFree(plDEV);
//   VERIF(status, "free plDEV");
//   status = hipFree(clDEV);
//   VERIF(status, "free clDEV");
//   status = hipFree(wlDEV);
//   VERIF(status, "free wlDEV");

//   status = hipFree(rrDEV);
//   VERIF(status, "free rrDEV");
//   status = hipFree(urDEV);
//   VERIF(status, "free urDEV");
//   status = hipFree(prDEV);
//   VERIF(status, "free prDEV");
//   status = hipFree(crDEV);
//   VERIF(status, "free crDEV");
//   status = hipFree(wrDEV);
//   VERIF(status, "free wrDEV");

//   status = hipFree(roDEV);
//   VERIF(status, "free roDEV");
//   status = hipFree(uoDEV);
//   VERIF(status, "free uoDEV");
//   status = hipFree(poDEV);
//   VERIF(status, "free poDEV");
//   status = hipFree(coDEV);
//   VERIF(status, "free coDEV");
//   status = hipFree(woDEV);
//   VERIF(status, "free woDEV");

//   status = hipFree(rstarDEV);
//   VERIF(status, "free rstarDEV");
//   status = hipFree(ustarDEV);
//   VERIF(status, "free ustarDEV");
//   status = hipFree(pstarDEV);
//   VERIF(status, "free pstarDEV");
//   status = hipFree(cstarDEV);
//   VERIF(status, "free cstarDEV");

//   status = hipFree(spinDEV);
//   VERIF(status, "free spinDEV");
//   status = hipFree(spoutDEV);
//   VERIF(status, "free spoutDEV");

//   status = hipFree(ushockDEV);
//   VERIF(status, "free ushockDEV");
//   status = hipFree(fracDEV);
//   VERIF(status, "free fracDEV");
//   status = hipFree(scrDEV);
//   VERIF(status, "free scrDEV");
//   status = hipFree(delpDEV);
//   VERIF(status, "free delpDEV");
//   status = hipFree(poldDEV);
//   VERIF(status, "free poldDEV");

//   status = hipFree(indDEV);
//   VERIF(status, "free indDEV");
//   status = hipFree(ind2DEV);
//   VERIF(status, "free ind2DEV");
}

#define GETARRV(vdev, v) do { status = hipMemcpy((v), (vdev), Hstep * H.nxyt * H.nvar * sizeof(double), hipMemcpyDeviceToHost); VERIF(status, "hipMemcpy");} while(0);
#define GETARR(vdev, v) do { status = hipMemcpy((v), (vdev), Hstep * H.nxyt * sizeof(double), hipMemcpyDeviceToHost); VERIF(status, "hipMemcpy");} while(0);


void
Dmemset(double *dq, double motif, size_t nbr) {
  long i;
  for (i = 0; i < nbr; i++) {
    dq[i] = motif;
  }
}

void
cuHydroGodunov(long idimStart, double dt, const hydroparam_t H, hydrovar_t * Hv, hydrowork_t * Hw, hydrovarwork_t * Hvw) {

  // Local variables
  long i;
  double dtdx;
  int slices, iend;
  int Hmin, Hmax, Hstep, Hnxystep;
  int Hdimsize;
  int Hndim_1;
  int idimIndex = 0;
  int clear = 0;
  static FILE *fic = NULL;

  if (fic == NULL && H.prt) {
    char logname[256];
    sprintf(logname, "TRACE.%04d_%04d.txt", H.nproc, H.mype);
    fic = fopen(logname, "w");
  }

  hipError_t status;

  WHERE("hydro_godunov");

  for (idimIndex = 0; idimIndex < 2; idimIndex++) {
    int idim = (idimStart - 1 + idimIndex) % 2 + 1;
    // constant
    dtdx = dt / H.dx;

    // Update boundary conditions
    if (H.prt) {
      fprintf(fic, "godunov %d\n", idim);
      PRINTUOLD(fic, H, Hv);
    }
#define GETUOLD cuGetUoldFromDevice(H, Hv)
    cuMakeBoundary(idim, H, Hv, uoldDEV);
    if (H.prt) {fprintf(fic, "MakeBoundary\n");}
    if (H.prt) {GETUOLD; PRINTUOLD(fic, H, Hv);}

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
    Hnxystep = Hstep;
    for (i = Hmin; i < Hmax; i += Hstep) {
      int Hnxyt = H.nxyt;
      iend = i + Hstep;
      if (iend >= Hmax)
	iend = Hmax;
      slices = iend - i;

      if (clear) status = hipMemset(uDEV, 0, Hstep * H.arVarSz * sizeof(double));
      cuGatherConservativeVars(idim, i, H.imin, H.imax, H.jmin, H.jmax, H.nvar, H.nxt, H.nyt, H.nxyt, slices, Hstep, uoldDEV, uDEV);
      
      if (H.prt) {fprintf(fic, "ConservativeVars %d %d %d %d %d %d\n", H.nvar, H.nxt, H.nyt, H.nxyt, slices, Hstep);}
      if (H.prt) { GETARRV(uDEV, Hvw->u); }
      PRINTARRAYV2(fic, Hvw->u, Hdimsize, "u", H);
      // Convert to primitive variables

      cuConstoprim(Hdimsize, H.nxyt, H.nvar, H.smallr, slices, Hstep, uDEV, qDEV, eDEV);
      if (H.prt) { GETARR (eDEV, Hw->e); }
      if (H.prt) { GETARRV(qDEV, Hvw->q); }
      PRINTARRAY(fic, Hw->e, Hdimsize, "e", H);
      PRINTARRAYV2(fic, Hvw->q, Hdimsize, "q", H);

//       double *qIDDEV = &qDEV[IHvw(0, ID)];
//       double *qIPDEV = &qDEV[IHvw(0, IP)];
      double *qIDDEV = &qDEV[IHVWS(0, 0, ID)];
      double *qIPDEV = &qDEV[IHVWS(0, 0, IP)];
      cuEquationOfState(0, Hdimsize, H.smallc, H.gamma, H.nxyt, slices, qIDDEV, eDEV, qIPDEV, cDEV);
      if (H.prt) { GETARR (cDEV, Hw->c); }
      PRINTARRAY(fic, Hw->c, Hdimsize, "c", H);
      if (H.prt) { GETARRV (qDEV, Hvw->q); }
      PRINTARRAYV2(fic, Hvw->q, Hdimsize, "q", H);

      if (clear) status = hipMemset(dqDEV, 0,  Hstep * H.arVarSz * sizeof(double));
      if (clear) VERIF(status, "cmset dq");
      if (clear) status = hipMemset(qxmDEV, 0,  Hstep * H.arVarSz * sizeof(double));
      if (clear) VERIF(status, "cmset dq");
      if (clear) status = hipMemset(qxpDEV, 0,  Hstep * H.arVarSz * sizeof(double));
      if (clear) VERIF(status, "cmset dq");
      if (clear) status = hipMemset(qleftDEV, 0,  Hstep * H.arVarSz * sizeof(double));
      if (clear) VERIF(status, "cmset dq");
      if (clear) status = hipMemset(qrightDEV, 0,  Hstep * H.arVarSz * sizeof(double));
      if (clear) VERIF(status, "cmset dq");
      if (clear) status = hipMemset(fluxDEV, 0,  Hstep * H.arVarSz * sizeof(double));
      if (clear) VERIF(status, "cmset dq");
      if (clear) status = hipMemset(qgdnvDEV, 0,  Hstep * H.arVarSz * sizeof(double));
      if (clear) VERIF(status, "cmset dq");

      // Characteristic tracing
      if (H.iorder != 1) {
	cuSlope(Hdimsize, H.nvar, H.nxyt, H.slope_type, slices, Hstep, qDEV, dqDEV);
	if (H.prt) { GETARRV(dqDEV, Hvw->dq); }
	PRINTARRAYV2(fic, Hvw->dq, Hdimsize, "dq", H);
      }
      cuTrace(dtdx, Hdimsize, H.scheme, H.nvar, H.nxyt, slices, Hstep, qDEV, dqDEV, cDEV, qxmDEV, qxpDEV);
      if (H.prt) { GETARRV(qxmDEV, Hvw->qxm); }
      if (H.prt) { GETARRV(qxpDEV, Hvw->qxp); }
      PRINTARRAYV2(fic, Hvw->qxm, Hdimsize, "qxm", H);
      PRINTARRAYV2(fic, Hvw->qxp, Hdimsize, "qxp", H);
      cuQleftright(idim, H.nx, H.ny, H.nxyt, H.nvar, slices, Hstep, qxmDEV, qxpDEV, qleftDEV, qrightDEV);
      if (H.prt) { GETARRV(qleftDEV, Hvw->qleft); }
      if (H.prt) { GETARRV(qrightDEV, Hvw->qright); }
      PRINTARRAYV2(fic, Hvw->qleft, Hdimsize, "qleft", H);
      PRINTARRAYV2(fic, Hvw->qright, Hdimsize, "qright", H);

      // Solve Riemann problem at interfaces
      cuRiemann(Hndim_1, H.smallr, H.smallc, H.gamma, H.niter_riemann, H.nvar, H.nxyt, slices, Hstep, //
		qleftDEV, qrightDEV, qgdnvDEV, sgnmDEV //
// 		, rlDEV, ulDEV, plDEV, clDEV, wlDEV,
// 		rrDEV, urDEV, prDEV, crDEV, wrDEV,
// 		roDEV, uoDEV, poDEV, coDEV, woDEV,
// 		rstarDEV, ustarDEV, pstarDEV, cstarDEV,
// 		spinDEV, spoutDEV, ushockDEV, fracDEV, scrDEV, delpDEV, poldDEV, indDEV, ind2DEV
		);
      if (H.prt) { GETARRV(qgdnvDEV, Hvw->qgdnv); }
      PRINTARRAYV2(fic, Hvw->qgdnv, Hdimsize, "qgdnv", H);
      // Compute fluxes
      if (clear) status = hipMemset(fluxDEV, 0, Hstep * H.arVarSz * sizeof(double));
      cuCmpflx(Hdimsize, H.nxyt, H.nvar, H.gamma, slices, Hnxystep, qgdnvDEV, fluxDEV);
      if (H.prt) { GETARRV(fluxDEV, Hvw->flux); }
      PRINTARRAYV2(fic, Hvw->flux, Hdimsize, "flux", H);
      if (H.prt) { GETARRV(uDEV, Hvw->u); }
      PRINTARRAYV2(fic, Hvw->u, Hdimsize, "u", H);
      
      cuUpdateConservativeVars(idim, i, dtdx,
			       H.imin, H.imax, H.jmin, H.jmax, H.nvar, H.nxt, H.nyt, H.nxyt,
			       slices, Hnxystep, uoldDEV, uDEV, fluxDEV);
      if (H.prt) {
	GETUOLD; PRINTUOLD(fic, H, Hv);
      }
    }                           // for j

    if (H.prt) {
      // printf("[%d] After pass %d\n", H.mype, idim);
      GETUOLD; PRINTUOLD(fic, H, Hv);
    }
  }
}                               // hydro_godunov
