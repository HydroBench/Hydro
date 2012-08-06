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
#include <CL/cl.h>

#include "parametres.h"
#include "hydro_funcs.h"
#include "utils.h"

#include "oclInit.h"
#include "ocltools.h"
#include "oclHydroGodunov.h"
#include "oclConservar.h"
#include "oclConstoprim.h"
#include "oclSlope.h"
#include "oclTrace.h"
#include "oclEquationOfState.h"
#include "oclQleftright.h"
#include "oclRiemann.h"
#include "oclCmpflx.h"
#include "oclMakeBoundary.h"

// Variables DEVice 
cl_mem uoldDEV = 0, uDEV = 0, eDEV = 0, qDEV = 0, dqDEV = 0, cDEV = 0, qxpDEV = 0, qxmDEV = 0;
cl_mem qleftDEV = 0, qrightDEV = 0, qgdnvDEV = 0, fluxDEV = 0;
cl_mem rlDEV = 0, ulDEV = 0, plDEV = 0, clDEV = 0, wlDEV = 0, rrDEV = 0, urDEV = 0;
cl_mem prDEV = 0, crDEV = 0, wrDEV = 0, roDEV = 0, uoDEV = 0, poDEV = 0, coDEV = 0, woDEV = 0;
cl_mem rstarDEV = 0, ustarDEV = 0, pstarDEV = 0, cstarDEV = 0;
cl_mem sgnmDEV = 0;
cl_mem spinDEV = 0, spoutDEV = 0, ushockDEV = 0, fracDEV = 0, scrDEV = 0, delpDEV = 0, poldDEV = 0;
cl_mem indDEV = 0, ind2DEV = 0;

void ClearArray(cl_mem array, size_t lg)
{
  int lzero = 0;
  OCLSETARG03(ker[KernelMemset], array, lzero, lg);
  oclLaunchKernel(ker[KernelMemset], cqueue, lg, THREADSSZ);
}

void
oclGetUoldQECDevicePtr(cl_mem * uoldDEV_p, cl_mem * qDEV_p, cl_mem * eDEV_p, cl_mem * cDEV_p)
{
  *uoldDEV_p = uoldDEV;
  *qDEV_p = qDEV;
  *eDEV_p = eDEV;
  *cDEV_p = cDEV;
}

void
oclPutUoldOnDevice(const hydroparam_t H, hydrovar_t * Hv)
{
  //   cudaError_t status;
  //   status = cudaMemcpy(uoldDEV, Hv->uold, H.arUoldSz * sizeof(double), cudaMemcpyHostToDevice);
  //   VERIF(status, "cmcpy H2D uoldDEV");
  cl_int err = 0;
  cl_event event;
  err = clEnqueueWriteBuffer(cqueue, uoldDEV, CL_TRUE, 0, H.arUoldSz * sizeof(double), Hv->uold, 0, NULL, &event);
  oclCheckErr(err, "clEnqueueWriteBuffer");
  err = clWaitForEvents(1, &event);
  oclCheckErr(err, "clWaitForEvents");
  err = clReleaseEvent(event);
  oclCheckErr(err, "clReleaseEvent");
}

void
oclGetUoldFromDevice(const hydroparam_t H, hydrovar_t * Hv)
{
  //   cudaError_t status;
  //   status = cudaMemcpy(Hv->uold, uoldDEV, H.arUoldSz * sizeof(double), cudaMemcpyDeviceToHost);
  //   VERIF(status, "cmcpy D2H uoldDEV");
  cl_int err = 0;
  cl_event event;
  err = clEnqueueReadBuffer(cqueue, uoldDEV, CL_TRUE, 0, H.arUoldSz * sizeof(double), Hv->uold, 0, NULL, &event);
  oclCheckErr(err, "clEnqueueWriteBuffer");
  err = clWaitForEvents(1, &event);
  oclCheckErr(err, "clWaitForEvents");
  err = clReleaseEvent(event);
  oclCheckErr(err, "clReleaseEvent");
}

void
oclAllocOnDevice(const hydroparam_t H)
{
  cl_int status = 0;

  size_t lVarSz = H.arVarSz * H.nxystep * sizeof(double);
  size_t lSz = H.arSz * H.nxystep * sizeof(double);
  size_t lSzL = H.arSz * H.nxystep * sizeof(long);

  uoldDEV = clCreateBuffer(ctx, CL_MEM_READ_WRITE, H.arUoldSz * sizeof(double), NULL, &status);
  oclCheckErr(status, "");
  uDEV = clCreateBuffer(ctx, CL_MEM_READ_WRITE, lVarSz, NULL, &status);
  oclCheckErr(status, "");
  qDEV = clCreateBuffer(ctx, CL_MEM_READ_WRITE, lVarSz, NULL, &status);
  oclCheckErr(status, "");
  dqDEV = clCreateBuffer(ctx, CL_MEM_READ_WRITE, lVarSz, NULL, &status);
  oclCheckErr(status, "");
  qxpDEV = clCreateBuffer(ctx, CL_MEM_READ_WRITE, lVarSz, NULL, &status);
  oclCheckErr(status, "");
  qleftDEV = clCreateBuffer(ctx, CL_MEM_READ_WRITE, lVarSz, NULL, &status);
  oclCheckErr(status, "");
  qrightDEV = clCreateBuffer(ctx, CL_MEM_READ_WRITE, lVarSz, NULL, &status);
  oclCheckErr(status, "");
  qxmDEV = clCreateBuffer(ctx, CL_MEM_READ_WRITE, lVarSz, NULL, &status);
  oclCheckErr(status, "");
  qgdnvDEV = clCreateBuffer(ctx, CL_MEM_READ_WRITE, lVarSz, NULL, &status);
  oclCheckErr(status, "");
  fluxDEV = clCreateBuffer(ctx, CL_MEM_READ_WRITE, lVarSz, NULL, &status);
  oclCheckErr(status, "");


  eDEV = clCreateBuffer(ctx, CL_MEM_READ_WRITE, lSz, NULL, &status);
  oclCheckErr(status, "");
  cDEV = clCreateBuffer(ctx, CL_MEM_READ_WRITE, lSz, NULL, &status);
  oclCheckErr(status, "");

  rlDEV = clCreateBuffer(ctx, CL_MEM_READ_WRITE, lSz, NULL, &status);
  oclCheckErr(status, "");
  ulDEV = clCreateBuffer(ctx, CL_MEM_READ_WRITE, lSz, NULL, &status);
  oclCheckErr(status, "");
  plDEV = clCreateBuffer(ctx, CL_MEM_READ_WRITE, lSz, NULL, &status);
  oclCheckErr(status, "");
  clDEV = clCreateBuffer(ctx, CL_MEM_READ_WRITE, lSz, NULL, &status);
  oclCheckErr(status, "");
  wlDEV = clCreateBuffer(ctx, CL_MEM_READ_WRITE, lSz, NULL, &status);
  oclCheckErr(status, "");

  rrDEV = clCreateBuffer(ctx, CL_MEM_READ_WRITE, lSz, NULL, &status);
  oclCheckErr(status, "");
  urDEV = clCreateBuffer(ctx, CL_MEM_READ_WRITE, lSz, NULL, &status);
  oclCheckErr(status, "");
  prDEV = clCreateBuffer(ctx, CL_MEM_READ_WRITE, lSz, NULL, &status);
  oclCheckErr(status, "");
  crDEV = clCreateBuffer(ctx, CL_MEM_READ_WRITE, lSz, NULL, &status);
  oclCheckErr(status, "");
  wrDEV = clCreateBuffer(ctx, CL_MEM_READ_WRITE, lSz, NULL, &status);
  oclCheckErr(status, "");

  roDEV = clCreateBuffer(ctx, CL_MEM_READ_WRITE, lSz, NULL, &status);
  oclCheckErr(status, "");
  uoDEV = clCreateBuffer(ctx, CL_MEM_READ_WRITE, lSz, NULL, &status);
  oclCheckErr(status, "");
  poDEV = clCreateBuffer(ctx, CL_MEM_READ_WRITE, lSz, NULL, &status);
  oclCheckErr(status, "");
  coDEV = clCreateBuffer(ctx, CL_MEM_READ_WRITE, lSz, NULL, &status);
  oclCheckErr(status, "");
  woDEV = clCreateBuffer(ctx, CL_MEM_READ_WRITE, lSz, NULL, &status);
  oclCheckErr(status, "");

  rstarDEV = clCreateBuffer(ctx, CL_MEM_READ_WRITE, lSz, NULL, &status);
  oclCheckErr(status, "");
  ustarDEV = clCreateBuffer(ctx, CL_MEM_READ_WRITE, lSz, NULL, &status);
  oclCheckErr(status, "");
  pstarDEV = clCreateBuffer(ctx, CL_MEM_READ_WRITE, lSz, NULL, &status);
  oclCheckErr(status, "");
  cstarDEV = clCreateBuffer(ctx, CL_MEM_READ_WRITE, lSz, NULL, &status);
  oclCheckErr(status, "");

  spinDEV = clCreateBuffer(ctx, CL_MEM_READ_WRITE, lSz, NULL, &status);
  oclCheckErr(status, "");
  spoutDEV = clCreateBuffer(ctx, CL_MEM_READ_WRITE, lSz, NULL, &status);
  oclCheckErr(status, "");

  ushockDEV = clCreateBuffer(ctx, CL_MEM_READ_WRITE, lSz, NULL, &status);
  oclCheckErr(status, "");
  fracDEV = clCreateBuffer(ctx, CL_MEM_READ_WRITE, lSz, NULL, &status);
  oclCheckErr(status, "");
  scrDEV = clCreateBuffer(ctx, CL_MEM_READ_WRITE, lSz, NULL, &status);
  oclCheckErr(status, "");
  delpDEV = clCreateBuffer(ctx, CL_MEM_READ_WRITE, lSz, NULL, &status);
  oclCheckErr(status, "");
  poldDEV = clCreateBuffer(ctx, CL_MEM_READ_WRITE, lSz, NULL, &status);
  oclCheckErr(status, "");

  sgnmDEV = clCreateBuffer(ctx, CL_MEM_READ_WRITE, lSzL, NULL, &status);
  oclCheckErr(status, "");
  indDEV = clCreateBuffer(ctx, CL_MEM_READ_WRITE, lSzL, NULL, &status);
  oclCheckErr(status, "");
  ind2DEV = clCreateBuffer(ctx, CL_MEM_READ_WRITE, lSzL, NULL, &status);
  oclCheckErr(status, "");

  ClearArray(qleftDEV, lVarSz);
  ClearArray(qrightDEV, lVarSz);
  ClearArray(fluxDEV, lVarSz);
  ClearArray(qxmDEV, lVarSz);
  ClearArray(qxpDEV, lVarSz);
  ClearArray(qgdnvDEV, lVarSz);
  ClearArray(cDEV, lSz);
  ClearArray(eDEV, lSz);
}

void
oclFreeOnDevice()
{
  cl_int status = 0;
  // liberation de la memoire sur le device (en attendant de la remonter dans le main).
  status = clReleaseMemObject(uoldDEV);
  oclCheckErr(status, "");
  status = clReleaseMemObject(uDEV);
  oclCheckErr(status, "");
  status = clReleaseMemObject(qDEV);
  oclCheckErr(status, "");
  status = clReleaseMemObject(dqDEV);
  oclCheckErr(status, "");
  status = clReleaseMemObject(qxpDEV);
  oclCheckErr(status, "");
  status = clReleaseMemObject(qxmDEV);
  oclCheckErr(status, "");
  status = clReleaseMemObject(eDEV);
  oclCheckErr(status, "");
  status = clReleaseMemObject(cDEV);
  oclCheckErr(status, "");
  status = clReleaseMemObject(qleftDEV);
  oclCheckErr(status, "");
  status = clReleaseMemObject(qrightDEV);
  oclCheckErr(status, "");
  status = clReleaseMemObject(qgdnvDEV);
  oclCheckErr(status, "");
  status = clReleaseMemObject(fluxDEV);
  oclCheckErr(status, "");

  status = clReleaseMemObject(rlDEV);
  oclCheckErr(status, "");
  status = clReleaseMemObject(ulDEV);
  oclCheckErr(status, "");
  status = clReleaseMemObject(plDEV);
  oclCheckErr(status, "");
  status = clReleaseMemObject(clDEV);
  oclCheckErr(status, "");
  status = clReleaseMemObject(wlDEV);
  oclCheckErr(status, "");

  status = clReleaseMemObject(rrDEV);
  oclCheckErr(status, "");
  status = clReleaseMemObject(urDEV);
  oclCheckErr(status, "");
  status = clReleaseMemObject(prDEV);
  oclCheckErr(status, "");
  status = clReleaseMemObject(crDEV);
  oclCheckErr(status, "");
  status = clReleaseMemObject(wrDEV);
  oclCheckErr(status, "");

  status = clReleaseMemObject(roDEV);
  oclCheckErr(status, "");
  status = clReleaseMemObject(uoDEV);
  oclCheckErr(status, "");
  status = clReleaseMemObject(poDEV);
  oclCheckErr(status, "");
  status = clReleaseMemObject(coDEV);
  oclCheckErr(status, "");
  status = clReleaseMemObject(woDEV);
  oclCheckErr(status, "");

  status = clReleaseMemObject(rstarDEV);
  oclCheckErr(status, "");
  status = clReleaseMemObject(ustarDEV);
  oclCheckErr(status, "");
  status = clReleaseMemObject(pstarDEV);
  oclCheckErr(status, "");
  status = clReleaseMemObject(cstarDEV);
  oclCheckErr(status, "");

  status = clReleaseMemObject(sgnmDEV);
  oclCheckErr(status, "");
  status = clReleaseMemObject(spinDEV);
  oclCheckErr(status, "");
  status = clReleaseMemObject(spoutDEV);
  oclCheckErr(status, "");

  status = clReleaseMemObject(ushockDEV);
  oclCheckErr(status, "");
  status = clReleaseMemObject(fracDEV);
  oclCheckErr(status, "");
  status = clReleaseMemObject(scrDEV);
  oclCheckErr(status, "");
  status = clReleaseMemObject(delpDEV);
  oclCheckErr(status, "");
  status = clReleaseMemObject(poldDEV);
  oclCheckErr(status, "");

  status = clReleaseMemObject(indDEV);
  oclCheckErr(status, "");
  status = clReleaseMemObject(ind2DEV);
  oclCheckErr(status, "");
}

#define GETARRV(vdev, v) do { status = clEnqueueReadBuffer(cqueue, (vdev), CL_TRUE, 0, Hstep * H.nxyt * H.nvar * sizeof(double), (v), 0, NULL, &event); oclCheckErr(status, "");} while(0);
#define GETARR(vdev, v)  do { status = clEnqueueReadBuffer(cqueue, (vdev), CL_TRUE, 0, Hstep * H.nxyt * sizeof(double), (v), 0, NULL, &event); oclCheckErr(status, "");} while(0);

void
oclHydroGodunov(long idimStart, double dt, const hydroparam_t H, hydrovar_t * Hv, hydrowork_t * Hw, hydrovarwork_t * Hvw)
{
  cl_int status;
  cl_event event;
  // Local variables
  int i, j, idim, idimIndex;
  int Hmin, Hmax, Hstep, Hnxystep;
  int Hdimsize;
  int Hndim_1;
  int slices, iend;
  double dtdx;
  size_t lVarSz = H.arVarSz * H.nxystep * sizeof(double);
  long Hnxyt = H.nxyt;
  static FILE *fic = NULL;

  long offsetIP = IHVW(0, IP);
  long offsetID = IHVW(0, ID);

  if (fic == NULL && H.prt) {
    char logname[256];
    sprintf(logname, "TRACE.%04d_%04d.txt", H.nproc, H.mype);
    fic = fopen(logname, "w");
  }

  WHERE("hydro_godunov");

  for (idimIndex = 0; idimIndex < 2; idimIndex++) {
    idim = (idimStart - 1 + idimIndex) % 2 + 1;
    // constant
    // constant
    dtdx = dt / H.dx;

    // Update boundary conditions
    if (H.prt) {
      fprintf(fic, "godunov %d\n", idim);
      PRINTUOLD(fic, H, Hv);
    }
#define GETUOLD oclGetUoldFromDevice(H, Hv)
    oclMakeBoundary(idim, H, Hv, uoldDEV);
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
      if (iend >= Hmax) iend = Hmax;
      slices = iend - i;

      oclMemset(uDEV, 0, lVarSz);
      oclGatherConservativeVars(idim, i, H.imin, H.imax, H.jmin, H.jmax, H.nvar, H.nxt, H.nyt, H.nxyt, slices, uoldDEV, uDEV);
      if (H.prt) {fprintf(fic, "ConservativeVars %ld %ld %ld %ld %d %d\n", H.nvar, H.nxt, H.nyt, H.nxyt, slices, Hstep);}
      if (H.prt) { GETARRV(uDEV, Hvw->u); }
      PRINTARRAYV2(fic, Hvw->u, Hdimsize, "u", H);

      // Convert to primitive variables
      oclConstoprim(Hdimsize, H.nxyt, H.nvar, H.smallr, slices, uDEV, qDEV, eDEV);
      if (H.prt) { GETARR (eDEV, Hw->e); }
      if (H.prt) { GETARRV(qDEV, Hvw->q); }
      PRINTARRAY(fic, Hw->e, Hdimsize, "e", H);
      PRINTARRAYV2(fic, Hvw->q, Hdimsize, "q", H);

      oclEquationOfState(offsetIP, offsetID, 0, Hdimsize, H.smallc, H.gamma, slices, qDEV, eDEV, cDEV);
      if (H.prt) { GETARR (cDEV, Hw->c); }
      PRINTARRAY(fic, Hw->c, Hdimsize, "c", H);
      if (H.prt) { GETARRV (qDEV, Hvw->q); }
      PRINTARRAYV2(fic, Hvw->q, Hdimsize, "q", H);

      oclMemset(dqDEV, 0, H.arVarSz * H.nxystep);
      // Characteristic tracing
      if (H.iorder != 1) {
        oclSlope(Hdimsize, H.nvar, H.nxyt, H.slope_type, slices, qDEV, dqDEV);
	if (H.prt) { GETARRV(dqDEV, Hvw->dq); }
	PRINTARRAYV2(fic, Hvw->dq, Hdimsize, "dq", H);
      }
      oclTrace(dtdx, Hdimsize, H.scheme, H.nvar, H.nxyt, slices, Hstep, qDEV, dqDEV, cDEV, qxmDEV, qxpDEV);
      if (H.prt) { GETARRV(qxmDEV, Hvw->qxm); }
      if (H.prt) { GETARRV(qxpDEV, Hvw->qxp); }
      PRINTARRAYV2(fic, Hvw->qxm, Hdimsize, "qxm", H);
      PRINTARRAYV2(fic, Hvw->qxp, Hdimsize, "qxp", H);
      oclQleftright(idim, H.nx, H.ny, H.nxyt, H.nvar, slices, Hstep, qxmDEV, qxpDEV, qleftDEV, qrightDEV);
      if (H.prt) { GETARRV(qleftDEV, Hvw->qleft); }
      if (H.prt) { GETARRV(qrightDEV, Hvw->qright); }
      PRINTARRAYV2(fic, Hvw->qleft, Hdimsize, "qleft", H);
      PRINTARRAYV2(fic, Hvw->qright, Hdimsize, "qright", H);

      // Solve Riemann problem at interfaces
      oclRiemann(Hndim_1, H.smallr, H.smallc, H.gamma, H.niter_riemann, H.nvar, H.nxyt, slices, Hstep,
		 qleftDEV, qrightDEV, qgdnvDEV,
                 rlDEV, ulDEV, plDEV, clDEV, wlDEV,
                 rrDEV, urDEV, prDEV, crDEV, wrDEV,
                 roDEV, uoDEV, poDEV, coDEV, woDEV,
                 rstarDEV, ustarDEV, pstarDEV, cstarDEV,
                 sgnmDEV, spinDEV, spoutDEV, ushockDEV, fracDEV,
                 scrDEV, delpDEV, poldDEV, indDEV, ind2DEV);
      if (H.prt) { GETARRV(qgdnvDEV, Hvw->qgdnv); }
      PRINTARRAYV2(fic, Hvw->qgdnv, Hdimsize, "qgdnv", H);
      // Compute fluxes
      oclMemset(fluxDEV, 0, H.arVarSz);
      oclCmpflx(Hdimsize, H.nxyt, H.nvar, H.gamma, slices, Hnxystep, qgdnvDEV, fluxDEV);
      if (H.prt) { GETARRV(fluxDEV, Hvw->flux); }
      PRINTARRAYV2(fic, Hvw->flux, Hdimsize, "flux", H);
      if (H.prt) { GETARRV(uDEV, Hvw->u); }
      PRINTARRAYV2(fic, Hvw->u, Hdimsize, "u", H);
      oclUpdateConservativeVars(idim, i, dtdx, H.imin, H.imax, H.jmin, H.jmax, H.nvar, H.nxt, H.nyt, H.nxyt, slices, Hnxystep, 
				uoldDEV, uDEV, fluxDEV);
      if (H.prt) {
	GETUOLD; PRINTUOLD(fic, H, Hv);
      }
    }                           // for j

    if (H.prt) {
      // printf("After pass %d\n", idim);
      PRINTUOLD(fic, H, Hv);
    }
  } 
}                               // hydro_godunov
