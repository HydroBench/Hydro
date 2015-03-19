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
#include <CL/cl.h>

#include "parametres.h"
#include "hydro_funcs.h"
#include "utils.h"
#include "cclock.h"

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
cl_mem sgnmDEV = 0;

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
  cl_int err = 0;
  cl_event event;
  err = clEnqueueWriteBuffer(cqueue, uoldDEV, CL_TRUE, 0, H.arUoldSz * sizeof(real_t), Hv->uold, 0, NULL, &event);
  oclCheckErr(err, "clEnqueueWriteBuffer");
  err = clWaitForEvents(1, &event);
  oclCheckErr(err, "clWaitForEvents");
  err = clReleaseEvent(event);
  oclCheckErr(err, "clReleaseEvent");
}

void
oclGetUoldFromDevice(const hydroparam_t H, hydrovar_t * Hv)
{
  cl_int err = 0;
  cl_event event;
  err = clEnqueueReadBuffer(cqueue, uoldDEV, CL_TRUE, 0, H.arUoldSz * sizeof(real_t), Hv->uold, 0, NULL, &event);
  oclCheckErr(err, "clEnqueueWriteBuffer");
  err = clWaitForEvents(1, &event);
  oclCheckErr(err, "clWaitForEvents");
  err = clReleaseEvent(event);
  oclCheckErr(err, "clReleaseEvent");
}

static void ClearArrayDble(cl_mem array, long lgr)
{
  real_t lzero = 0.;
  assert(array != NULL);
  assert(lgr > 0);
  OCLSETARG03(ker[KernelMemset], array, lzero, lgr);
  oclLaunchKernel(ker[KernelMemset], cqueue, lgr, THREADSSZ, __FILE__, __LINE__);
}

cl_mem  AllocClear(size_t lgrBytes) {
  cl_mem tab;
  cl_int status;

  tab = clCreateBuffer(ctx, CL_MEM_READ_WRITE, lgrBytes, NULL, &status); 
  oclCheckErr(status, "AllocClear");
  ClearArrayDble(tab, lgrBytes / sizeof(real_t));
  return tab;
}

cl_mem  AllocClearL(size_t lgrBytes) {
  cl_mem tab;
  cl_int status;

  tab = clCreateBuffer(ctx, CL_MEM_READ_WRITE, lgrBytes, NULL, &status); 
  oclCheckErr(status, "AllocClearL");
  // ClearArrayDble(tab, lgrBytes / sizeof(real_t));
  return tab;
}

void
oclAllocOnDevice(const hydroparam_t H)
{
  cl_int status = 0;
  size_t lVarSz = H.arVarSz * H.nxystep * sizeof(real_t);
  size_t lUold = H.arUoldSz * sizeof(real_t);
  size_t lSz = H.arSz * H.nxystep * sizeof(real_t);
  size_t lSzL = H.arSz * H.nxystep * sizeof(long);
                                  
  uoldDEV = AllocClear(lUold);    
  uDEV = AllocClear(lVarSz);      
  qDEV = AllocClear(lVarSz);      
  dqDEV = AllocClear(lVarSz);     
  qxpDEV = AllocClear(lVarSz);    
  qleftDEV = AllocClear(lVarSz); 
  qrightDEV = AllocClear(lVarSz);
  qxmDEV = AllocClear(lVarSz);    
  qgdnvDEV = AllocClear(lVarSz);
  fluxDEV = AllocClear(lVarSz);   

  eDEV = AllocClear(lSz);         
  cDEV = AllocClear(lSz);         
  sgnmDEV = AllocClearL(lSzL);    
}

void
oclFreeOnDevice()
{
  OCLFREE(uoldDEV);
  OCLFREE(uDEV);
  OCLFREE(qDEV);
  OCLFREE(dqDEV);
  OCLFREE(qxpDEV);
  OCLFREE(qxmDEV);
  OCLFREE(eDEV);
  OCLFREE(cDEV);
  OCLFREE(qleftDEV);
  OCLFREE(qrightDEV);
  OCLFREE(qgdnvDEV);
  OCLFREE(fluxDEV);
  OCLFREE(sgnmDEV);
}

#define GETARRV(vdev, v) do { cl_event event; status = clEnqueueReadBuffer(cqueue, (vdev), CL_TRUE, 0, Hstep * H.nxyt * H.nvar * sizeof(real_t), (v), 0, NULL, &event); oclCheckErr(status, ""); status = clReleaseEvent(event); oclCheckErr(status, ""); } while(0);
#define GETARR(vdev, v)  do { cl_event event; status = clEnqueueReadBuffer(cqueue, (vdev), CL_TRUE, 0, Hstep * H.nxyt * sizeof(real_t), (v), 0, NULL, &event); oclCheckErr(status, ""); status = clReleaseEvent(event); oclCheckErr(status, ""); } while(0);

void
oclHydroGodunov(long idimStart, real_t dt, const hydroparam_t H, hydrovar_t * Hv, hydrowork_t * Hw, hydrovarwork_t * Hvw)
{
  cl_int status;
  // Local variables
  struct timespec start, end;
  int i, j, idim, idimIndex;
  int Hmin, Hmax, Hstep, Hnxystep;
  int Hdimsize;
  int Hndim_1;
  int slices, iend;
  real_t dtdx;
  size_t lVarSz = H.arVarSz * H.nxystep * sizeof(real_t);
  long Hnxyt = H.nxyt;
  int clear = 0;
  static FILE *fic = NULL;

  if (fic == NULL && H.prt) {
    char logname[256];
    sprintf(logname, "TRACE.%04d_%04d.txt", H.nproc, H.mype);
    fic = fopen(logname, "w");
  }

  WHERE("hydro_godunov");
  if (H.prt) fprintf(fic, "loop dt=%lg\n", dt);

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
    start = cclock();
    oclMakeBoundary(idim, H, Hv, uoldDEV);
    end = cclock();
    functim[TIM_MAKBOU] += ccelaps(start, end);
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
      long offsetIP = IHVWS(0, 0, IP);
      long offsetID = IHVWS(0, 0, ID);
      int Hnxyt = H.nxyt;
      iend = i + Hstep;
      if (iend >= Hmax) iend = Hmax;
      slices = iend - i;

      if (clear) oclMemset(uDEV, 0, lVarSz);
      start = cclock();
      oclGatherConservativeVars(idim, i, H.imin, H.imax, H.jmin, H.jmax, H.nvar, H.nxt, H.nyt, H.nxyt, slices, Hnxystep, uoldDEV, uDEV);
      end = cclock();
      functim[TIM_GATCON] += ccelaps(start, end);
      if (H.prt) {fprintf(fic, "ConservativeVars %ld %ld %ld %ld %d %d\n", H.nvar, H.nxt, H.nyt, H.nxyt, slices, Hstep);}
      if (H.prt) { GETARRV(uDEV, Hvw->u); }
      PRINTARRAYV2(fic, Hvw->u, Hdimsize, "u", H);

      // Convert to primitive variables
      start = cclock();
      oclConstoprim(Hdimsize, H.nxyt, H.nvar, H.smallr, slices, Hnxystep, uDEV, qDEV, eDEV);
      end = cclock();
      functim[TIM_CONPRI] += ccelaps(start, end);
      if (H.prt) { GETARR (eDEV, Hw->e); }
      if (H.prt) { GETARRV(qDEV, Hvw->q); }
      PRINTARRAY(fic, Hw->e, Hdimsize, "e", H);
      PRINTARRAYV2(fic, Hvw->q, Hdimsize, "q", H);

      start = cclock();
      oclEquationOfState(offsetIP, offsetID, 0, Hdimsize, H.smallc, H.gamma, slices, H.nxyt, qDEV, eDEV, cDEV);
      end = cclock();
      functim[TIM_EOS] += ccelaps(start, end);
      if (H.prt) { GETARR (cDEV, Hw->c); }
      PRINTARRAY(fic, Hw->c, Hdimsize, "c", H);
      if (H.prt) { GETARRV (qDEV, Hvw->q); }
      PRINTARRAYV2(fic, Hvw->q, Hdimsize, "q", H);

      if (clear) oclMemset(dqDEV, 0, H.arVarSz * H.nxystep);
      // Characteristic tracing
      if (H.iorder != 1) {
	if (clear) oclMemset(dqDEV, 0, H.arVarSz);
	start = cclock();
        oclSlope(Hdimsize, H.nvar, H.nxyt, H.slope_type, slices, Hstep, qDEV, dqDEV);
	end = cclock();
	functim[TIM_SLOPE] += ccelaps(start, end);
	if (H.prt) { GETARRV(dqDEV, Hvw->dq); }
	PRINTARRAYV2(fic, Hvw->dq, Hdimsize, "dq", H);
      }
      start = cclock();
      oclTrace(dtdx, Hdimsize, H.scheme, H.nvar, H.nxyt, slices, Hstep, qDEV, dqDEV, cDEV, qxmDEV, qxpDEV);
      end = cclock();
      functim[TIM_TRACE] += ccelaps(start, end);
      if (H.prt) { GETARRV(qxmDEV, Hvw->qxm); }
      if (H.prt) { GETARRV(qxpDEV, Hvw->qxp); }
      PRINTARRAYV2(fic, Hvw->qxm, Hdimsize, "qxm", H);
      PRINTARRAYV2(fic, Hvw->qxp, Hdimsize, "qxp", H);
      start = cclock();
      oclQleftright(idim, H.nx, H.ny, H.nxyt, H.nvar, slices, Hstep, qxmDEV, qxpDEV, qleftDEV, qrightDEV);
      end = cclock();
      functim[TIM_QLEFTR] += ccelaps(start, end);
      if (H.prt) { GETARRV(qleftDEV, Hvw->qleft); }
      if (H.prt) { GETARRV(qrightDEV, Hvw->qright); }
      PRINTARRAYV2(fic, Hvw->qleft, Hdimsize, "qleft", H);
      PRINTARRAYV2(fic, Hvw->qright, Hdimsize, "qright", H);

      // Solve Riemann problem at interfaces
      start = cclock();
      oclRiemann(Hndim_1, H.smallr, H.smallc, H.gamma, H.niter_riemann, H.nvar, H.nxyt, slices, Hstep,
		 qleftDEV, qrightDEV, qgdnvDEV,sgnmDEV);
      end = cclock();
      functim[TIM_RIEMAN] += ccelaps(start, end);
      if (H.prt) { GETARRV(qgdnvDEV, Hvw->qgdnv); }
      PRINTARRAYV2(fic, Hvw->qgdnv, Hdimsize, "qgdnv", H);
      // Compute fluxes
      if (clear) oclMemset(fluxDEV, 0, H.arVarSz);
      start = cclock();
      oclCmpflx(Hdimsize, H.nxyt, H.nvar, H.gamma, slices, Hnxystep, qgdnvDEV, fluxDEV);
      end = cclock();
      functim[TIM_CMPFLX] += ccelaps(start, end);
      if (H.prt) { GETARRV(fluxDEV, Hvw->flux); }
      PRINTARRAYV2(fic, Hvw->flux, Hdimsize, "flux", H);
      if (H.prt) { GETARRV(uDEV, Hvw->u); }
      PRINTARRAYV2(fic, Hvw->u, Hdimsize, "u", H);
      // if (H.prt) {
      // 	GETUOLD; PRINTUOLD(fic, H, Hv);
      // }
      if (H.prt) fprintf(fic, "dxdt=%lg\n", dtdx);
      start = cclock();
      oclUpdateConservativeVars(idim, i, dtdx, H.imin, H.imax, H.jmin, H.jmax, H.nvar, H.nxt, H.nyt, H.nxyt, slices, Hnxystep, 
				uoldDEV, uDEV, fluxDEV);
      end = cclock();
      functim[TIM_UPDCON] += ccelaps(start, end);
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
