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

#include <stdio.h>
// #include <stdlib.h>
#include <malloc.h>
// #include <unistd.h>
#include <math.h>
#include <float.h>
#include <assert.h>

#include "parametres.h"
#include "oclComputeDeltat.h"
#include "oclHydroGodunov.h"
#include "utils.h"
#include "oclEquationOfState.h"

#include "oclInit.h"
#include "ocltools.h"
#include "oclReduce.h"

static void
ClearArrayDble(cl_mem array, long lgr) {
  real_t lzero = 0.;
  assert(array != NULL);
  assert(lgr  >= 0);
  OCLSETARG03(ker[KernelMemset], array, lzero, lgr);
  oclLaunchKernel(ker[KernelMemset], cqueue, lgr, THREADSSZ, __FILE__, __LINE__);
}

void
oclComputeQEforRow(const long j, cl_mem uold, cl_mem q, cl_mem e,
                   const real_t Hsmallr, const long Hnx, const long Hnxt,
                   const long Hnyt, const long Hnxyt, const int slices, const int Hstep) {
  cl_int err = 0;
  dim3 gws, lws;
  double elapsk;

  OCLSETARG11(ker[LoopKQEforRow], j, uold, q, e, Hsmallr, Hnxt, Hnyt, Hnxyt, Hnx, slices, Hstep);
  elapsk = oclLaunchKernel2D(ker[LoopKQEforRow], cqueue, Hnxyt, slices, THREADSSZ, __FILE__, __LINE__);
}

void
oclCourantOnXY(cl_mem courant, const long Hnx, const long Hnxyt, cl_mem c, cl_mem q,
               real_t Hsmallc, const int slices, const int Hstep) {
  double elapsk;
  OCLSETARG08(ker[LoopKcourant], q, courant, Hsmallc, c, Hnxyt, Hnx, slices, Hstep);
  elapsk = oclLaunchKernel2D(ker[LoopKcourant], cqueue, Hnxyt, slices, THREADSSZ, __FILE__, __LINE__);
}

#define GETARRV(vdev, v) do { cl_event event;   cl_int status; status = clEnqueueReadBuffer(cqueue, (vdev), CL_TRUE, 0, Hstep * H.nxyt * H.nvar * sizeof(real_t), (v), 0, NULL, &event); oclCheckErr(status, ""); status = clReleaseEvent(event); oclCheckErr(status, ""); } while(0);
#define GETARR(vdev, v)  do { cl_event event;   cl_int status; status = clEnqueueReadBuffer(cqueue, (vdev), CL_TRUE, 0, Hstep * H.nxyt * sizeof(real_t), (v), 0, NULL, &event); oclCheckErr(status, ""); status = clReleaseEvent(event); oclCheckErr(status, ""); } while(0);

void
oclComputeDeltat(real_t *dt, const hydroparam_t H, hydrowork_t * Hw, hydrovar_t * Hv, hydrovarwork_t * Hvw) {
  long j;
  cl_mem uoldDEV, qDEV, eDEV, cDEV, courantDEV;
  real_t *lcourant;
  real_t maxCourant = -1.0 * FLT_MAX;
  real_t lmaxCourant = -1.0 * FLT_MAX;
  long Hnxyt = H.nxyt;
  cl_int err = 0;
  int slices = 1, jend, Hstep, Hnxystep, i;
  long Hmin, Hmax;
  static FILE *fic = NULL;

  WHERE("compute_deltat");

  if (fic == NULL && H.prt) {
    char logname[256];
    sprintf(logname, "DT.%04d_%04d.txt", H.nproc, H.mype);
    fic = fopen(logname, "w");
  }
  //   compute time step on grid interior
  Hnxystep = H.nxystep;
  Hstep = H.nxystep;
  // Hstep = 1;
  Hmin = H.jmin + ExtraLayer;
  Hmax = H.jmax - ExtraLayer;

  //   compute time step on grid interior

  // reuse of already allocated buffers
  oclGetUoldQECDevicePtr(&uoldDEV, &qDEV, &eDEV, &cDEV);

  lcourant = (real_t *) calloc(Hnxyt * H.nxystep, sizeof(real_t));
  // the buffer is created and filled by zeros immediately
  courantDEV = clCreateBuffer(ctx, CL_MEM_READ_WRITE, H.nxyt * H.nxystep * sizeof(real_t), NULL, &err);
  oclCheckErr(err, "clCreateBuffer");
  ClearArrayDble(courantDEV, H.nxyt * H.nxystep);

  long offsetIP = IHVWS(0, 0, IP);
  long offsetID = IHVWS(0, 0, ID);


  for (j = Hmin; j < Hmax; j += Hstep) {
    int par10;
    jend = j + Hstep;
    if (jend >= Hmax)
      jend = Hmax;
    slices = jend - j;

    //merged 3 kernel calls into one
    par10 = (int)H.nxystep;
    OCLSETARG17( ker[LoopKComputeDeltat], j, uoldDEV, qDEV, eDEV, H.nxt, H.nyt, H.nxyt, H.nx, slices, par10,
		 offsetIP, offsetID, H.smallc, H.gamma, H.smallr, cDEV, courantDEV );
    oclLaunchKernel2D(ker[LoopKComputeDeltat], cqueue, H.nxyt, slices, THREADSSZ, __FILE__, __LINE__);
    
    if (H.prt) {
      GETARR(courantDEV, lcourant);
      PRINTARRAY(fic, lcourant, H.nx, "lcourant", H);
    }

  }

  // find the global max of the local maxs
  // GETARR (courantDEV, lcourant);
  // PRINTARRAY(fic, lcourant, H.nx, "lcourant avant reduction", H);
  maxCourant = oclReduceMax(courantDEV, H.nxyt * H.nxystep);

  lmaxCourant = -1.0 * FLT_MAX;
  for (i = 0; i < H.nxyt * H.nxystep; i++) {
    lmaxCourant = fmax(lmaxCourant, lcourant[i]);
  }

  *dt = H.courant_factor * H.dx / maxCourant;
  if (H.prt)
    fprintf(fic, "(%02d) hnxystep=%d/%ld maxCourant=%lf/%lf dt=%lg %ld\n", H.mype, Hstep, H.nxystep, maxCourant,
            lmaxCourant, *dt, H.nxyt);
  fflush(stdout);
  OCLFREE(courantDEV);
  free(lcourant);
}                               // compute_deltat


//EOF
