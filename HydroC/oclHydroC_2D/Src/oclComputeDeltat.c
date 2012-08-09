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
ClearArrayDble(cl_mem array, size_t lgrBytes) {
  int lzero = 0;
  long ldble = lgrBytes / sizeof(double);
  assert(array != NULL);
  OCLSETARG03(ker[KernelMemset], array, lzero, ldble);
  oclLaunchKernel(ker[KernelMemset], cqueue, lgrBytes, THREADSSZ, __FILE__, __LINE__);
}

void
oclComputeQEforRow(const long j, cl_mem uold, cl_mem q, cl_mem e,
                   const double Hsmallr, const long Hnx, const long Hnxt,
                   const long Hnyt, const long Hnxyt, const int slices, const int Hstep) {
  cl_int err = 0;
  dim3 gws, lws;
  double elapsk;

  OCLSETARG11(ker[LoopKQEforRow], j, uold, q, e, Hsmallr, Hnxt, Hnyt, Hnxyt, Hnx, slices, Hstep);
  elapsk = oclLaunchKernel(ker[LoopKQEforRow], cqueue, Hnxyt * slices, THREADSSZ, __FILE__, __LINE__);
}

void
oclCourantOnXY(cl_mem courant, const long Hnx, const long Hnxyt, cl_mem c, cl_mem q,
               double Hsmallc, const int slices, const int Hstep) {
  double elapsk;
  OCLSETARG08(ker[LoopKcourant], q, courant, Hsmallc, c, Hnxyt, Hnx, slices, Hstep);
  elapsk = oclLaunchKernel(ker[LoopKcourant], cqueue, Hnxyt * slices, THREADSSZ, __FILE__, __LINE__);
}

#define GETARRV(vdev, v) do { cl_event event;   cl_int status; status = clEnqueueReadBuffer(cqueue, (vdev), CL_TRUE, 0, Hstep * H.nxyt * H.nvar * sizeof(double), (v), 0, NULL, &event); oclCheckErr(status, ""); status = clReleaseEvent(event); oclCheckErr(status, ""); } while(0);
#define GETARR(vdev, v)  do { cl_event event;   cl_int status; status = clEnqueueReadBuffer(cqueue, (vdev), CL_TRUE, 0, Hstep * H.nxyt * sizeof(double), (v), 0, NULL, &event); oclCheckErr(status, ""); status = clReleaseEvent(event); oclCheckErr(status, ""); } while(0);

void
oclComputeDeltat(double *dt, const hydroparam_t H, hydrowork_t * Hw, hydrovar_t * Hv, hydrovarwork_t * Hvw) {
  long j;
  cl_mem uoldDEV, qDEV, eDEV, cDEV, courantDEV;
  double *lcourant;
  double maxCourant;
  double lmaxCourant;
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

  lcourant = (double *) calloc(Hnxyt * H.nxystep, sizeof(double));
  // the buffer is created and filled by zeros immediately
  courantDEV = clCreateBuffer(ctx, CL_MEM_READ_WRITE, H.nxyt * H.nxystep * sizeof(double), NULL, &err);
  oclCheckErr(err, "clCreateBuffer");
  ClearArrayDble(courantDEV, H.nxyt * H.nxystep * sizeof(double));

  long offsetIP = IHVWS(0, 0, IP);
  long offsetID = IHVWS(0, 0, ID);


  for (j = Hmin; j < Hmax; j += Hstep) {
    jend = j + Hstep;
    if (jend >= Hmax)
      jend = Hmax;
    slices = jend - j;
    // fprintf(stdout, "(%02d) slices=%d\n", H.mype, slices);

    // ClearArrayDble(eDEV, Hnxyt * H.nxystep * sizeof(double));
    // ClearArrayDble(cDEV, Hnxyt * H.nxystep * sizeof(double));
    // ClearArrayDble(qDEV, Hnxyt * H.nxystep * H.nvar * sizeof(double));

    oclComputeQEforRow(j, uoldDEV, qDEV, eDEV, H.smallr, H.nx, H.nxt, H.nyt, H.nxyt, slices, H.nxystep);
    // GETARR (eDEV, Hw->e);
    // PRINTARRAY(fic, Hw->e, H.nx, "e", H);
    // GETARRV (qDEV, Hvw->q);
    // PRINTARRAYV2(fic, Hvw->q, H.nx, "q", H);

    oclEquationOfState(offsetIP, offsetID, 0, H.nx, H.smallc, H.gamma, slices, H.nxyt, qDEV, eDEV, cDEV);
    // GETARR (cDEV, Hw->c);
    // PRINTARRAY(fic, Hw->c, H.nx, "c", H);
    oclCourantOnXY(courantDEV, H.nx, H.nxyt, cDEV, qDEV, H.smallc, slices, H.nxystep);
    if (H.prt) {
      GETARR(courantDEV, lcourant);
      PRINTARRAY(fic, lcourant, H.nx, "lcourant", H);
    }

  }

  // find the global max of the local maxs
  // GETARR (courantDEV, lcourant);
  // PRINTARRAY(fic, lcourant, H.nx, "lcourant avant reduction", H);
  maxCourant = oclReduceMax(courantDEV, H.nxyt * H.nxystep);

  lmaxCourant = 0.;
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
