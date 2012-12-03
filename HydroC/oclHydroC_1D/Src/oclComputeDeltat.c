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

#include "parametres.h"
#include "oclComputeDeltat.h"
#include "oclHydroGodunov.h"
#include "utils.h"
#include "oclEquationOfState.h"

#include "oclInit.h"
#include "ocltools.h"
#include "oclReduce.h"

#define IHV(i, j, v)  ((i) + Hnxt * ((j) + Hnyt * (v)))
#define IHVW(i, v) ((i) + (v) * Hnxyt)

void
oclComputeQEforRow(const long j, cl_mem uold, cl_mem q, cl_mem e,
                   const double Hsmallr, const long Hnx, const long Hnxt, const long Hnyt, const long Hnxyt)
{
  cl_int err = 0;
  dim3 gws, lws;
  cl_event event;
  double elapsk;

  oclMkNDrange(Hnx, THREADSSZ, NDR_1D, gws, lws);
  oclSetArg(ker[LoopKQEforRow], 0, sizeof(j), &j, __FILE__, __LINE__);
  oclSetArg(ker[LoopKQEforRow], 1, sizeof(uold), &uold, __FILE__, __LINE__);
  oclSetArg(ker[LoopKQEforRow], 2, sizeof(q), &q, __FILE__, __LINE__);
  oclSetArg(ker[LoopKQEforRow], 3, sizeof(e), &e, __FILE__, __LINE__);
  oclSetArg(ker[LoopKQEforRow], 4, sizeof(Hsmallr), &Hsmallr, __FILE__, __LINE__);
  oclSetArg(ker[LoopKQEforRow], 5, sizeof(Hnxt), &Hnxt, __FILE__, __LINE__);
  oclSetArg(ker[LoopKQEforRow], 6, sizeof(Hnyt), &Hnyt, __FILE__, __LINE__);
  oclSetArg(ker[LoopKQEforRow], 7, sizeof(Hnxyt), &Hnxyt, __FILE__, __LINE__);
  oclSetArg(ker[LoopKQEforRow], 8, sizeof(Hnx), &Hnx, __FILE__, __LINE__);

  // LoopKQEforRow <<< grid, block >>> (j, uold, q, e, Hsmallr, Hnxt, Hnyt, Hnxyt, Hnx);
  err = clEnqueueNDRangeKernel(cqueue, ker[LoopKQEforRow], 1, NULL, gws, lws, 0, NULL, &event);
  oclCheckErr(err, "clEnqueueNDRangeKernel LoopKQEforRow");
  err = clWaitForEvents(1, &event);
  oclCheckErr(err, "clWaitForEvents");
  elapsk = oclChronoElaps(event);
  err = clReleaseEvent(event);
  oclCheckErr(err, "clReleaseEvent");
}

void
oclCourantOnXY(cl_mem courant, const long Hnx, const long Hnxyt, cl_mem c, cl_mem q, double Hsmallc)
{
//     dim3 grid, block;
//     SetBlockDims(Hnx, THREADSSZ, block, grid);
//     LoopKcourant <<< grid, block >>> (q, courant, Hsmallc, c, Hnxyt, Hnx);
//     CheckErr("courantOnXY");
//     cudaThreadSynchronize();
//     CheckErr("courantOnXY");
  double elapsk;
  OCLINITARG;

  OCLSETARG(ker[LoopKcourant], q);
  OCLSETARG(ker[LoopKcourant], courant);
  OCLSETARG(ker[LoopKcourant], Hsmallc);
  OCLSETARG(ker[LoopKcourant], c);
  OCLSETARG(ker[LoopKcourant], Hnxyt);
  OCLSETARG(ker[LoopKcourant], Hnx);

  elapsk = oclLaunchKernel(ker[LoopKcourant], cqueue, Hnx, THREADSSZ, __FILE__, __LINE__);
}

void
oclComputeDeltat(double *dt, const hydroparam_t H, hydrowork_t * Hw, hydrovar_t * Hv, hydrovarwork_t * Hvw)
{
  long j;
  cl_mem uoldDEV, qDEV, eDEV, cDEV, courantDEV;
  double *lcourant;
  double maxCourant;
  long Hnxyt = H.nxyt;
  cl_int err = 0;
  long offsetIP = IHVW(0, IP);
  long offsetID = IHVW(0, ID);

  WHERE("compute_deltat");

  //   compute time step on grid interior

  // on recupere les buffers du device qui sont deja alloues
  oclGetUoldQECDevicePtr(&uoldDEV, &qDEV, &eDEV, &cDEV);

  lcourant = (double *) calloc(Hnxyt, sizeof(double));
  courantDEV = clCreateBuffer(ctx, CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, Hnxyt * sizeof(double), lcourant, &err);
  oclCheckErr(err, "clCreateBuffer");

//     status = cudaMalloc((void **) &courantDEV, H.nxyt * sizeof(double));
//     VERIF(status, "cudaMalloc cuComputeDeltat");
//     status = cudaMemset(courantDEV, 0, H.nxyt * sizeof(double));
//     VERIF(status, "cudaMemset cuComputeDeltat");

  for (j = H.jmin + ExtraLayer; j < H.jmax - ExtraLayer; j++) {
    oclComputeQEforRow(j, uoldDEV, qDEV, eDEV, H.smallr, H.nx, H.nxt, H.nyt, H.nxyt);
    oclEquationOfState(qDEV, eDEV, cDEV, offsetIP, offsetID, 0, H.nx, H.smallc, H.gamma);
    // on calcule courant pour chaque cellule de la ligne pour tous les j
    oclCourantOnXY(courantDEV, H.nx, H.nxyt, cDEV, qDEV, H.smallc);
  }

  err = clEnqueueReadBuffer(cqueue, courantDEV, CL_TRUE, 0, H.nx * sizeof(double), lcourant, 0, NULL, NULL);

  int ic;
  double lmax = 0.;
  for (ic = 0; ic < H.nx; ic++) {
    lmax = fmax(lmax, lcourant[ic]);
  }

  // on cherche le max global des max locaux
  maxCourant = oclReduceMax(courantDEV, H.nx);
  // fprintf(stderr, "Courant=%lg (%lg)\n", maxCourant, lmax);
  *dt = H.courant_factor * H.dx / maxCourant;
  err = clReleaseMemObject(courantDEV);
  oclCheckErr(err, "clReleaseMemObject");
  free(lcourant);
  // exit(0);
  // fprintf(stdout, "%g %g %g %g\n", cournox, cournoy, H.smallc, H.courant_factor);
}                               // compute_deltat


//EOF
