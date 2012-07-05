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

#include "parametres.h"
#include "utils.h"
#include "cuConservar.h"
#include "gridfuncs.h"
#include "perfcnt.h"

__global__ void
Loop1KcuGather(const long rowcol, const long Hnxt, const long Himin, const long Himax, const long Hnyt, const long Hnxyt, const int slices, const int Hnxystep,     //
               double *RESTRICT uold,   // [Hnvar * Hnxt * Hnyt]
               double *RESTRICT u       // [Hnvar][Hnxystep][Hnxyt]
	       ) {
  int i, j, idx, ivar;
  i = idx1d();

  if (i < Himin)
    return;
  if (i >= Himax)
    return;

  // printf("%ld %ld \n", i, j);
  for (j = 0; j < slices; j++) {
    u[IHVWS(i, j, ID)] = uold[IHU(i, rowcol + j, ID)];
    u[IHVWS(i, j, IU)] = uold[IHU(i, rowcol + j, IU)];
    u[IHVWS(i, j, IV)] = uold[IHU(i, rowcol + j, IV)];
    u[IHVWS(i, j, IP)] = uold[IHU(i, rowcol + j, IP)];
  }
}

__global__ void
Loop2KcuGather(const long rowcol, const long Hnxt, const long Hjmin, const long Hjmax, const long Hnyt, const long Hnxyt, const int slices, const int Hnxystep,     //
               double *RESTRICT uold,   // [Hnvar * Hnxt * Hnyt]
               double *RESTRICT u       // [Hnvar][Hnxystep][Hnxyt]
	       ) {
  int i, j, idx, ivar;
  i = idx1d();
  if (i < Hjmin)
    return;
  if (i >= Hjmax)
    return;
  // printf("%d\n", slices);
  for (j = 0; j < slices; j++) {
    u[IHVWS(i, j, ID)] = uold[IHU(rowcol + j, i, ID)];
    u[IHVWS(i, j, IV)] = uold[IHU(rowcol + j, i, IU)];
    u[IHVWS(i, j, IU)] = uold[IHU(rowcol + j, i, IV)];
    u[IHVWS(i, j, IP)] = uold[IHU(rowcol + j, i, IP)];
  }
}

__global__ void
Loop3KcuGather(const long rowcol, const long Hnxt, const long Himin, const long Himax, const long Hnyt, const long Hnxyt, const long Hnvar, const int slices, const int Hnxystep,   //
               double *RESTRICT uold,   // [Hnvar * Hnxt * Hnyt]
               double *RESTRICT u       // [Hnvar][Hnxystep][Hnxyt]
	       ) {
  int i, j;
  int ivar;
  idx2d(i, j, Hnxyt);

  if (j >= slices)
    return;

  for (ivar = IP + 1; ivar < Hnvar; ivar++) {
    u[IHVWS(i, j, ivar)] = uold[IHU(i, rowcol + j, ivar)];
  }
}

__global__ void
Loop4KcuGather(const long rowcol, const long Hnxt, const long Himin, const long Himax, const long Hnyt, const long Hnxyt, const long Hnvar, const int slices, const int Hnxystep,   //
               double *RESTRICT uold,   // [Hnvar * Hnxt * Hnyt]
               double *RESTRICT u       // [Hnvar][Hnxystep][Hnxyt]
	       ) {
  int i, j;
  int ivar;
  idx2d(i, j, Hnxyt);

  if (j >= slices)
    return;

  // reconsiderer le calcul d'indices en supprimant la boucle sur ivar et
  // en la ventilant par thread
  for (ivar = IP + 1; ivar < Hnvar; ivar++) {
    u[IHVWS(i, j, ivar)] = uold[IHU(rowcol + j, i, ivar)];
  }
}
void
cuGatherConservativeVars(const long idim, const long rowcol, const long Himin, const long Himax, const long Hjmin,      //
                         const long Hjmax, const long Hnvar, const long Hnxt, const long Hnyt, const long Hnxyt, const int slices, const int Hnxystep,       //
                         double *RESTRICT uoldDEV,      // [Hnvar * Hnxt * Hnyt]
                         double *RESTRICT uDEV  // [Hnvar][Hnxystep][Hnxyt]
			 ) {
  dim3 grid, block;


  WHERE("gatherConservativeVars");
  if (idim == 1) {
    // Gather conservative variables
    SetBlockDims(Hnxyt, THREADSSZ, block, grid);
    Loop1KcuGather <<< grid, block >>> (rowcol, Hnxt, Himin, Himax, Hnyt, Hnxyt, slices, Hnxystep, uoldDEV, uDEV);

    if (Hnvar > IP + 1) {
      Loop3KcuGather <<< grid, block >>> (rowcol, Hnxt, Himin, Himax, Hnyt, Hnxyt, Hnvar, slices, Hnxystep, uoldDEV, uDEV);
    }
  } else {
    // Gather conservative variables
    SetBlockDims(Hnxyt, THREADSSZ, block, grid);
    Loop2KcuGather <<< grid, block >>> (rowcol, Hnxt, Hjmin, Hjmax, Hnyt, Hnxyt, slices, Hnxystep, uoldDEV, uDEV);
    if (Hnvar > IP + 1) {
      Loop4KcuGather <<< grid, block >>> (rowcol, Hnxt, Hjmin, Hjmax, Hnyt, Hnxyt, Hnvar, slices, Hnxystep, uoldDEV, uDEV);
    }
  }
}

__global__ void
Loop1KcuUpdate(const long rowcol, const double dtdx, const long Himin, const long Himax, const long Hnxt, const long Hnyt, const long Hnxyt, const int slices, const int Hnxystep,   //
               double *RESTRICT uold,   // [Hnvar * Hnxt * Hnyt]
               double *RESTRICT u,      // [Hnvar][Hnxystep][Hnxyt]
               double *RESTRICT flux    // [Hnvar][Hnxystep][Hnxyt]
	       ) {
  int i, j;
  idx2d(i, j, Hnxyt);

  if (j >= slices)
    return;
  if (i >= (Himax  - ExtraLayer))
    return;
  if (i < (Himin + ExtraLayer))
    return;


  uold[IHU(i, rowcol + j, ID)] = u[IHVWS(i, j, ID)] + (flux[IHVWS(i - 2, j, ID)] - flux[IHVWS(i - 1, j, ID)]) * dtdx;
  uold[IHU(i, rowcol + j, IU)] = u[IHVWS(i, j, IU)] + (flux[IHVWS(i - 2, j, IU)] - flux[IHVWS(i - 1, j, IU)]) * dtdx;
  uold[IHU(i, rowcol + j, IV)] = u[IHVWS(i, j, IV)] + (flux[IHVWS(i - 2, j, IV)] - flux[IHVWS(i - 1, j, IV)]) * dtdx;
  uold[IHU(i, rowcol + j, IP)] = u[IHVWS(i, j, IP)] + (flux[IHVWS(i - 2, j, IP)] - flux[IHVWS(i - 1, j, IP)]) * dtdx;
}

__global__ void
Loop2KcuUpdate(const long rowcol, const double dtdx, const long Himin, const long Himax, const long Hnvar,      //
               const long Hnxt, const long Hnyt, const long Hnxyt, const int slices, const int Hnxystep,     //
               double *RESTRICT uold,   // [Hnvar * Hnxt * Hnyt]
               double *RESTRICT u,      // [Hnvar][Hnxystep][Hnxyt]
               double *RESTRICT flux    // [Hnvar][Hnxystep][Hnxyt]
	       ) {
  int i, j;
  idx2d(i, j, Hnxyt);
  int ivar;

  if (j >= slices)
    return;
  if (i >= (Himax  - ExtraLayer))
    return;
  if (i < (Himin + ExtraLayer))
    return;


  for (ivar = IP + 1; ivar < Hnvar; ivar++) {
    uold[IHU(i, rowcol + j, ivar)] =
      u[IHVWS(i, j, ivar)] + (flux[IHVWS(i - 2, j, ivar)] - flux[IHVWS(i - 1, j, ivar)]) * dtdx;
  }
}

__global__ void
Loop3KcuUpdate(const long rowcol, const double dtdx, const long Hjmin, const long Hjmax, const long Hnxt,       //
               const long Hnyt, const long Hnxyt, const int slices, const int Hnxystep,      //
               double *RESTRICT uold,   // [Hnvar * Hnxt * Hnyt]
               double *RESTRICT u,      // [Hnvar][Hnxystep][Hnxyt]
               double *RESTRICT flux    // [Hnvar][Hnxystep][Hnxyt]
	       ) {
  int s, j;
  idx2d(j, s, Hnxyt);
  if (s >= slices)
    return;

  if (j >= (Hjmax  - ExtraLayer))
    return;
  if (j < (Hjmin + ExtraLayer))
    return;


  uold[IHU(rowcol + s, j, ID)] = u[IHVWS(j, s, ID)] + (flux[IHVWS(j - 2, s, ID)] - flux[IHVWS(j - 1, s, ID)]) * dtdx;
  uold[IHU(rowcol + s, j, IP)] = u[IHVWS(j, s, IP)] + (flux[IHVWS(j - 2, s, IP)] - flux[IHVWS(j - 1, s, IP)]) * dtdx;
  uold[IHU(rowcol + s, j, IV)] = u[IHVWS(j, s, IU)] + (flux[IHVWS(j - 2, s, IU)] - flux[IHVWS(j - 1, s, IU)]) * dtdx;
  uold[IHU(rowcol + s, j, IU)] = u[IHVWS(j, s, IV)] + (flux[IHVWS(j - 2, s, IV)] - flux[IHVWS(j - 1, s, IV)]) * dtdx;
}

__global__ void
Loop4KcuUpdate(const long rowcol, const double dtdx, const long Hjmin,  //
               const long Hjmax, const long Hnvar, const long Hnxt, const long Hnyt, const long Hnxyt, const int slices, const int Hnxystep, //
               double *RESTRICT uold,   // [Hnvar * Hnxt * Hnyt]
               double *RESTRICT u,      // [Hnvar][Hnxystep][Hnxyt]
               double *RESTRICT flux    // [Hnvar][Hnxystep][Hnxyt]
	       ) {
  int s, j;
  idx2d(j, s, Hnxyt);
  int ivar;
  if (s >= slices)
    return;

  if (j >= (Hjmax  - ExtraLayer))
    return;
  if (j < (Hjmin + ExtraLayer))
    return;

  for (ivar = IP + 1; ivar < Hnvar; ivar++) {
    uold[IHU(rowcol + s, j, ivar)] = u[IHVWS(j, s, ivar)] + (flux[IHVWS(j - 2, s, ivar)] - flux[IHVWS(j - 1, s, ivar)]) * dtdx;
  }
}

void
cuUpdateConservativeVars(const long idim, const long rowcol, const double dtdx, const long Himin, const long Himax,     // 
                         const long Hjmin, const long Hjmax, const long Hnvar, const long Hnxt, const long Hnyt, const long Hnxyt, const int slices, const int Hnxystep,     //
                         double *RESTRICT uoldDEV,      // [Hnvar * Hnxt * Hnyt]
                         double *RESTRICT uDEV,         // [Hnvar][Hnxystep][Hnxyt]
                         double *RESTRICT fluxDEV       // [Hnvar][Hnxystep][Hnxyt]
			 ) {
  dim3 grid, block;
  int nops;
  WHERE("updateConservativeVars");

  if (idim == 1) {
    SetBlockDims(Hnxyt * slices, THREADSSZ, block, grid);
    // Update conservative variables
    Loop1KcuUpdate <<< grid, block >>> (rowcol, dtdx, Himin, Himax, Hnxt, Hnyt, Hnxyt, slices, Hnxystep,  uoldDEV, uDEV, fluxDEV);
    CheckErr("Loop1KcuUpdate");
    nops = (IP+1) * slices * ((Himax - ExtraLayer) - (Himin + ExtraLayer));
    FLOPS(3 * nops, 0 * nops, 0 * nops, 0 * nops);
    if (Hnvar > IP + 1) {
      Loop2KcuUpdate <<< grid, block >>> (rowcol, dtdx, Himin, Himax, Hnvar, Hnxt, Hnyt, Hnxyt, slices, Hnxystep,  uoldDEV, uDEV, fluxDEV);
      CheckErr("Loop2KcuUpdate");
    }
  } else {
    // Update conservative variables
    SetBlockDims(Hnxyt * slices, THREADSSZ, block, grid);
    Loop3KcuUpdate <<< grid, block >>> (rowcol, dtdx, Hjmin, Hjmax, Hnxt, Hnyt, Hnxyt, slices, Hnxystep,  uoldDEV, uDEV, fluxDEV);
    CheckErr("Loop3KcuUpdate");
    nops = slices * ((Hjmax - ExtraLayer) - (Hjmin + ExtraLayer));
    FLOPS(12 * nops, 0 * nops, 0 * nops, 0 * nops);
 if (Hnvar > IP + 1) {
      Loop4KcuUpdate <<< grid, block >>> (rowcol, dtdx, Hjmin, Hjmax, Hnvar, Hnxt, Hnyt, Hnxyt, slices, Hnxystep,  uoldDEV, uDEV, fluxDEV);
      CheckErr("Loop4KcuUpdate");
    }
  }
  cudaThreadSynchronize();
  CheckErr("cudaThreadSynchronize");
}

//EOF
