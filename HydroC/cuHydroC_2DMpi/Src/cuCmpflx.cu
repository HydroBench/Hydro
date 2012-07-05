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

#include <math.h>
#include <malloc.h>
// #include <unistd.h>
// #include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "parametres.h"
#include "perfcnt.h"
#include "utils.h"
#include "cuCmpflx.h"
#include "gridfuncs.h"

__global__ void
Loop1KcuCmpflx(long narray, long Hnxyt, double Hgamma, const int slices, const int Hnxystep, //
               double *RESTRICT qgdnv,  // [Hnvar][Hnxystep][Hnxyt]
               double *RESTRICT flux    // [Hnvar][Hnxystep][Hnxyt]
  ) {
  double entho, ekin, etot;
  int i, j;
  double fluxID, gdnvIU, gdnvIV, gdnvIP, gdnvID;
  idx2d(i, j, Hnxyt);
  if (j >= slices)
    return;

  if (i >= narray) return;

  entho = one / (Hgamma - one);
  gdnvIU = qgdnv[IHVWS(i, j, IU)];
  gdnvIV = qgdnv[IHVWS(i, j, IV)];
  gdnvIP = qgdnv[IHVWS(i, j, IP)];
  gdnvID = qgdnv[IHVWS(i, j, ID)];
  // Mass density
  fluxID = gdnvID * gdnvIU;
  // Normal momentum
  flux[IHVWS(i, j, IU)] = fluxID * gdnvIU + gdnvIP;
  // Transverse momentum 1
  flux[IHVWS(i, j, IV)] = fluxID * gdnvIV;
  // Total energy
  ekin = half * gdnvID * (Square(gdnvIU) + Square(gdnvIV));
  etot = gdnvIP * entho + ekin;
  flux[IHVWS(i, j, IP)] = gdnvIU * (etot + gdnvIP);
  flux[IHVWS(i, j, ID)] = fluxID;
}

__global__ void
Loop2KcuCmpflx(const long narray, const long Hnxyt, const long Hnvar, const int slices, const int Hnxystep,  //
               double *RESTRICT qgdnv,  // [Hnvar][Hnxystep][Hnxyt]
               double *RESTRICT flux    // [Hnvar][Hnxystep][Hnxyt]
  ) {
  int IN;
  int i, j;
  idx2d(i, j, Hnxyt);
  if (j >= slices)
    return;

  for (IN = IP + 1; IN < Hnvar; IN++) {
    flux[IHVWS(i, j, IN)] = flux[IHVWS(i, j, IN)] * qgdnv[IHVWS(i, j, IN)];
  }
}

void
cuCmpflx(int narray, int Hnxyt, int Hnvar, double Hgamma, int slices, const int Hnxstep,   //
         double *RESTRICT qgdnv,        //[Hnvar][Hnxstep][Hnxyt]
         double *RESTRICT flux          //[Hnvar][Hnxstep][Hnxyt]
  ) {
  dim3 grid, block;
  int nops;

  WHERE("cmpflx");

  SetBlockDims(Hnxyt * slices, THREADSSZ, block, grid);

  // Compute fluxes
  Loop1KcuCmpflx <<< grid, block >>> (narray, Hnxyt, Hgamma, slices, Hnxstep, qgdnv, flux);
  CheckErr("Loop1KcuCmpflx");
  nops = slices * narray;
  FLOPS(13 * nops, 0 * nops, 0 * nops, 0 * nops);

  // Other advected quantities
  if (Hnvar > IP + 1) {
    Loop2KcuCmpflx <<< grid, block >>> (narray, Hnxyt, Hnvar, slices, Hnxstep, qgdnv, flux);
    CheckErr("Loop2KcuCmpflx");
  }
  cudaThreadSynchronize();
  CheckErr("After synchronize cuCmpflx");
}                               // cmpflx

//EOF
