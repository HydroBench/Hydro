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
#include <cuda.h>
#include "parametres.h"
#include "utils.h"
#include "perfcnt.h"
#include "gridfuncs.h"
#include "cuConstoprim.h"


__global__ void
Loop1KcuConstoprim(const long n, const long Hnxyt, const double Hsmallr, const long Hnvar, const int slices, const int Hnxystep,     //
                   double *RESTRICT u,  // [Hnvar][Hnxystep][Hnxyt]
                   double *RESTRICT q,  // [Hnvar][Hnxystep][Hnxyt]
                   double *RESTRICT e   //        [Hnxystep][Hnxyt] 
  ) {
  double eken;
  int i, j, idx;
  idx = idx1d();
  j = idx / Hnxyt;
  i = idx % Hnxyt;

  if (j >= slices)
    return;
  if (i >= n) return;

  q[IHVWS(i,j,ID)] = MAX(u[IHVWS(i,j,ID)], Hsmallr);
  q[IHVWS(i,j,IU)] = u[IHVWS(i,j,IU)] / q[IHVWS(i,j,ID)];
  q[IHVWS(i,j,IV)] = u[IHVWS(i,j,IV)] / q[IHVWS(i,j,ID)];
  eken = half * (Square(q[IHVWS(i,j,IU)]) + Square(q[IHVWS(i,j,IV)]));
  q[IHVWS(i,j,IP)] = u[IHVWS(i,j,IP)] / q[IHVWS(i,j,ID)] - eken;
  e[IHS(i,j)] = q[IHVWS(i,j,IP)];
}

__global__ void
Loop2KcuConstoprim(const long n, const long Hnxyt, const long Hnvar, const int slices, const int Hnxystep,  //
                   double *RESTRICT u,  // [Hnvar][Hnxystep][Hnxyt]
                   double *RESTRICT q   // [Hnvar][Hnxystep][Hnxyt]
  ) {
  int IN;
  int i, j;
  idx2d(i, j, Hnxyt);
  if (j >= slices)
    return;
  if (i >= n) return;

  for (IN = IP + 1; IN < Hnvar; IN++) {
    q[IHVWS(i,j,IN)] = u[IHVWS(i,j,IN)] / q[IHVWS(i,j,IN)];
  }
}

void
cuConstoprim(const long n, const long Hnxyt, const long Hnvar, const double Hsmallr, const int slices, const int Hnxystep,  //
             double *RESTRICT uDEV,     // [Hnvar][Hnxystep][Hnxyt]
             double *RESTRICT qDEV,     // [Hnvar][Hnxystep][Hnxyt]
             double *RESTRICT eDEV      //        [Hnxystep][Hnxyt]
  ) {
  dim3 grid, block;
  int nops;
  WHERE("constoprim");
  SetBlockDims(Hnxyt * slices, THREADSSZ, block, grid);
  Loop1KcuConstoprim <<< grid, block >>> (n, Hnxyt, Hsmallr, Hnvar, slices, Hnxystep, uDEV, qDEV, eDEV);
  CheckErr("Loop1KcuConstoprim");
  nops = slices * n;
  FLOPS(5 * nops, 3 * nops, 1 * nops, 0 * nops);  

  if (Hnvar > IP + 1) {
    Loop2KcuConstoprim <<< grid, block >>> (n, Hnxyt, Hnvar, slices, Hnxystep,uDEV, qDEV);
    CheckErr("Loop2KcuConstoprim");
  }
  cudaDeviceSynchronize();
  CheckErr("After synchronize cuConstoprim");
}                               // constoprim


#undef IHS
#undef IHVW
#undef IHVWS
//EOF
