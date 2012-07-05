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

// #include <stdlib.h>
// #include <unistd.h>
#include <math.h>
#include <stdio.h>

#include "gridfuncs.h"
#include "cuEquationOfState.h"
#include "perfcnt.h"
#include "utils.h"

__global__ void
LoopEOS(const long imin, const long imax, const double Hsmallc, const double Hgamma, const int Hnxyt, const int slices,   //
        double *RESTRICT rho,   // [Hnxystep][Hnxyt]
        double *RESTRICT eint,  // [Hnxystep][Hnxyt]
        double *RESTRICT p,     // [Hnxystep][Hnxyt]
        double *RESTRICT c      // [Hnxystep][Hnxyt]
  ) {
  double smallp;
  int i, j;
  double pIJ, rhoIJ;
  idx2d(i, j, Hnxyt);
  if (j >= slices)
    return;
  if (i < imin) return;
  if (i >= imax) return;

  smallp = Square(Hsmallc) / Hgamma;
  rhoIJ = rho[IHS(i,j)];
  pIJ = (Hgamma - one) * rhoIJ * eint[IHS(i,j)];
  pIJ = MAX(pIJ, (double) (rhoIJ * smallp));
  c[IHS(i,j)] = sqrt(Hgamma * pIJ / rhoIJ);
  p[IHS(i,j)] = pIJ;
}

void
cuEquationOfState(const long imin, const long imax, //
		  const double Hsmallc, const double Hgamma, //
		  const long Hnxyt, const long slices, //
                  double *RESTRICT rhoDEV,      // [Hnxystep][Hnxyt]
                  double *RESTRICT eintDEV,     // [Hnxystep][Hnxyt]
                  double *RESTRICT pDEV,        // [Hnxystep][Hnxyt]
                  double *RESTRICT cDEV         // [Hnxystep][Hnxyt]
  ) {
  dim3 grid, block;
  int nops;
  WHERE("equation_of_state");
  SetBlockDims(Hnxyt * slices, THREADSSZ, block, grid);
  LoopEOS <<< grid, block >>> (imin, imax, Hsmallc, Hgamma, Hnxyt, slices, rhoDEV, eintDEV, pDEV, cDEV);
  CheckErr("LoopEOS");
  cudaThreadSynchronize();
  CheckErr("LoopEOS");
  nops = slices * (imax - imin);
  FLOPS(5 * nops, 2 * nops, 1 * nops, 0 * nops);}                               // equation_of_state


// EOF
