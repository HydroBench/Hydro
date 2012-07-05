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
#include "gridfuncs.h"
#include "cuQleftright.h"

__global__ void
Loop1KcuQleftright(const long bmax, const long Hnvar, const long Hnxyt, const int slices,  const int Hnxystep,      //
                   double *RESTRICT qxm, double *RESTRICT qxp, double *RESTRICT qleft, double *RESTRICT qright) {
  int nvar;
  int i, j;
  idx2d(i, j, Hnxyt);
  if (j >= slices)
    return;
  if (i >= bmax)
    return;

  for (nvar = 0; nvar < Hnvar; nvar++) {
    qleft[IHVWS(i, j, nvar)] = qxm[IHVWS(i + 1, j, nvar)];
    qright[IHVWS(i, j, nvar)] = qxp[IHVWS(i + 2, j, nvar)];
  }
}

void
cuQleftright(const long idim, const long Hnx, const long Hny, const long Hnxyt, const long Hnvar, const int slices,  const int Hnxystep,      // 
             double *RESTRICT qxmDEV,   // [Hnvar][Hnxystep][Hnxyt]
             double *RESTRICT qxpDEV,   // [Hnvar][Hnxystep][Hnxyt]
             double *RESTRICT qleftDEV, // [Hnvar][Hnxystep][Hnxyt]
             double *RESTRICT qrightDEV // [Hnvar][Hnxystep][Hnxyt]
  ) {
  int bmax;
  dim3 block, grid;
  WHERE("qleftright");
  if (idim == 1) {
    bmax = Hnx + 1;
  } else {
    bmax = Hny + 1;
  }
  SetBlockDims(Hnxyt * slices, THREADSSZ, block, grid);
  Loop1KcuQleftright <<< grid, block >>> (bmax, Hnvar, Hnxyt, slices, Hnxystep, qxmDEV, qxpDEV, qleftDEV, qrightDEV);
  CheckErr("Loop1KcuQleftright");
  cudaThreadSynchronize();
}

#undef IHVW

// EOF
