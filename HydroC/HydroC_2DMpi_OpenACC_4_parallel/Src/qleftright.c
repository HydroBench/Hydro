/*
  A simple 2D hydro code
  (C) Romain Teyssier : CEA/IRFU           -- original F90 code
  (C) Pierre-Francois Lavallee : IDRIS      -- original F90 code
  (C) Guillaume Colin de Verdiere : CEA/DAM -- for the C version
  (C) Jeffrey Poznanovic : CSCS             -- for the OpenACC version
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
#include "qleftright.h"

#ifndef HMPP

#define IDX(i,j,k)    ( (i*Hstep*Hnxyt) + (j*Hnxyt) + k )

void
// qleftright(const int idim, const hydroparam_t H, hydrovarwork_t * Hvw)
qleftright(const int idim,
           const int Hnx,
           const int Hny,
           const int Hnxyt,
           const int Hnvar,
           const int slices, const int Hstep,
	   double *qxm,
	   double *qxp, double *qleft, double *qright) {
           //double qxm[Hnvar][Hstep][Hnxyt],
           //double qxp[Hnvar][Hstep][Hnxyt], double qleft[Hnvar][Hstep][Hnxyt], double qright[Hnvar][Hstep][Hnxyt]) {
  // #define IHVW(i,v) ((i) + (v) * Hnxyt)
  int nvar, i, s;
  int bmax;
  WHERE("qleftright");
  if (idim == 1) {
    bmax = Hnx + 1;
  } else {
    bmax = Hny + 1;
  }

#pragma acc parallel pcopyin(qxm[0:Hnvar*Hstep*Hnxyt], qxp[0:Hnvar*Hstep*Hnxyt]) pcopyout(qleft[0:Hnvar*Hstep*Hnxyt], qright[0:Hnvar*Hstep*Hnxyt]) 
#pragma acc loop gang collapse(2)
  for (nvar = 0; nvar < Hnvar; nvar++) {
    for (s = 0; s < slices; s++) {
#pragma acc loop vector
      for (i = 0; i < bmax; i++) {
        qleft[IDX(nvar,s,i)] = qxm[IDX(nvar,s,i + 1)];
        qright[IDX(nvar,s,i)] = qxp[IDX(nvar,s,i + 2)];
      }
    }
  }
}

#undef IHVW
#undef IDX

#endif /* HMPP */
// EOF
