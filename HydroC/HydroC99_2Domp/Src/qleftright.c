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
#include "qleftright.h"

#ifndef HMPP

void
// qleftright(const int idim, const hydroparam_t H, hydrovarwork_t * Hvw)
qleftright(const int idim,
           const int Hnx,
           const int Hny,
           const int Hnxyt,
           const int Hnvar,
           const int slices, const int Hstep,
           real_t qxm[Hnvar][Hstep][Hnxyt],
           real_t qxp[Hnvar][Hstep][Hnxyt],
	   real_t qleft[Hnvar][Hstep][Hnxyt],
           real_t qright[Hnvar][Hstep][Hnxyt]) {
  // #define IHVW(i,v) ((i) + (v) * Hnxyt)
  int nvar, i, s;
  int bmax;
  WHERE("qleftright");
  if (idim == 1) {
    bmax = Hnx + 1;
  } else {
    bmax = Hny + 1;
  }

#ifdef TARGETON
#pragma message "TARGET on QLEFTRIGHT"
#pragma omp target				\
	map(to:qxm[0:Hnvar][0:Hstep][0:bmax])	\
	map(to:qxp[0:Hnvar][0:Hstep][0:bmax])	\
	map(from:qleft[0:Hnvar][0:Hstep][0:bmax])	\
	map(from:qright[0:Hnvar][0:Hstep][0:bmax])
#pragma omp teams distribute parallel for default(none) private(s, i, nvar), shared(qleft, qright, qxm, qxp)  collapse(3)
#else
#pragma omp parallel for private(nvar, i, s), shared(qleft, qright) 
#endif
  for (s = 0; s < slices; s++) {
    for (nvar = 0; nvar < Hnvar; nvar++) {
// #pragma omp simd
      for (i = 0; i < bmax; i++) {
        qleft[nvar][s][i] = qxm[nvar][s][i + 1];
        qright[nvar][s][i] = qxp[nvar][s][i + 2];
      }
    }
  }
}

#undef IHVW

#endif /* HMPP */
// EOF
