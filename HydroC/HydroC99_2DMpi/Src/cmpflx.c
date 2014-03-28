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

#ifndef HMPP
#include "parametres.h"
#include "utils.h"
#include "cmpflx.h"
#include "perfcnt.h"

void
cmpflx(const int narray,
       const int Hnxyt,
       const int Hnvar,
       const real_t Hgamma,
       const int slices, 
       const int Hstep, 
       real_t qgdnv[Hnvar][Hstep][Hnxyt], 
       real_t flux[Hnvar][Hstep][Hnxyt]) {
  int nface, i, IN;
  real_t entho, ekin, etot;
  WHERE("cmpflx");
  int s;

  nface = narray;
  entho = one / (Hgamma - one);
  FLOPS(1, 1, 0, 0);

  // Compute fluxes
#pragma omp parallel for private(s, i, ekin, etot), shared(flux) 
  for (s = 0; s < slices; s++) {
    for (i = 0; i < nface; i++) {
      real_t qgdnvID = qgdnv[ID][s][i];
      real_t qgdnvIU = qgdnv[IU][s][i];
      real_t qgdnvIP = qgdnv[IP][s][i];
      real_t qgdnvIV = qgdnv[IV][s][i];

      // Mass density
      real_t massDensity = qgdnvID * qgdnvIU;
      flux[ID][s][i] = massDensity;

      // Normal momentum
      flux[IU][s][i] = massDensity * qgdnvIU + qgdnvIP;
      // Transverse momentum 1
      flux[IV][s][i] = massDensity * qgdnvIV;

      // Total energy
      ekin = half * qgdnvID * (Square(qgdnvIU) + Square(qgdnvIV));
      etot = qgdnvIP * entho + ekin;

      flux[IP][s][i] = qgdnvIU * (etot + qgdnvIP);
    }
  }

  { 
    int nops = slices * nface;
    FLOPS(13 * nops, 0 * nops, 0 * nops, 0 * nops);
  }


  // Other advected quantities
  if (Hnvar > IP) {
    for (s = 0; s < slices; s++) {
      for (IN = IP + 1; IN < Hnvar; IN++) {
        for (i = 0; i < nface; i++) {
          flux[IN][s][i] = flux[IN][s][i] * qgdnv[IN][s][i];
        }
      }
    }
  }
}                               // cmpflx

#endif

//EOF
