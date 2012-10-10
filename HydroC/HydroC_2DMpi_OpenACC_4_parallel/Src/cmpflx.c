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

#define CFLOPS(c)               /* {flops+=c;} */

void
cmpflx(const int narray,
       const int Hnxyt,
       const int Hnvar,
       const double Hgamma,
       const int slices, const int Hstep, double *qgdnv, double *flux) {
  //       const int slices, const int Hstep, double qgdnv[Hnvar][Hstep][Hnxyt], double flux[Hnvar][Hstep][Hnxyt]) {
  int nface, i, IN;
  double entho, ekin, etot;
  WHERE("cmpflx");
  int s;

  nface = narray;
  entho = one / (Hgamma - one);

  #define IDX(i,j,k) ( (i*Hstep*Hnxyt) + (j*Hnxyt) + k )

  // Compute fluxes
#pragma acc parallel pcopyin(qgdnv[0:Hnvar*Hstep*Hnxyt]) pcopyout(flux[0:Hnvar*Hstep*Hnxyt])
#pragma acc loop gang
  for (s = 0; s < slices; s++) {
#pragma acc loop vector
    for (i = 0; i < nface; i++) {
      double qgdnvID = qgdnv[IDX(ID,s,i)]; 
      double qgdnvIU = qgdnv[IDX(IU,s,i)];
      double qgdnvIP = qgdnv[IDX(IP,s,i)]; 
      double qgdnvIV = qgdnv[IDX(IV,s,i)];

      // Mass density
      double massDensity = qgdnvID * qgdnvIU;
      flux[IDX(ID,s,i)] = massDensity; 

      // Normal momentum
      flux[IDX(IU,s,i)] = massDensity * qgdnvIU + qgdnvIP;
      // Transverse momentum 1
      flux[IDX(IV,s,i)] = massDensity * qgdnvIV; 

      // Total energy
      ekin = half * qgdnvID * (Square(qgdnvIU) + Square(qgdnvIV));
      etot = qgdnvIP * entho + ekin;

      flux[IDX(IP,s,i)] = qgdnvIU * (etot + qgdnvIP); 

      CFLOPS(15);
    }
  }

  // Other advected quantities
  if (Hnvar > IP) {
#pragma acc parallel pcopyin(qgdnv[0:Hnvar*Hstep*Hnxyt]) pcopy(flux[0:Hnvar*Hstep*Hnxyt])
#pragma acc loop gang collapse(2)
    for (s = 0; s < slices; s++) {
      for (IN = IP + 1; IN < Hnvar; IN++) {
#pragma acc loop vector
        for (i = 0; i < nface; i++) {
          flux[IDX(IN,s,i)] = flux[IDX(IN,s,i)] * qgdnv[IDX(IN,s,i)];
        }
      }
    }
  }
}                               // cmpflx

#undef IDX

#endif

//EOF
