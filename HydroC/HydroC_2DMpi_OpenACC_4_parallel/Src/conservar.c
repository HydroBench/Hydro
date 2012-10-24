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

#ifndef HMPP
#include "parametres.h"
#include "utils.h"
#include "conservar.h"

#define CFLOPS(c)               /* {flops+=c;} */

void
gatherConservativeVars(const int idim,
                       const int rowcol,
                       const int Himin,
                       const int Himax,
                       const int Hjmin,
                       const int Hjmax,
                       const int Hnvar,
                       const int Hnxt,
                       const int Hnyt,
                       const int Hnxyt,
                       const int slices, const int Hstep,
                       double uold[Hnvar * Hnxt * Hnyt], double *u
                       //double uold[Hnvar * Hnxt * Hnyt], double u[Hnvar][Hstep][Hnxyt]
  ) {
  int i, j, ivar, s;

#define IHU(i, j, v)  ((i) + Hnxt * ((j) + Hnyt * (v)))
#define IDX(i,j,k)    ( (i*Hstep*Hnxyt) + (j*Hnxyt) + k )

  WHERE("gatherConservativeVars");
  if (idim == 1) {
    // Gather conservative variables
#pragma acc parallel pcopyin(uold[0:Hnxt*Hnyt*Hnvar]) pcopyout(u[0:Hnvar*Hstep*Hnxyt]) 
#pragma acc loop gang
    for (s = 0; s < slices; s++) {
#pragma acc loop vector
      for (i = Himin; i < Himax; i++) {
        int idxuoID = IHU(i, rowcol + s, ID);
        u[IDX(ID,s,i)] = uold[idxuoID];

        int idxuoIU = IHU(i, rowcol + s, IU);
        u[IDX(IU,s,i)] = uold[idxuoIU];

        int idxuoIV = IHU(i, rowcol + s, IV);
        u[IDX(IV,s,i)] = uold[idxuoIV];

        int idxuoIP = IHU(i, rowcol + s, IP);
        u[IDX(IP,s,i)] = uold[idxuoIP];
      }
    }

    if (Hnvar > IP) {
#pragma acc parallel pcopyin(uold[0:Hnxt*Hnyt*Hnvar]) pcopyout(u[0:Hnvar*Hstep*Hnxyt]) 
#pragma acc loop gang collapse(2)
      for (ivar = IP + 1; ivar < Hnvar; ivar++) {
        for (s = 0; s < slices; s++) {
#pragma acc loop vector
          for (i = Himin; i < Himax; i++) {
            u[IDX(ivar,s,i)] = uold[IHU(i, rowcol + s, ivar)];
          }
        }
      }
    }
    //
  } else {
    // Gather conservative variables
#pragma acc parallel pcopyin(uold[0:Hnxt*Hnyt*Hnvar]) pcopyout(u[0:Hnvar*Hstep*Hnxyt]) 
#pragma acc loop gang
    for (s = 0; s < slices; s++) {
#pragma acc loop vector
      for (j = Hjmin; j < Hjmax; j++) {
        u[IDX(ID,s,j)] = uold[IHU(rowcol + s, j, ID)];
        u[IDX(IU,s,j)] = uold[IHU(rowcol + s, j, IV)];
        u[IDX(IV,s,j)] = uold[IHU(rowcol + s, j, IU)];
        u[IDX(IP,s,j)] = uold[IHU(rowcol + s, j, IP)];
      }
    }
    if (Hnvar > IP) {
#pragma acc parallel pcopyin(uold[0:Hnxt*Hnyt*Hnvar]) pcopyout(u[0:Hnvar*Hstep*Hnxyt]) 
#pragma acc loop gang collapse(2)
      for (ivar = IP + 1; ivar < Hnvar; ivar++) {
        for (s = 0; s < slices; s++) {
#pragma acc loop vector
          for (j = Hjmin; j < Hjmax; j++) {
            u[IDX(ivar,s,j)] = uold[IHU(rowcol + s, j, ivar)];
          }
        }
      }
    }
  }
}

#undef IHU
#undef IDX

void
updateConservativeVars(const int idim,
                       const int rowcol,
                       const double dtdx,
                       const int Himin,
                       const int Himax,
                       const int Hjmin,
                       const int Hjmax,
                       const int Hnvar,
                       const int Hnxt,
                       const int Hnyt,
                       const int Hnxyt,
                       const int slices, const int Hstep,
                       double uold[Hnvar * Hnxt * Hnyt], double *u, double *flux
                       //double uold[Hnvar * Hnxt * Hnyt], double u[Hnvar][Hstep][Hnxyt], double flux[Hnvar][Hstep][Hnxyt]
  ) {
  int i, j, ivar, s;
  WHERE("updateConservativeVars");

#define IHU(i, j, v)  ((i) + Hnxt * ((j) + Hnyt * (v)))
#define IDX(i,j,k)    ( (i*Hstep*Hnxyt) + (j*Hnxyt) + k )

  if (idim == 1) {

    // Update conservative variables
#pragma acc parallel pcopyin(u[0:Hnvar*Hstep*Hnxyt], flux[0:Hnvar*Hstep*Hnxyt]) pcopy(uold[0:Hnxt*Hnyt*Hnvar])
#pragma acc loop gang collapse(2)
    for (ivar = 0; ivar <= IP; ivar++) {
      for (s = 0; s < slices; s++) {
#pragma acc loop vector
        for (i = Himin + ExtraLayer; i < Himax - ExtraLayer; i++) {
          uold[IHU(i, rowcol + s, ivar)] = u[IDX(ivar,s,i)] + (flux[IDX(ivar,s,i - 2)] - flux[IDX(ivar,s,i - 1)]) * dtdx;
          CFLOPS(3);
        }
      }
    }

    if (Hnvar > IP) {
#pragma acc parallel pcopyin(u[0:Hnvar*Hstep*Hnxyt], flux[0:Hnvar*Hstep*Hnxyt]) pcopy(uold[0:Hnxt*Hnyt*Hnvar])
#pragma acc loop gang collapse(2)
      for (ivar = IP + 1; ivar < Hnvar; ivar++) {
        for (s = 0; s < slices; s++) {
#pragma acc loop vector
          for (i = Himin + ExtraLayer; i < Himax - ExtraLayer; i++) {
            uold[IHU(i, rowcol + s, ivar)] = u[IDX(ivar,s,i)] + (flux[IDX(ivar,s,i - 2)] - flux[IDX(ivar,s,i - 1)]) * dtdx;
            CFLOPS(3);
          }
        }
      }
    }
  } else {
    // Update conservative variables
#pragma acc parallel pcopyin(u[0:Hnvar*Hstep*Hnxyt], flux[0:Hnvar*Hstep*Hnxyt]) pcopy(uold[0:Hnxt*Hnyt*Hnvar])
#pragma acc loop gang
    for (s = 0; s < slices; s++) {
#pragma acc loop vector
      for (j = Hjmin + ExtraLayer; j < Hjmax - ExtraLayer; j++) {
        uold[IHU(rowcol + s, j, ID)] = u[IDX(ID,s,j)] + (flux[IDX(ID,s,j - 2)] - flux[IDX(ID,s,j - 1)]) * dtdx;
        CFLOPS(3);

        uold[IHU(rowcol + s, j, IV)] = u[IDX(IU,s,j)] + (flux[IDX(IU,s,j - 2)] - flux[IDX(IU,s,j - 1)]) * dtdx;
        CFLOPS(3);

        uold[IHU(rowcol + s, j, IU)] = u[IDX(IV,s,j)] + (flux[IDX(IV,s,j - 2)] - flux[IDX(IV,s,j - 1)]) * dtdx;
        CFLOPS(3);

        uold[IHU(rowcol + s, j, IP)] = u[IDX(IP,s,j)] + (flux[IDX(IP,s,j - 2)] - flux[IDX(IP,s,j - 1)]) * dtdx;
        CFLOPS(3);
      }
    }

    if (Hnvar > IP) {
#pragma acc parallel pcopyin(u[0:Hnvar*Hstep*Hnxyt], flux[0:Hnvar*Hstep*Hnxyt]) pcopy(uold[0:Hnxt*Hnyt*Hnvar])
#pragma acc loop gang collapse(2)
      for (ivar = IP + 1; ivar < Hnvar; ivar++) {
        for (s = 0; s < slices; s++) {
#pragma acc loop vector
          for (j = Hjmin + ExtraLayer; j < Hjmax - ExtraLayer; j++) {
            uold[IHU(rowcol + s, j, ivar)] = u[IDX(ivar,s,j)] + (flux[IDX(ivar,s,j - 2)] - flux[IDX(ivar,s,j - 1)]) * dtdx;
            CFLOPS(3);
          }
        }
      }
    }
  }
}

#undef IHU
#undef IDX
#endif
//EOF
