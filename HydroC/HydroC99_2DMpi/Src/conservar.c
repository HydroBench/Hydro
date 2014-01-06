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

#ifndef HMPP
#include "parametres.h"
#include "utils.h"
#include "conservar.h"
#include "perfcnt.h"

#define BLOCKING 0
#define SSST 32
#define JJST 32

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
                       real_t uold[Hnvar * Hnxt * Hnyt], real_t u[Hnvar][Hstep][Hnxyt]
		       ) {
  int i, j, ivar, s;

#define IHU(i, j, v)  ((i) + Hnxt  * ((j) + Hnyt  * (v)))
#define IHST(v,s,i)   ((i) + Hstep * ((j) + Hnvar * (v)))

  WHERE("gatherConservativeVars");
  if (idim == 1) {
    // Gather conservative variables
#pragma omp parallel for private(i, s), shared(u) COLLAPSE
    for (s = 0; s < slices; s++) {
      for (i = Himin; i < Himax; i++) {
        int idxuoID = IHU(i, rowcol + s, ID);
        u[ID][s][i] = uold[idxuoID];

        int idxuoIU = IHU(i, rowcol + s, IU);
        u[IU][s][i] = uold[idxuoIU];

        int idxuoIV = IHU(i, rowcol + s, IV);
        u[IV][s][i] = uold[idxuoIV];

        int idxuoIP = IHU(i, rowcol + s, IP);
        u[IP][s][i] = uold[idxuoIP];
      }
    }

    if (Hnvar > IP) {
      for (ivar = IP + 1; ivar < Hnvar; ivar++) {
        for (s = 0; s < slices; s++) {
          for (i = Himin; i < Himax; i++) {
            u[ivar][s][i] = uold[IHU(i, rowcol + s, ivar)];
          }
        }
      }
    }
    //
  } else {
    // Gather conservative variables
#pragma omp parallel for private(j, s), shared(u) 
    for (s = 0; s < slices; s++) {
      for (j = Hjmin; j < Hjmax; j++) {
	u[ID][s][j] = uold[IHU(rowcol + s, j, ID)];
	u[IU][s][j] = uold[IHU(rowcol + s, j, IV)];
	u[IV][s][j] = uold[IHU(rowcol + s, j, IU)];
	u[IP][s][j] = uold[IHU(rowcol + s, j, IP)];
      }
    }

    if (Hnvar > IP) {
      for (ivar = IP + 1; ivar < Hnvar; ivar++) {
	for (s = 0; s < slices; s++) {
	  for (j = Hjmin; j < Hjmax; j++) {
	    u[ivar][s][j] = uold[IHU(rowcol + s, j, ivar)];
	  }
	}
      }
    }
  }
}

#undef IHU

void
updateConservativeVars(const int idim,
                       const int rowcol,
                       const real_t dtdx,
                       const int Himin,
                       const int Himax,
                       const int Hjmin,
                       const int Hjmax,
                       const int Hnvar,
                       const int Hnxt,
                       const int Hnyt,
                       const int Hnxyt,
                       const int slices, const int Hstep,
                       real_t uold[Hnvar * Hnxt * Hnyt], real_t u[Hnvar][Hstep][Hnxyt], real_t flux[Hnvar][Hstep][Hnxyt]
		       ) {
  int i, j, ivar, s;
  WHERE("updateConservativeVars");

#define IHU(i, j, v)  ((i) + Hnxt * ((j) + Hnyt * (v)))

  if (idim == 1) {

    // Update conservative variables
#pragma omp parallel for private(ivar, s,i), shared(uold) COLLAPSE
    for (s = 0; s < slices; s++) {
      for (ivar = 0; ivar <= IP; ivar++) {
	for (i = Himin + ExtraLayer; i < Himax - ExtraLayer; i++) {
	  uold[IHU(i, rowcol + s, ivar)] = u[ivar][s][i] + (flux[ivar][s][i - 2] - flux[ivar][s][i - 1]) * dtdx;
        }
      }
    }
    { 
      int nops = (IP+1) * slices * ((Himax - ExtraLayer) - (Himin + ExtraLayer));
      FLOPS(3 * nops, 0 * nops, 0 * nops, 0 * nops);
    }

    if (Hnvar > IP) {
      for (ivar = IP + 1; ivar < Hnvar; ivar++) {
        for (s = 0; s < slices; s++) {
          for (i = Himin + ExtraLayer; i < Himax - ExtraLayer; i++) {
            uold[IHU(i, rowcol + s, ivar)] = u[ivar][s][i] + (flux[ivar][s][i - 2] - flux[ivar][s][i - 1]) * dtdx;
          }
        }
      }
    }
  } else {
    // Update conservative variables
#pragma omp parallel for private(j, s), shared(uold) 
    for (s = 0; s < slices; s++) {
      for (j = (Hjmin + ExtraLayer); j < (Hjmax - ExtraLayer); j++) {
	uold[IHU(rowcol + s, j, ID)] = u[ID][s][j] + (flux[ID][s][j - 2] - flux[ID][s][j - 1]) * dtdx;
	uold[IHU(rowcol + s, j, IV)] = u[IU][s][j] + (flux[IU][s][j - 2] - flux[IU][s][j - 1]) * dtdx;
	uold[IHU(rowcol + s, j, IU)] = u[IV][s][j] + (flux[IV][s][j - 2] - flux[IV][s][j - 1]) * dtdx;
	uold[IHU(rowcol + s, j, IP)] = u[IP][s][j] + (flux[IP][s][j - 2] - flux[IP][s][j - 1]) * dtdx;
      }
    }
    { 
      int nops = slices * ((Hjmax - ExtraLayer) - (Hjmin + ExtraLayer));
      FLOPS(12 * nops, 0 * nops, 0 * nops, 0 * nops);
    }

    if (Hnvar > IP) {
      for (ivar = IP + 1; ivar < Hnvar; ivar++) {
	for (s = 0; s < slices; s++) {
	  for (j = Hjmin + ExtraLayer; j < Hjmax - ExtraLayer; j++) {
	    uold[IHU(rowcol + s, j, ivar)] = u[ivar][s][j] + (flux[ivar][s][j - 2] - flux[ivar][s][j - 1]) * dtdx;
	  }
	}
      }
    }
  }
}

#undef IHU
#endif
//EOF
