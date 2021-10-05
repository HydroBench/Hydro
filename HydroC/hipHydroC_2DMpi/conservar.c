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
#include "conservar.h"
void
gatherConservativeVars(const long idim, const long rowcol,
                       double *RESTRICT uold,
                       double *RESTRICT u,
                       const long Himin,
                       const long Himax,
                       const long Hjmin, const long Hjmax, const long Hnvar, const long Hnxt, const long Hnyt, const long Hnxyt)
{
  long i, j, ivar;

#define IHU(i, j, v)  ((i) + Hnxt * ((j) + Hnyt * (v)))
#define IHVW(i, v) ((i) + (v) * Hnxyt)

  WHERE("gatherConservativeVars");
  if (idim == 1) {
    // Gather conservative variables
    for (i = Himin; i < Himax; i++) {
      long idxuoID = IHU(i, rowcol, ID);
      long idxuoIP = IHU(i, rowcol, IP);
      long idxuoIV = IHU(i, rowcol, IV);
      long idxuoIU = IHU(i, rowcol, IU);
      u[IHVW(i, ID)] = uold[idxuoID];
      // }

      // for (i = Himin; i < Himax; i++) {
      u[IHVW(i, IU)] = uold[idxuoIU];
      // }

      // for (i = Himin; i < Himax; i++) {
      u[IHVW(i, IV)] = uold[idxuoIV];
      // }

      // for (i = Himin; i < Himax; i++) {
      u[IHVW(i, IP)] = uold[idxuoIP];
    }

    if (Hnvar > IP + 1) {
      for (ivar = IP + 1; ivar < Hnvar; ivar++) {
        for (i = Himin; i < Himax; i++) {
          u[IHVW(i, ivar)] = uold[IHU(i, rowcol, ivar)];
        }
      }
    }
  } else {
    // Gather conservative variables
    for (j = Hjmin; j < Hjmax; j++) {
      u[IHVW(j, ID)] = uold[IHU(rowcol, j, ID)];
      // }
      // for (j = Hjmin; j < Hjmax; j++) {
      u[IHVW(j, IU)] = uold[IHU(rowcol, j, IV)];
      // }
      // for (j = Hjmin; j < Hjmax; j++) {
      u[IHVW(j, IV)] = uold[IHU(rowcol, j, IU)];
      // }
      // for (j = Hjmin; j < Hjmax; j++) {
      u[IHVW(j, IP)] = uold[IHU(rowcol, j, IP)];
    }
    if (Hnvar > IP + 1) {
      for (ivar = IP + 1; ivar < Hnvar; ivar++) {
        for (j = Hjmin; j < Hjmax; j++) {
          u[IHVW(j, ivar)] = uold[IHU(rowcol, j, ivar)];
        }
      }
    }
  }
}

#undef IHVW
#undef IHU

void
updateConservativeVars(const long idim, const long rowcol, const double dtdx,
                       double *RESTRICT uold,
                       double *RESTRICT u,
                       double *RESTRICT flux,
                       const long Himin,
                       const long Himax,
                       const long Hjmin, const long Hjmax, const long Hnvar, const long Hnxt, const long Hnyt, const long Hnxyt)
{
  long i, j, ivar;
  WHERE("updateConservativeVars");

#define IHU(i, j, v)  ((i) + Hnxt * ((j) + Hnyt * (v)))
#define IHVW(i, v) ((i) + (v) * Hnxyt)

  if (idim == 1) {
    // Update conservative variables
    // #define NEW 1
#ifdef NEW
#define BLOCKSIZE 128
    long trip = (Himax - 2) - (Himin + 2);
    long reste = trip - trip / BLOCKSIZE;
    long t;
// taille des tableaux auxiliaires 
// le dernier +1 est pour eviter les conflits de bancs
#define AIDS BLOCKSIZE+1+1
#define UIDS BLOCKSIZE
// definition des offsets dans le tableau en shared
#define AIDO 0
#define UIDO AIDO+AIDS
#define AIUO UIDO+UIDS
#define UIUO AIUO+AIDS
#define AIVO UIUO+UIDS
#define UIVO AIVO+AIDS
#define AIPO UIVO+UIDS
#define UIPO AIPO+AIDS
// taille totale de la memoire shared necessaire
#define CSHS UIPO+UIDS
    double cshared[CSHS];
    double *aid = &cshared[AIDO], *uid = &cshared[UIDO], *aiu = &cshared[AIUO], *uiu = &cshared[UIUO];
    double *aiv = &cshared[AIVO], *uiv = &cshared[UIVO], *aip = &cshared[AIPO], *uip = &cshared[UIPO];
    for (i = Himin + ExtraLayer; i < Himax - ExtraLayer - reste; i += BLOCKSIZE) {
      for (t = 0; t < BLOCKSIZE; t++) {
        aid[t] = (flux[IHVW(i + t - 2, ID)]);
        uid[t] = u[IHVW(i + t, ID)];
        aiu[t] = (flux[IHVW(i + t - 2, IU)]);
        uiu[t] = u[IHVW(i + t, IU)];
        aiv[t] = (flux[IHVW(i + t - 2, IV)]);
        uiv[t] = u[IHVW(i + t, IV)];
        aip[t] = (flux[IHVW(i + t - 2, IP)]);
        uip[t] = u[IHVW(i + t, IP)];
        if (t == BLOCKSIZE - 1) {
          aid[t + 1] = (flux[IHVW(i + t - 1, ID)]);
          aiu[t + 1] = (flux[IHVW(i + t - 1, IU)]);
          aiv[t + 1] = (flux[IHVW(i + t - 1, IV)]);
          aip[t + 1] = (flux[IHVW(i + t - 1, IP)]);
        }
        uold[IHU(i + t, rowcol, ID)] = uid[t] + (aid[t] - aid[t + 1]) * dtdx;
        uold[IHU(i + t, rowcol, IU)] = uiu[t] + (aiu[t] - aiu[t + 1]) * dtdx;
        uold[IHU(i + t, rowcol, IV)] = uiv[t] + (aiv[t] - aiv[t + 1]) * dtdx;
        uold[IHU(i + t, rowcol, IP)] = uip[t] + (aip[t] - aip[t + 1]) * dtdx;
        MFLOPS(12, 0, 0, 0);
      }
    }
    if (reste > 0) {
      for (i = Himax - ExtraLayer - reste; i < Himax - ExtraLayer; i++) {
        uold[IHU(i, rowcol, ID)] = u[IHVW(i, ID)] + (flux[IHVW(i - 2, ID)] - flux[IHVW(i - 1, ID)]) * dtdx;
        uold[IHU(i, rowcol, IU)] = u[IHVW(i, IU)] + (flux[IHVW(i - 2, IU)] - flux[IHVW(i - 1, IU)]) * dtdx;
        uold[IHU(i, rowcol, IV)] = u[IHVW(i, IV)] + (flux[IHVW(i - 2, IV)] - flux[IHVW(i - 1, IV)]) * dtdx;
        uold[IHU(i, rowcol, IP)] = u[IHVW(i, IP)] + (flux[IHVW(i - 2, IP)] - flux[IHVW(i - 1, IP)]) * dtdx;
        MFLOPS(12, 0, 0, 0);
      }
    }
#else
    for (i = Himin + ExtraLayer; i < Himax - ExtraLayer; i++) {
      uold[IHU(i, rowcol, ID)] = u[IHVW(i, ID)] + (flux[IHVW(i - 2, ID)] - flux[IHVW(i - 1, ID)]) * dtdx;
      uold[IHU(i, rowcol, IU)] = u[IHVW(i, IU)] + (flux[IHVW(i - 2, IU)] - flux[IHVW(i - 1, IU)]) * dtdx;
      uold[IHU(i, rowcol, IV)] = u[IHVW(i, IV)] + (flux[IHVW(i - 2, IV)] - flux[IHVW(i - 1, IV)]) * dtdx;
      uold[IHU(i, rowcol, IP)] = u[IHVW(i, IP)] + (flux[IHVW(i - 2, IP)] - flux[IHVW(i - 1, IP)]) * dtdx;
      MFLOPS(12, 0, 0, 0);
    }
#endif
    if (Hnvar > IP + 1) {
      for (ivar = IP + 1; ivar < Hnvar; ivar++) {
        for (i = Himin + ExtraLayer; i < Himax - ExtraLayer; i++) {
          uold[IHU(i, rowcol, ivar)] = u[IHVW(i, ivar)] + (flux[IHVW(i - 2, ivar)] - flux[IHVW(i - 1, ivar)]) * dtdx;
          MFLOPS(3, 0, 0, 0);
        }
      }
    }
  } else {
    // Update conservative variables
#ifdef NEW
    long trip = (Hjmax - 2) - (Hjmin + 2);
    long reste = trip - trip / BLOCKSIZE;
    long t;
// taille des tableaux auxiliaires 
// le dernier +1 est pour eviter les conflits de bancs
#define AIDS BLOCKSIZE+1+1
#define UIDS BLOCKSIZE
// definition des offsets dans le tableau en shared
#define AIDO 0
#define UIDO AIDO+AIDS
#define AIUO UIDO+UIDS
#define UIUO AIUO+AIDS
#define AIVO UIUO+UIDS
#define UIVO AIVO+AIDS
#define AIPO UIVO+UIDS
#define UIPO AIPO+AIDS
// taille totale de la memoire shared necessaire
#define CSHS UIPO+UIDS
    double cshared[CSHS];
    double *aid = &cshared[AIDO], *uid = &cshared[UIDO], *aiu = &cshared[AIUO], *uiu = &cshared[UIUO];
    double *aiv = &cshared[AIVO], *uiv = &cshared[UIVO], *aip = &cshared[AIPO], *uip = &cshared[UIPO];
    for (j = Hjmin + ExtraLayer; j < Hjmax - ExtraLayer - reste; j += BLOCKSIZE) {
      for (t = 0; t < BLOCKSIZE; t++) {
        aid[t] = (flux[IHVW(j + t - 2, ID)]);
        uid[t] = u[IHVW(j + t, ID)];
        aiu[t] = (flux[IHVW(j + t - 2, IU)]);
        uiu[t] = u[IHVW(j + t, IU)];
        aiv[t] = (flux[IHVW(j + t - 2, IV)]);
        uiv[t] = u[IHVW(j + t, IV)];
        aip[t] = (flux[IHVW(j + t - 2, IP)]);
        uip[t] = u[IHVW(j + t, IP)];
        if (t == BLOCKSIZE - 1) {
          aid[t + 1] = (flux[IHVW(j + t - 1, ID)]);
          aiu[t + 1] = (flux[IHVW(j + t - 1, IU)]);
          aiv[t + 1] = (flux[IHVW(j + t - 1, IV)]);
          aip[t + 1] = (flux[IHVW(j + t - 1, IP)]);
        }
        uold[IHU(rowcol, j + t, ID)] = uid[t] + (aid[t] - aid[t + 1]) * dtdx;
        uold[IHU(rowcol, j + t, IV)] = uiu[t] + (aiu[t] - aiu[t + 1]) * dtdx;
        uold[IHU(rowcol, j + t, IU)] = uiv[t] + (aiv[t] - aiv[t + 1]) * dtdx;
        uold[IHU(rowcol, j + t, IP)] = uip[t] + (aip[t] - aip[t + 1]) * dtdx;
        MFLOPS(12, 0, 0, 0);
      }
    }
    if (reste > 0) {
      for (j = Hjmax - ExtraLayer - reste; j < Hjmax - ExtraLayer; j++) {
        uold[IHU(rowcol, j, ID)] = u[IHVW(j, ID)] + (flux[IHVW(j - 2, ID)] - flux[IHVW(j - 1, ID)]) * dtdx;
        uold[IHU(rowcol, j, IV)] = u[IHVW(j, IU)] + (flux[IHVW(j - 2, IU)] - flux[IHVW(j - 1, IU)]) * dtdx;
        uold[IHU(rowcol, j, IU)] = u[IHVW(j, IV)] + (flux[IHVW(j - 2, IV)] - flux[IHVW(j - 1, IV)]) * dtdx;
        uold[IHU(rowcol, j, IP)] = u[IHVW(j, IP)] + (flux[IHVW(j - 2, IP)] - flux[IHVW(j - 1, IP)]) * dtdx;
        MFLOPS(12, 0, 0, 0);
      }
    }
#else
    for (j = Hjmin + ExtraLayer; j < Hjmax - ExtraLayer; j++) {
      uold[IHU(rowcol, j, ID)] = u[IHVW(j, ID)] + (flux[IHVW(j - 2, ID)] - flux[IHVW(j - 1, ID)]) * dtdx;
      uold[IHU(rowcol, j, IP)] = u[IHVW(j, IP)] + (flux[IHVW(j - 2, IP)] - flux[IHVW(j - 1, IP)]) * dtdx;
      uold[IHU(rowcol, j, IV)] = u[IHVW(j, IU)] + (flux[IHVW(j - 2, IU)] - flux[IHVW(j - 1, IU)]) * dtdx;
      uold[IHU(rowcol, j, IU)] = u[IHVW(j, IV)] + (flux[IHVW(j - 2, IV)] - flux[IHVW(j - 1, IV)]) * dtdx;
      MFLOPS(12, 0, 0, 0);
    }
#endif
    if (Hnvar > IP + 1) {
      for (ivar = IP + 1; ivar < Hnvar; ivar++) {
        for (j = Hjmin + ExtraLayer; j < Hjmax - ExtraLayer; j++) {
          uold[IHU(rowcol, j, ivar)] = u[IHVW(j, ivar)] + (flux[IHVW(j - 2, ivar)] - flux[IHVW(j - 1, ivar)]) * dtdx;
          MFLOPS(3, 0, 0, 0);
        }
      }
    }
  }
}

#undef IHVW
#undef IHU

//EOF
