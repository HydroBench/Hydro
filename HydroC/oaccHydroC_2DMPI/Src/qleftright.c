/*
  A simple 2D hydro code
  (C) Romain Teyssier : CEA/IRFU           -- original F90 code
  (C) Pierre-Francois Lavallee : IDRIS      -- original F90 code
  (C) Guillaume Colin de Verdiere : CEA/DAM -- for the C version
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
qleftright (const int idim,
	    const int Hnx,
	    const int Hny,
	    const int Hnxyt,
	    const int Hnvar,
	    const int slices, const int Hstep,
	    hydro_real_t *qxm, hydro_real_t *qxp, hydro_real_t *qleft, hydro_real_t *qright)
{
  //double qxm[Hnvar][Hstep][Hnxyt],
  //double qxp[Hnvar][Hstep][Hnxyt], double qleft[Hnvar][Hstep][Hnxyt], double qright[Hnvar][Hstep][Hnxyt]) {
  // #define IHVW(i,v) ((i) + (v) * Hnxyt)
  //int nvar, i, s;
  int bmax;
  WHERE ("qleftright");
  if (idim == 1)
    {
      bmax = Hnx + 1;
    }
  else
    {
      bmax = Hny + 1;
    }

  #pragma acc kernels present(qxm[0:Hnvar*Hstep*Hnxyt], qxp[0:Hnvar*Hstep*Hnxyt]) present(qleft[0:Hnvar*Hstep*Hnxyt], qright[0:Hnvar*Hstep*Hnxyt]) 
  {

#ifdef GRIDIFY
#ifndef GRIDIFY_TUNE_PHI
#pragma hmppcg gridify(nvar*s,i)
#else
#pragma hmppcg gridify(nvar*s,i), blocksize 512x1
#endif
#endif /* GRIDIFY */
#ifndef GRIDIFY
#pragma acc loop independent
#endif /* !GRIDIFY */
    for (int nvar = 0; nvar < Hnvar; nvar++)
    {
#ifndef GRIDIFY
#pragma acc loop independent
#endif /* !GRIDIFY */
      for (int s = 0; s < slices; s++)
	    {
#ifndef GRIDIFY
#pragma acc loop independent
#endif /* !GRIDIFY */
	      for (int i = 0; i < bmax; i++)
        {
          qleft[IDX (nvar, s, i)] = qxm[IDX (nvar, s, i + 1)];
          qright[IDX (nvar, s, i)] = qxp[IDX (nvar, s, i + 2)];
        }
	    }
    }
  }//kernels region
}

#undef IHVW
#undef IDX

#endif /* HMPP */
// EOF
