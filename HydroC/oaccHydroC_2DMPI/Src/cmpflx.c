/*
  A simple 2D hydro code
  (C) Romain Teyssier : CEA/IRFU           -- original F90 code
  (C) Pierre-Francois Lavallee : IDRIS      -- original F90 code
  (C) Guillaume Colin de Verdiere : CEA/DAM -- for the C version
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

#define CFLOPS(c)		/* {flops+=c;} */

void
cmpflx (const int narray,
	const int Hnxyt,
	const int Hnvar,
	const real Hgamma,
	const int slices, const int Hstep, real *qgdnv, real *flux)
{
  //       const int slices, const int Hstep, double qgdnv[Hnvar][Hstep][Hnxyt], double flux[Hnvar][Hstep][Hnxyt]) {
  //int nface, i, IN;
  //double entho, ekin, etot;
  WHERE ("cmpflx");
  //int s;

  //nface = narray;
  //entho = one / (Hgamma - one);

#define IDX(i,j,k) ( (i*Hstep*Hnxyt) + (j*Hnxyt) + k )

  // Compute fluxes
#pragma acc kernels present(qgdnv[0:Hnvar*Hstep*Hnxyt]) present(flux[0:Hnvar*Hstep*Hnxyt])
{
  real entho, ekin, etot;
  int nface;
  nface = narray;
  entho = one / (Hgamma - one);

#ifdef GRIDIFY
#ifndef GRIDIFY_TUNE_PHI
#pragma hmppcg gridify(s,i)
#else
#pragma hmppcg gridify(s,i), blocksize 128x1
#endif
#endif /* GRIDIFY */
#ifndef GRIDIFY
#pragma acc loop independent
#endif /* !GRIDIFY */
  for (int s = 0; s < slices; s++)
    {
#ifndef GRIDIFY
#pragma acc loop independent
#endif /* !GRIDIFY */
      for (int i = 0; i < nface; i++)
	      {
	        real qgdnvID = qgdnv[IDX (ID, s, i)];
	        real qgdnvIU = qgdnv[IDX (IU, s, i)];
	        real qgdnvIP = qgdnv[IDX (IP, s, i)];
	        real qgdnvIV = qgdnv[IDX (IV, s, i)];

	        // Mass density
	        real massDensity = qgdnvID * qgdnvIU;
	        flux[IDX (ID, s, i)] = massDensity;

	        // Normal momentum
	        flux[IDX (IU, s, i)] = massDensity * qgdnvIU + qgdnvIP;
	        // Transverse momentum 1
	        flux[IDX (IV, s, i)] = massDensity * qgdnvIV;

	        // Total energy
	        ekin = half * qgdnvID * (Square (qgdnvIU) + Square (qgdnvIV));
	        etot = qgdnvIP * entho + ekin;

	        flux[IDX (IP, s, i)] = qgdnvIU * (etot + qgdnvIP);

	        CFLOPS (15);
	      }
    }
}//kernels region

  // Other advected quantities
  if (Hnvar > IP)
  {
    #pragma acc kernels present(qgdnv[0:Hnvar*Hstep*Hnxyt]) present(flux[0:Hnvar*Hstep*Hnxyt])
    {  
      int nface;
      nface = narray;

#ifdef GRIDIFY
#ifndef GRIDIFY_TUNE_PHI
#pragma hmppcg gridify(s*IN,i)
#else
#pragma hmppcg gridify(s*IN,i), blocksize 128x1
#endif
#endif /* GRIDIFY */
#ifndef GRIDIFY
#pragma acc loop independent
#endif /* !GRIDIFY */
      for (int s = 0; s < slices; s++)
	    {
#ifndef GRIDIFY
#pragma acc loop independent
#endif /* !GRIDIFY */
	      for (int IN = IP + 1; IN < Hnvar; IN++)
	      {
	        for (int i = 0; i < nface; i++)
		      {
		        flux[IDX (IN, s, i)] =
		        flux[IDX (IN, s, i)] * qgdnv[IDX (IN, s, i)];
		      }
	      }
	    }
    }//kernels region
  }
}				// cmpflx

#undef IDX

#endif

//EOF
