/*
  A simple 2D hydro code
  (C) Romain Teyssier : CEA/IRFU           -- original F90 code
  (C) Pierre-Francois Lavallee : IDRIS      -- original F90 code
  (C) Guillaume Colin de Verdiere : CEA/DAM -- for the C version
*/

// #include <stdlib.h>
// #include <unistd.h>
#include <math.h>
#include <stdio.h>

#ifndef HMPP
#include "equation_of_state.h"
#include "parametres.h"
#include "utils.h"

#define CFLOPS(c)		/* {flops+=c;} */
#define IDX(i,j,k)    ( (i*Hstep*Hnxyt) + (j*Hnxyt) + k )
#define IDXE(i,j)     ( (i*Hnxyt) + j )


void
equation_of_state (const int imin,
		   const int imax,
		   const int Hnxyt,
		   const int Hnvar,
		   const hydro_real_t Hsmallc,
		   const hydro_real_t Hgamma,
		   const int slices, const int Hstep,
		   hydro_real_t *eint, hydro_real_t *q, hydro_real_t *c)
{
  //double eint[Hstep][Hnxyt], double q[Hnvar][Hstep][Hnxyt], double c[Hstep][Hnxyt]) {
  //int k, s;
  //double smallp;

  WHERE ("equation_of_state"); 
  //smallp = Square (Hsmallc) / Hgamma;
  //CFLOPS (1);

  #pragma acc kernels present(eint[0:Hstep*Hnxyt], q[0:Hnvar*Hstep*Hnxyt], c[0:Hstep*Hnxyt])
  {
    hydro_real_t smallp = Square (Hsmallc) / Hgamma;
    CFLOPS (1);
#ifdef GRIDIFY
#ifndef GRIDIFY_TUNE_PHI
#pragma hmppcg gridify(s,k)
#else
#pragma hmppcg gridify(s,k), blocksize 512x1
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
      for (int k = imin; k < imax; k++)
	    {
	      hydro_real_t rhok = q[IDX (ID, s, k)];
	      hydro_real_t base = (Hgamma - one) * rhok * eint[IDXE (s, k)];
	      base = MAX (base, (hydro_real_t) (rhok * smallp));

	      q[IDX (IP, s, k)] = base;
	      c[IDXE (s, k)] = sqrt (Hgamma * base / rhok);

	      CFLOPS (7);
	    }
    }
  }//kernels region
}				// equation_of_state

#undef IDX
#undef IDXE

#endif
// EOF
