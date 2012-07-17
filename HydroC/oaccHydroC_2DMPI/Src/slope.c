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
#include "slope.h"

#ifndef HMPP

#define DABS(x) (double) fabs((x))
#define IDX(i,j,k) ( (i*Hstep*Hnxyt) + (j*Hnxyt) + k )

void
slope (const int n,
       const int Hnvar,
       const int Hnxyt,
       const double Hslope_type,
       const int slices, const int Hstep, double *q, double *dq){
       //const int slices, const int Hstep, double* q[Hnvar][Hstep][Hnxyt], double* dq) {
  //int nbv, i, ijmin, ijmax, s;
  //double dlft, drgt, dcen, dsgn, slop, dlim;
  // long ihvwin, ihvwimn, ihvwipn;
  // #define IHVW(i, v) ((i) + (v) * Hnxyt)

  WHERE ("slope");
  //ijmin = 0;
  //ijmax = n;

  #pragma acc kernels present(q[0: Hnvar * Hstep * Hnxyt], dq[0:Hnvar * Hstep * Hnxyt])
  {

    double dlft, drgt, dcen, dsgn, slop, dlim;
    int  ijmin, ijmax;
    ijmin = 0;
    ijmax = n;
    //#pragma hmppcg unroll i:4
    #pragma acc  loop independent 
    for (int nbv = 0; nbv < Hnvar; nbv++)
    {
      #pragma acc loop independent private (dlft, drgt, dcen, dsgn, slop, dlim)
      for (int s = 0; s < slices; s++)
      {
        for (int i = ijmin + 1; i < ijmax - 1; i++)
        {
            dlft = Hslope_type * (q[IDX (nbv, s, i)]      - q[IDX (nbv, s, i - 1)]);
            drgt = Hslope_type * (q[IDX (nbv, s, i + 1)]  - q[IDX (nbv, s, i)]);
            dcen = half * (dlft + drgt) / Hslope_type;
            dsgn = (dcen > 0) ? (double) 1.0 : (double) -1.0;	// sign(one, dcen);
            slop = fmin (fabs (dlft), fabs (drgt));
            dlim = slop;
            if ((dlft * drgt) <= zero){
	            dlim = zero;
	          }
          
            dq[IDX(nbv, s, i)] = dsgn * fmin (dlim, fabs (dcen));

           #ifdef FLOPS
            flops += 8;
           #endif
        }
      }
    }
  }//kernels region
}				// slope

//#undef IHVW
#undef IDX

#endif /* HMPP */
//EOF
