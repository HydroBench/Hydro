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

#ifndef HMPP
#include "parametres.h"
#include "constoprim.h"
#include "utils.h"

#define CFLOPS(c)		/* {flops+=c;} */
#define IDX(i,j,k)    ( (i*Hstep*Hnxyt) + (j*Hnxyt) + k )
#define IDXE(i,j)     ( (i*Hnxyt) + j )


void
constoprim (const int n,
	    const int Hnxyt,
	    const int Hnvar,
	    const double Hsmallr,
	    const int slices, const int Hstep,
	    double *u, double *q, double *e)
{
  //double u[Hnvar][Hstep][Hnxyt], double q[Hnvar][Hstep][Hnxyt], double e[Hstep][Hnxyt]) {
  //int ijmin, ijmax, IN, i, s;
  //double eken;
  // const int nxyt = Hnxyt;
  WHERE ("constoprim");
  //ijmin = 0;
  //ijmax = n;

  

#pragma acc kernels present(u[0:Hnvar*Hstep*Hnxyt]) present(q[0:Hnvar*Hstep*Hnxyt], e[0:Hstep*Hnxyt])
{
  const int ijmin=0, ijmax=n;
#ifdef GRIDIFY
#pragma hmppcg gridify(s,i)
#endif /* GRIDIFY */
#ifndef GRIDIFY
#pragma acc loop independent
#endif /* !GRIDIFY */
  for (int s = 0; s < slices; s++)
    {
#ifndef GRIDIFY
#pragma acc loop independent
#endif /* !GRIDIFY */
      for (int i = ijmin; i < ijmax; i++)
	    {
	      double qid = MAX (u[IDX (ID, s, i)], Hsmallr);
	      q[IDX (ID, s, i)] = qid;

	      double qiu = u[IDX (IU, s, i)] / qid;
	      double qiv = u[IDX (IV, s, i)] / qid;
	      q[IDX (IU, s, i)] = qiu;
	      q[IDX (IV, s, i)] = qiv;

	      double eken = half * (Square (qiu) + Square (qiv));

	      double qip = u[IDX (IP, s, i)] / qid - eken;
	      q[IDX (IP, s, i)] = qip;
	      e[IDXE (s, i)] = qip;

	      CFLOPS (9);
	    }
    }
}//kernels region

  if (Hnvar > IP)
  {
    #pragma acc kernels present(u[0:Hnvar*Hstep*Hnxyt]) present(q[0:Hnvar*Hstep*Hnxyt])
    {
      const int ijmin=0, ijmax=n;
#ifdef GRIDIFY
#pragma hmppcg gridify(s*IN,i)
#endif /* GRIDIFY */
#ifndef GRIDIFY
#pragma acc loop independent
#endif /* !GRIDIFY */
      for (int IN = IP + 1; IN < Hnvar; IN++)
	    {
#ifndef GRIDIFY
#pragma acc loop independent
#endif /* !GRIDIFY */
	      for (int s = 0; s < slices; s++)
	      {
	        for (int i = ijmin; i < ijmax; i++)
		      {
		        q[IDX (IN, s, i)] = u[IDX (IN, s, i)] / q[IDX (IN, s, i)];
		        CFLOPS (1);
		      }
	      }
	    }
    }//kernels region
  }
}				// constoprim


#undef IHVW
#undef IDX
#undef IDXE
#endif
//EOF
