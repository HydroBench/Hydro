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
#include "trace.h"

#ifndef HMPP
#define CFLOPS(c)		/* {flops+=c;} */
#define IDX(i,j,k) ( (i*Hstep*Hnxyt) + (j*Hnxyt) + k )
#define IDXE(i,j) ( (i*Hnxyt) + j )

void
trace (const hydro_real_t dtdx,
       const int n,
       const int Hscheme,
       const int Hnvar,
       const int Hnxyt,
       const int slices, const int Hstep,
       hydro_real_t *q, hydro_real_t *dq, hydro_real_t *c, hydro_real_t *qxm, hydro_real_t *qxp)
{
  //double q[Hnvar][Hstep][Hnxyt],
  //double dq[Hnvar][Hstep][Hnxyt], double c[Hstep][Hnxyt], double qxm[Hnvar][Hstep][Hnxyt],
  //double qxp[Hnvar][Hstep][Hnxyt]) {
  //int ijmin, ijmax;
  //int i, IN, s;
  //double cc, csq,r, u, v, p, a;
  //double dr, du, dv, dp, da;
  //double alpham, alphap, alpha0r, alpha0v;
  //double spminus, spplus, spzero;
  //double apright, amright, azrright, azv1right, acmpright;
  //double apleft, amleft, azrleft, azv1left, acmpleft;
  
  hydro_real_t zerol = zero, zeror = zero, project = zero;

  WHERE ("trace");
  //ijmin = 0;
  //ijmax = n;

  // if (strcmp(Hscheme, "muscl") == 0) {       // MUSCL-Hancock method
  if (Hscheme == HSCHEME_MUSCL)
    {				// MUSCL-Hancock method
      zerol = -hundred / dtdx;
      zeror = hundred / dtdx;
      project = one;
      CFLOPS (2);
    }
  // if (strcmp(Hscheme, "plmde") == 0) {       // standard PLMDE
  if (Hscheme == HSCHEME_PLMDE)
    {				// standard PLMDE
      zerol = zero;
      zeror = zero;
      project = one;
    }
  // if (strcmp(Hscheme, "collela") == 0) {     // Collela's method
  if (Hscheme == HSCHEME_COLLELA)
    {				// Collela's method
      zerol = zero;
      zeror = zero;
      project = zero;
    }

  #pragma acc kernels present(q[0:Hnvar*Hstep*Hnxyt], dq[0:Hnvar*Hstep*Hnxyt], c[0:Hstep*Hnxyt]) present(qxm[0:Hnvar*Hstep*Hnxyt], qxp[0:Hnvar*Hstep*Hnxyt]) 
  {
/*    double cc, csq,r, u, v, p;
    double dr, du, dv, dp;
    double alpham, alphap, alpha0r, alpha0v;
    double spminus, spplus, spzero;
    double apright, amright, azrright, azv1right;
    double apleft, amleft, azrleft, azv1left;
*/
    int ijmin=0, ijmax=n;


#ifdef GRIDIFY
#ifndef GRIDIFY_TUNE_PHI
#pragma hmppcg gridify(s,i)
#else
#pragma hmppcg gridify(s,i), blocksize 256x1
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
      for (int i = ijmin + 1; i < ijmax - 1; i++)
	    {
    hydro_real_t cc, csq,r, u, v, p;
    hydro_real_t dr, du, dv, dp;
    hydro_real_t alpham, alphap, alpha0r, alpha0v;
    hydro_real_t spminus, spplus, spzero;
    hydro_real_t apright, amright, azrright, azv1right;
    hydro_real_t apleft, amleft, azrleft, azv1left;

        cc = c[IDXE (s, i)];
        csq = Square (cc);
        r = q[IDX (ID, s, i)];
        u = q[IDX (IU, s, i)];
        v = q[IDX (IV, s, i)];
        p = q[IDX (IP, s, i)];
        dr = dq[IDX (ID, s, i)];
        du = dq[IDX (IU, s, i)];
        dv = dq[IDX (IV, s, i)];
        dp = dq[IDX (IP, s, i)];
        alpham = half * (dp / (r * cc) - du) * r / cc;
        alphap = half * (dp / (r * cc) + du) * r / cc;
        alpha0r = dr - dp / csq;
        alpha0v = dv;

        // Right state
        spminus = (u - cc) * dtdx + one;
        spplus = (u + cc) * dtdx + one;
        spzero = u * dtdx + one;
        if ((u - cc) >= zeror)
        {
          spminus = project;
        }
        if ((u + cc) >= zeror)
        {
          spplus = project;
        }
        if (u >= zeror)
        {
          spzero = project;
        }
        apright = -half * spplus * alphap;
        amright = -half * spminus * alpham;
        azrright = -half * spzero * alpha0r;
        azv1right = -half * spzero * alpha0v;
        qxp[IDX (ID, s, i)] = r + (apright + amright + azrright);
        qxp[IDX (IU, s, i)] = u + (apright - amright) * cc / r;
        qxp[IDX (IV, s, i)] = v + (azv1right);
        qxp[IDX (IP, s, i)] = p + (apright + amright) * csq;

        // Left state
        spminus = (u - cc) * dtdx - one;
        spplus = (u + cc) * dtdx - one;
        spzero = u * dtdx - one;
        if ((u - cc) <= zerol)
        {
          spminus = -project;
        }
        if ((u + cc) <= zerol)
        {
          spplus = -project;
        }
        if (u <= zerol)
        {
          spzero = -project;
        }
        apleft = -half * spplus * alphap;
        amleft = -half * spminus * alpham;
        azrleft = -half * spzero * alpha0r;
        azv1left = -half * spzero * alpha0v;
        qxm[IDX (ID, s, i)] = r + (apleft + amleft + azrleft);
        qxm[IDX (IU, s, i)] = u + (apleft - amleft) * cc / r;
        qxm[IDX (IV, s, i)] = v + (azv1left);
        qxm[IDX (IP, s, i)] = p + (apleft + amleft) * csq;

        CFLOPS (78);
      }
    }
  }//kernels region
  if (Hnvar > IP)
  {
    #pragma acc kernels present(q[0:Hnvar*Hstep*Hnxyt], dq[0:Hnvar*Hstep*Hnxyt]) present(qxp[0:Hnvar*Hstep*Hnxyt], qxm[0:Hnvar*Hstep*Hnxyt])
    {
///      double  u, a, acmpleft;
///      double  da, acmpright, spzero;
      int ijmin=0, ijmax=n;
#ifdef GRIDIFY
#ifndef GRIDIFY_TUNE_PHI
#pragma hmppcg gridify(IN*s,i)
#else
#pragma hmppcg gridify(IN*s,i), blocksize 256x1
#endif
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
	        for (int i = ijmin + 1; i < ijmax - 1; i++)
		      {
hydro_real_t  u, a, acmpleft;
hydro_real_t  da, acmpright, spzero;
		        u = q[IDX (IU, s, i)];
		        a = q[IDX (IN, s, i)];
		        da = dq[IDX (IN, s, i)];

		        // Right state
		        spzero = u * dtdx + one;
		        if (u >= zeror)
		        {
		          spzero = project;
		        }
		        acmpright = -half * spzero * da;
		        qxp[IDX (IN, s, i)] = a + acmpright;

		        // Left state
		        spzero = u * dtdx - one;
		        if (u <= zerol)
		        {
		          spzero = -project;
		        }
		        acmpleft = -half * spzero * da;
		        qxm[IDX (IN, s, i)] = a + acmpleft;

		        CFLOPS (10);
		      }
	      }
	    }
    }//kernels region
  }
}				// trace

#undef IDX
#undef IDXE

#endif /* HMPP */

//EOF
