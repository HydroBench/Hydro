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
#include "riemann.h"

#ifdef HMPP
#undef HMPP
#include "constoprim.c"
#include "equation_of_state.c"
#include "slope.c"
#include "trace.c"
#include "qleftright.c"
#include "cmpflx.c"
#include "conservar.c"
#define HMPP
#endif

#define PRECISION 1e-6
#define IDX(i,j,k)    ( (i*Hstep*Hnxyt) + (j*Hnxyt) + k )
#define IDXE(i,j)    ( (i*Hnxyt) + j )


void
Dmemset (const size_t nbr, double t[nbr], const double motif)
{
  //int i;
#pragma acc parallel loop pcopyout(t[0:nbr])
  for (int i = 0; i < nbr; i++)
    {
      t[i] = motif;
    }
}


#define DABS(x) (double) fabs((x))
#ifndef HMPP
#define CFLOPS(c) {flops+=c;}
#else
#define MAX(x,y) fmax(x,y)
#define CFLOPS(c)
#endif

/* For CAL/IL */
/* #define sqrt(x) ((double) sqrtf((float)x)) */
/* #define DABS(x) (x > 0.0 ? x : -x) */

void
riemann (const int narray,
	 const double Hsmallr,
	 const double Hsmallc,
	 const double Hgamma,
	 const int Hniter_riemann,
	 const int Hnvar,
	 const int Hnxyt,
	 const int slices, const int Hstep,
	 double *qleft, double *qright, double *qgdnv, int *sgnm)
{
  //double qleft[Hnvar][Hstep][Hnxyt],
  //double qright[Hnvar][Hstep][Hnxyt], //
  //double qgdnv[Hnvar][Hstep][Hnxyt], //
  //int sgnm[Hstep][Hnxyt]) {
  // #define IHVW(i, v) ((i) + (v) * Hnxyt)
  //int i, s;
  const double smallp_ = Square (Hsmallc) / Hgamma;
  const double gamma6_ = (Hgamma + one) / (two * Hgamma);
  const double smallpp_ = Hsmallr * smallp_;

  // Pressure, density and velocity
    #pragma acc kernels present(qleft[0:Hnvar*Hstep*Hnxyt], qright[0:Hnvar*Hstep*Hnxyt]) present(qgdnv[0:Hnvar*Hstep*Hnxyt], sgnm[0:Hstep*Hnxyt]) 
    {
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
          for (int i = 0; i < narray; i++)
	        {
	          double smallp = smallp_;
	          double gamma6 = gamma6_;
	          double smallpp = smallpp_;
	          double rl_i = MAX (qleft[IDX (ID, s, i)], Hsmallr);
	          double ul_i = qleft[IDX (IU, s, i)];
	          double pl_i = MAX (qleft[IDX (IP, s, i)], (double) (rl_i * smallp));
	          double rr_i = MAX (qright[IDX (ID, s, i)], Hsmallr);
	          double ur_i = qright[IDX (IU, s, i)];
	          double pr_i =
	            MAX (qright[IDX (IP, s, i)], (double) (rr_i * smallp));
	          CFLOPS (2);

	          // Lagrangian sound speed
	          double cl_i = Hgamma * pl_i * rl_i;
	          double cr_i = Hgamma * pr_i * rr_i;
	          CFLOPS (4);
	          // First guess

	          double wl_i = sqrt (cl_i);
	          double wr_i = sqrt (cr_i);
	          double pstar_i =
	            MAX (((wr_i * pl_i + wl_i * pr_i) +
		          wl_i * wr_i * (ul_i - ur_i)) / (wl_i + wr_i), 0.0);
	          CFLOPS (9);

	          // Newton-Raphson iterations to find pstar at the required accuracy
	          {
	            int iter;
	            int goon = 1;
	            for (iter = 0; iter < Hniter_riemann; iter++)
	              {
		        if (goon)
		          {
		            double wwl, wwr;
		            wwl =
		              sqrt (cl_i * (one + gamma6 * (pstar_i - pl_i) / pl_i));
		            wwr =
		              sqrt (cr_i * (one + gamma6 * (pstar_i - pr_i) / pr_i));
		            double ql =
		              two * wwl * Square (wwl) / (Square (wwl) + cl_i);
		            double qr =
		              two * wwr * Square (wwr) / (Square (wwr) + cr_i);
		            double usl = ul_i - (pstar_i - pl_i) / wwl;
		            double usr = ur_i + (pstar_i - pr_i) / wwr;
		            double delp_i =
		              MAX ((qr * ql / (qr + ql) * (usl - usr)), (-pstar_i));
		            CFLOPS (38);

		            //PRINTARRAY(delp, narray, "delp", H);
		            pstar_i = pstar_i + delp_i;
		            CFLOPS (1);

		            // Convergence indicator
		            double uo_i = DABS (delp_i / (pstar_i + smallpp));
		            CFLOPS (2);

		            goon = uo_i > PRECISION;
		          }
	              }			// iter_riemann
	          }

	          if (wr_i)
	            {			// Bug CUDA !!
	              wr_i = sqrt (cr_i * (one + gamma6 * (pstar_i - pr_i) / pr_i));
	              wl_i = sqrt (cl_i * (one + gamma6 * (pstar_i - pl_i) / pl_i));
	              CFLOPS (10);
	            }

	          double ustar_i =
	            half * (ul_i + (pl_i - pstar_i) / wl_i + ur_i -
		            (pr_i - pstar_i) / wr_i);
	          CFLOPS (8);

	          int left = ustar_i > 0;
	          double ro_i, uo_i, po_i, wo_i;

	          if (left)
	            {
	              sgnm[IDXE (s, i)] = 1;
	              ro_i = rl_i;
	              uo_i = ul_i;
	              po_i = pl_i;
	              wo_i = wl_i;
	            }
	          else
	            {
	              sgnm[IDXE (s, i)] = -1;
	              ro_i = rr_i;
	              uo_i = ur_i;
	              po_i = pr_i;
	              wo_i = wr_i;
	            }

	          double co_i = sqrt (DABS (Hgamma * po_i / ro_i));
	          co_i = MAX (Hsmallc, co_i);
	          CFLOPS (2);

	          double rstar_i =
	            ro_i / (one + ro_i * (po_i - pstar_i) / Square (wo_i));
	          rstar_i = MAX (rstar_i, Hsmallr);
	          CFLOPS (6);

	          double cstar_i = sqrt (DABS (Hgamma * pstar_i / rstar_i));
	          cstar_i = MAX (Hsmallc, cstar_i);
	          CFLOPS (2);

	          double spout_i = co_i - sgnm[IDXE (s, i)] * uo_i;
	          double spin_i = cstar_i - sgnm[IDXE (s, i)] * ustar_i;
	          double ushock_i = wo_i / ro_i - sgnm[IDXE (s, i)] * uo_i;
	          CFLOPS (7);

	          if (pstar_i >= po_i)
	            {
	              spin_i = ushock_i;
	              spout_i = ushock_i;
	            }

	          double scr_i = MAX ((double) (spout_i - spin_i),
			              (double) (Hsmallc + DABS (spout_i + spin_i)));
	          CFLOPS (3);

	          double frac_i = (one + (spout_i + spin_i) / scr_i) * half;
	          frac_i = MAX (zero, (double) (MIN (one, frac_i)));
	          CFLOPS (4);

	          int addSpout = spout_i < zero;
	          int addSpin = spin_i > zero;
	          // double originalQgdnv = !addSpout & !addSpin;
	          double qgdnv_ID, qgdnv_IU, qgdnv_IP;

	          if (addSpout)
	            {
	              qgdnv_ID = ro_i;
	              qgdnv_IU = uo_i;
	              qgdnv_IP = po_i;
	            }
	          else if (addSpin)
	            {
	              qgdnv_ID = rstar_i;
	              qgdnv_IU = ustar_i;
	              qgdnv_IP = pstar_i;
	            }
	          else
	            {
	              qgdnv_ID = (frac_i * rstar_i + (one - frac_i) * ro_i);
	              qgdnv_IU = (frac_i * ustar_i + (one - frac_i) * uo_i);
	              qgdnv_IP = (frac_i * pstar_i + (one - frac_i) * po_i);
	            }

	          qgdnv[IDX (ID, s, i)] = qgdnv_ID;
	          qgdnv[IDX (IU, s, i)] = qgdnv_IU;
	          qgdnv[IDX (IP, s, i)] = qgdnv_IP;

	          // transverse velocity
	          if (left)
	            {
	              qgdnv[IDX (IV, s, i)] = qleft[IDX (IV, s, i)];
	            }
	          else
	            {
	              qgdnv[IDX (IV, s, i)] = qright[IDX (IV, s, i)];
	            }
	        }
      }
    }//kernels region
  // other passive variables
  if (Hnvar > IP)
  {
 
    //int invar;
    #pragma acc kernels present(qleft[0:Hnvar*Hstep*Hnxyt], qright[0:Hnvar*Hstep*Hnxyt], sgnm[0:Hstep*Hnxyt]) present(qgdnv[0:Hnvar*Hstep*Hnxyt])
    {
#ifdef GRIDIFY
#pragma hmppcg gridify(s*invar,i)
#endif /* GRIDIFY */
#ifndef GRIDIFY
#pragma acc loop independent
#endif /* !GRIDIFY */
      for (int invar = IP + 1; invar < Hnvar; invar++)
      {
#ifndef GRIDIFY
#pragma acc loop independent
#endif /* !GRIDIFY */
        for (int s = 0; s < slices; s++)
        {
          for (int i = 0; i < narray; i++)
          {
            int left = (sgnm[IDXE (s, i)] == 1);
            qgdnv[IDX (invar, s, i)] =
              qleft[IDX (invar, s, i)] * left +
              qright[IDX (invar, s, i)] * !left;
          }
        }
      }
    }//kernels region
  }
}				// riemann

#undef IDX
#undef IDXE

//EOF
