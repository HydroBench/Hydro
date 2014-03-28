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

#define PRECISION autocast(1e-6)
#define IDX(i,j,k)    ( (i*Hstep*Hnxyt) + (j*Hnxyt) + k )
#define IDXE(i,j)    ( (i*Hnxyt) + j )


void
Dmemset (const size_t nbr, hydro_real_t t[nbr], const hydro_real_t motif)
{
  //int i;
#ifndef GRIDIFY
#pragma acc kernels pcopyout(t[0:nbr])
#endif
#ifdef GRIDIFY
#ifndef GRIDIFY_TUNE_PHI
#pragma hmppcg gridify(i)
#else
#pragma hmppcg gridify(i), blocksize 64x8
#endif
#endif
#ifndef GRIDIFY
#pragma acc loop independent
#endif
  for (int i = 0; i < nbr; i++)
    {
      t[i] = motif;
    }
}


#ifndef HMPP
#define CFLOPS(c) /*{flops+=c;}*/
#else
//#define MAX(x,y) fmaxf(x,y)
#define CFLOPS(c)
#endif

/* For CAL/IL */
/* #define DABS(x) (x > 0.0 ? x : -x) */

void
riemann (const int narray,
	 const hydro_real_t Hsmallr,
	 const hydro_real_t Hsmallc,
	 const hydro_real_t Hgamma,
	 const int Hniter_riemann,
	 const int Hnvar,
	 const int Hnxyt,
	 const int slices, const int Hstep,
	 hydro_real_t *qleft, hydro_real_t *qright, hydro_real_t *qgdnv, int *sgnm)
{
  //double qleft[Hnvar][Hstep][Hnxyt],
  //double qright[Hnvar][Hstep][Hnxyt], //
  //double qgdnv[Hnvar][Hstep][Hnxyt], //
  //int sgnm[Hstep][Hnxyt]) {
  // #define IHVW(i, v) ((i) + (v) * Hnxyt)
  //int i, s;
  const hydro_real_t smallp_ = Square (Hsmallc) / Hgamma;
  const hydro_real_t gamma6_ = (Hgamma + one) / (two * Hgamma);
  const hydro_real_t smallpp_ = Hsmallr * smallp_;

  // Pressure, density and velocity
    #pragma acc kernels present(qleft[0:Hnvar*Hstep*Hnxyt], qright[0:Hnvar*Hstep*Hnxyt]) present(qgdnv[0:Hnvar*Hstep*Hnxyt], sgnm[0:Hstep*Hnxyt]) 
    {
#ifdef GRIDIFY
#ifndef GRIDIFY_TUNE_PHI
#pragma hmppcg gridify(s,i)
#else
#pragma hmppcg gridify(s,i), blocksize 64x8
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
          for (int i = 0; i < narray; i++)
	        {
	          hydro_real_t smallp = smallp_;
	          hydro_real_t gamma6 = gamma6_;
	          hydro_real_t smallpp = smallpp_;
	          hydro_real_t rl_i = MAX (qleft[IDX (ID, s, i)], Hsmallr);
	          hydro_real_t ul_i = qleft[IDX (IU, s, i)];
	          hydro_real_t pl_i = MAX (qleft[IDX (IP, s, i)], (hydro_real_t) (rl_i * smallp));
	          hydro_real_t rr_i = MAX (qright[IDX (ID, s, i)], Hsmallr);
	          hydro_real_t ur_i = qright[IDX (IU, s, i)];
	          hydro_real_t pr_i =
	            MAX (qright[IDX (IP, s, i)], (hydro_real_t) (rr_i * smallp));
	          CFLOPS (2);

	          // Lagrangian sound speed
	          hydro_real_t cl_i = Hgamma * pl_i * rl_i;
	          hydro_real_t cr_i = Hgamma * pr_i * rr_i;
	          CFLOPS (4);
	          // First guess

	          hydro_real_t wl_i = sqrt (cl_i);
	          hydro_real_t wr_i = sqrt (cr_i);
	          hydro_real_t pstar_i =
	            MAX (((wr_i * pl_i + wl_i * pr_i) +
		          wl_i * wr_i * (ul_i - ur_i)) / (wl_i + wr_i), zero);
	          CFLOPS (9);

	          // Newton-Raphson iterations to find pstar at the required accuracy
	          {
	            int iter;
	            int goon = 1;
	            for (iter = 0; iter < Hniter_riemann; iter++)
	              {
		        if (goon)
		          {
		            hydro_real_t wwl, wwr;
		            wwl =
		              sqrt (cl_i * (one + gamma6 * (pstar_i - pl_i) / pl_i));
		            wwr =
		              sqrt (cr_i * (one + gamma6 * (pstar_i - pr_i) / pr_i));
		            hydro_real_t ql =
		              two * wwl * Square (wwl) / (Square (wwl) + cl_i);
		            hydro_real_t qr =
		              two * wwr * Square (wwr) / (Square (wwr) + cr_i);
		            hydro_real_t usl = ul_i - (pstar_i - pl_i) / wwl;
		            hydro_real_t usr = ur_i + (pstar_i - pr_i) / wwr;
		            hydro_real_t delp_i =
		              MAX ((qr * ql / (qr + ql) * (usl - usr)), (-pstar_i));
		            CFLOPS (38);

		            //PRINTARRAY(delp, narray, "delp", H);
		            pstar_i = pstar_i + delp_i;
		            CFLOPS (1);

		            // Convergence indicator
		            hydro_real_t uo_i = DABS (delp_i / (pstar_i + smallpp));
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

	          hydro_real_t ustar_i =
	            half * (ul_i + (pl_i - pstar_i) / wl_i + ur_i -
		            (pr_i - pstar_i) / wr_i);
	          CFLOPS (8);

	          int left = ustar_i > 0;
	          hydro_real_t ro_i, uo_i, po_i, wo_i;

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

	          hydro_real_t co_i = sqrt (DABS (Hgamma * po_i / ro_i));
	          co_i = MAX (Hsmallc, co_i);
	          CFLOPS (2);

	          hydro_real_t rstar_i =
	            ro_i / (one + ro_i * (po_i - pstar_i) / Square (wo_i));
	          rstar_i = MAX (rstar_i, Hsmallr);
	          CFLOPS (6);

	          hydro_real_t cstar_i = sqrt (DABS (Hgamma * pstar_i / rstar_i));
	          cstar_i = MAX (Hsmallc, cstar_i);
	          CFLOPS (2);

	          hydro_real_t spout_i = co_i - sgnm[IDXE (s, i)] * uo_i;
	          hydro_real_t spin_i = cstar_i - sgnm[IDXE (s, i)] * ustar_i;
	          hydro_real_t ushock_i = wo_i / ro_i - sgnm[IDXE (s, i)] * uo_i;
	          CFLOPS (7);

	          if (pstar_i >= po_i)
	            {
	              spin_i = ushock_i;
	              spout_i = ushock_i;
	            }

	          hydro_real_t scr_i = MAX ((hydro_real_t) (spout_i - spin_i),
			              (hydro_real_t) (Hsmallc + DABS (spout_i + spin_i)));
	          CFLOPS (3);

	          hydro_real_t frac_i = (one + (spout_i + spin_i) / scr_i) * half;
	          frac_i = MAX (zero, (hydro_real_t) (MIN (one, frac_i)));
	          CFLOPS (4);

	          int addSpout = spout_i < zero;
	          int addSpin = spin_i > zero;
	          // hydro_real_t originalQgdnv = !addSpout & !addSpin;
	          hydro_real_t qgdnv_ID, qgdnv_IU, qgdnv_IP;

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
#ifndef GRIDIFY_TUNE_PHI
#pragma hmppcg gridify(invar*s,i)
#else
#pragma hmppcg gridify(invar*s,i), blocksize 64x8
#endif
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
