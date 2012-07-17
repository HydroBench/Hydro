/*
  A simple 2D hydro code
  (C) Romain Teyssier : CEA/IRFU           -- original F90 code
  (C) Pierre-Francois Lavallee : IDRIS      -- original F90 code
  (C) Guillaume Colin de Verdiere : CEA/DAM -- for the C version
*/

#include <stdio.h>
// #include <stdlib.h>
#include <malloc.h>
// #include <unistd.h>
#include <math.h>

#ifdef HMPP
#undef HMPP
#endif

#include "parametres.h"
#include "compute_deltat.h"
#include "utils.h"
#include "equation_of_state.h"

#define DABS(x) (double) fabs((x))

static void
ComputeQEforRow (const int j,
		 const double Hsmallr,
		 const int Hnx,
		 const int Hnxt,
		 const int Hnyt,
		 const int Hnxyt,
		 const int Hnvar,
		 const int slices, const int Hstep, double *uold,
		 double q[Hnvar][Hstep][Hnxyt], double e[Hstep][Hnxyt])
{
  int i, s;
  double eken;

#define IHV(i, j, v)  ((i) + Hnxt * ((j) + Hnyt * (v)))

  for (s = 0; s < slices; s++)
    {
      for (i = 0; i < Hnx; i++)
	{
	  int idxuID = IHV (i + ExtraLayer, j + s, ID);
	  int idxuIU = IHV (i + ExtraLayer, j + s, IU);
	  int idxuIV = IHV (i + ExtraLayer, j + s, IV);
	  int idxuIP = IHV (i + ExtraLayer, j + s, IP);
	  q[ID][s][i] = MAX (uold[idxuID], Hsmallr);
	  q[IU][s][i] = uold[idxuIU] / q[ID][s][i];
	  q[IV][s][i] = uold[idxuIV] / q[ID][s][i];
	  eken = half * (Square (q[IU][s][i]) + Square (q[IV][s][i]));
	  q[IP][s][i] = uold[idxuIP] / q[ID][s][i] - eken;
	  e[s][i] = q[IP][s][i];
	}
    }
#undef IHV
#undef IHVW
}

static void
courantOnXY (double *cournox,
	     double *cournoy,
	     const int Hnx,
	     const int Hnxyt,
	     const int Hnvar, const int slices, const int Hstep,
	     double c[Hstep][Hnxyt], double q[Hnvar][Hstep][Hnxyt])
{
  int i, s;
  // double maxValC = zero;
  double tmp1, tmp2;

  // #define IHVW(i,v) ((i) + (v) * nxyt)
  //     maxValC = c[0];
  //     for (i = 0; i < Hnx; i++) {
  //         maxValC = MAX(maxValC, c[i]);
  //     }
  //     for (i = 0; i < Hnx; i++) {
  //         *cournox = MAX(*cournox, maxValC + DABS(q[IU][i]));
  //         *cournoy = MAX(*cournoy, maxValC + DABS(q[IV][i]));
  //     }
  for (s = 0; s < slices; s++)
    {
      for (i = 0; i < Hnx; i++)
	{
	  tmp1 = c[s][i] + DABS (q[IU][s][i]);
	  tmp2 = c[s][i] + DABS (q[IV][s][i]);
	  *cournox = MAX (*cournox, tmp1);
	  *cournoy = MAX (*cournoy, tmp2);
	}
    }

#undef IHVW
}

void
compute_deltat (double *dt, const hydroparam_t H, hydrowork_t * Hw,
		hydrovar_t * Hv, hydrovarwork_t * Hvw)
{
  double cournox, cournoy;
  int j, jend, slices, Hstep, Hmin, Hmax;
  double (*e)[H.nxyt];
  double (*c)[H.nxystep];
  double (*q)[H.nxystep][H.nxyt];
  WHERE ("compute_deltat");

  //   compute time step on grid interior
  cournox = zero;
  cournoy = zero;
  //Hvw->q = (double (*)) calloc (H.nvar * H.nxystep * H.nxyt, sizeof (double));
  //Hw->e = (double (*)) malloc ((H.nxyt) * H.nxystep * sizeof (double));
  //Hw->c = (double (*)) malloc ((H.nxyt) * H.nxystep * sizeof (double));

  c = (double (*)[H.nxystep]) Hw->c;
  e = (double (*)[H.nxystep]) Hw->e;
  q = (double (*)[H.nxystep][H.nxyt]) Hvw->q;

  Hstep = H.nxystep;
  Hmin = H.jmin + ExtraLayer;
  Hmax = H.jmax - ExtraLayer;
  
  for (j = Hmin; j < Hmax; j += Hstep)
  {
      jend = j + Hstep;
      if (jend >= Hmax)
			jend = Hmax;
      slices = jend - j;	// numbre of slices to compute
      ComputeQEforRow (j, H.smallr, H.nx, H.nxt, H.nyt, H.nxyt, H.nvar,
		       slices, Hstep, Hv->uold, q, e);
		       
		  #pragma acc update device (q[0:H.nvar], e[0:H.nxystep], c[0:H.nxystep])
      equation_of_state (0, H.nx, H.nxyt, H.nvar, H.smallc, H.gamma, slices, Hstep, e, q, c);

			//download everything on Host
			#pragma acc update host (q[0:H.nvar],c[0:H.nxystep])
      courantOnXY (&cournox, &cournoy, H.nx, H.nxyt, H.nvar, slices, Hstep, c,
		   q);
		   
		   
#ifdef FLOPS
      flops += 10;
#endif /*  */
    }
  //Free (Hvw->q);
  //Free (Hw->e);
  //Free (Hw->c);
  *dt = H.courant_factor * H.dx / MAX (cournox, MAX (cournoy, H.smallc));

#ifdef FLOPS
  flops += 2;

#endif /*  */

  // fprintf(stdout, "%g %g %g %g\n", cournox, cournoy, H.smallc, H.courant_factor);
}				// compute_deltat

//EOF
