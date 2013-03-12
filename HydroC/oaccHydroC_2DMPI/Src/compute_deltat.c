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

//#define DABS(x) (hydro_real_t) fabsf((x))

static void
ComputeQEforRow (const int j,
		 const hydro_real_t Hsmallr,
		 const int Hnx,
		 const int Hnxt,
		 const int Hnyt,
		 const int Hnxyt,
		 const int Hnvar,
		 const int slices, const int Hstep, hydro_real_t *uold,
		 hydro_real_t *q, hydro_real_t *e)
{
  int i, s;
  //  double eken;

#define IHV(i, j, v)  ((i) + Hnxt * ((j) + Hnyt * (v)))
#define IDX(i,j,k)    ( (i*Hstep*Hnxyt) + (j*Hnxyt) + k )
#define IDXE(i,j)     ( (i*Hnxyt) + j )
#pragma acc kernels present(q[0:Hnvar*Hstep*Hnxyt], uold[0:Hnvar*Hnxt*Hnyt],e[0:Hstep*Hnxyt])
  {
#ifdef GRIDIFY
#ifndef GRIDIFY_TUNE_PHI
#pragma hmppcg gridify(s,i)
#else
#pragma hmppcg gridify(s,i), blocksize 256x2
#endif
#endif /* GRIDIFY */
#ifndef GRIDIFY
#pragma acc loop independent
#endif /* !GRIDIFY */
  for (s = 0; s < slices; s++)
    {
#ifndef GRIDIFY
#pragma acc loop independent
#endif /* !GRIDIFY */
      for (i = 0; i < Hnx; i++)
	{
hydro_real_t eken;
	  int idxuID = IHV (i + ExtraLayer, j + s, ID);
	  int idxuIU = IHV (i + ExtraLayer, j + s, IU);
	  int idxuIV = IHV (i + ExtraLayer, j + s, IV);
	  int idxuIP = IHV (i + ExtraLayer, j + s, IP);
	  q[IDX(ID,s,i)] = MAX (uold[idxuID], Hsmallr);
	  q[IDX(IU,s,i)] = uold[idxuIU] / q[IDX(ID,s,i)];
	  q[IDX(IV,s,i)] = uold[idxuIV] / q[IDX(ID,s,i)];
	  eken = half * (Square (q[IDX(IU,s,i)]) + Square (q[IDX(IV,s,i)]));
	  q[IDX(IP,s,i)] = uold[idxuIP] / q[IDX(ID,s,i)] - eken;
	  e[IDXE(s,i)] = q[IDX(IP,s,i)];
	}
    }
  }
#undef IHV
#undef IHVW
}

static void
courantOnXY (hydro_real_t *cournox,
	     hydro_real_t *cournoy,
	     const int Hnx,
	     const int Hnxyt,
	     const int Hnvar, const int slices, const int Hstep,
	     hydro_real_t *c, hydro_real_t *q)
{
  int i, s;
  // double maxValC = zero;
  hydro_real_t tmp1, tmp2;

  // #define IHVW(i,v) ((i) + (v) * nxyt)
  //     maxValC = c[0];
  //     for (i = 0; i < Hnx; i++) {
  //         maxValC = MAX(maxValC, c[i]);
  //     }
  //     for (i = 0; i < Hnx; i++) {
  //         *cournox = MAX(*cournox, maxValC + DABS(q[IU][i]));
  //         *cournoy = MAX(*cournoy, maxValC + DABS(q[IV][i]));
  //     }
  hydro_real_t _cournox = *cournox;
  hydro_real_t _cournoy = *cournoy;
#pragma acc kernels present(q[0:Hnvar*Hstep*Hnxyt],c[0:Hstep*Hnxyt])
  {
#pragma acc loop independent reduction(max:_cournox) reduction(max:_cournoy) gang(128)
  for (s = 0; s < slices; s++)
    {
#pragma acc loop independent reduction(max:_cournox) reduction(max:_cournoy) worker(64)
      for (i = 0; i < Hnx; i++)
	{
	  tmp1 = c[IDXE(s,i)] + DABS (q[IDX(IU,s,i)]);
	  tmp2 = c[IDXE(s,i)] + DABS (q[IDX(IV,s,i)]);
	  _cournox = MAX (_cournox, tmp1);
	  _cournoy = MAX (_cournoy, tmp2);
	}
    }
  }
  *cournox = _cournox;
  *cournoy = _cournoy;

#undef IHVW
}

void
compute_deltat (hydro_real_t *dt, const hydroparam_t H, hydrowork_t * Hw,
		hydrovar_t * Hv, hydrovarwork_t * Hvw)
{
  hydro_real_t cournox, cournoy;
  int j, jend, slices, Hstep, Hmin, Hmax;
  hydro_real_t (*e)[H.nxyt];
  hydro_real_t (*c)[H.nxyt];
  hydro_real_t (*q)[H.nxystep][H.nxyt];
  WHERE ("compute_deltat");

  //   compute time step on grid interior
  cournox = zero;
  cournoy = zero;
  //Hvw->q = (double (*)) calloc (H.nvar * H.nxystep * H.nxyt, sizeof (double));
  //Hw->e = (double (*)) malloc ((H.nxyt) * H.nxystep * sizeof (double));
  //Hw->c = (double (*)) malloc ((H.nxyt) * H.nxystep * sizeof (double));

  c = (hydro_real_t (*)[H.nxyt]) Hw->c;
  e = (hydro_real_t (*)[H.nxyt]) Hw->e;
  q = (hydro_real_t (*)[H.nxystep][H.nxyt]) Hvw->q;

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
		       
//#pragma acc update device (q[0:H.nvar], e[0:H.nxystep], c[0:H.nxystep])
      equation_of_state (0, H.nx, H.nxyt, H.nvar, H.smallc, H.gamma, slices, Hstep, e, q, c);

			//download everything on Host
      //#pragma acc update host (q[0:H.nvar],c[0:H.nxystep])
      courantOnXY (&cournox, &cournoy, H.nx, H.nxyt, H.nvar, slices, Hstep, c, q);
		   
		   
#ifdef FLOPS
      flops += 10;
#endif /*  */
    }
  //Free (Hvw->q);
  //Free (Hw->e);
  //Free (Hw->c);
  *dt =(hydro_real_t)( H.courant_factor * H.dx / MAX (cournox, MAX (cournoy, H.smallc)));
#ifdef FLOPS
  flops += 2;

#endif /*  */

  // fprintf(stdout, "%g %g %g %g\n", cournox, cournoy, H.smallc, H.courant_factor);
}				// compute_deltat

//EOF
