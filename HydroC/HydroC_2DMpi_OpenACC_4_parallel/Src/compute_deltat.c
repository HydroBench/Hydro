/*
  A simple 2D hydro code
  (C) Romain Teyssier : CEA/IRFU           -- original F90 code
  (C) Pierre-Francois Lavallee : IDRIS      -- original F90 code
  (C) Guillaume Colin de Verdiere : CEA/DAM -- for the C version
  (C) Jeffrey Poznanovic : CSCS             -- for the OpenACC version
*/

/*

This software is governed by the CeCILL license under French law and
abiding by the rules of distribution of free software.  You can  use, 
modify and/ or redistribute the software under the terms of the CeCILL
license as circulated by CEA, CNRS and INRIA at the following URL
"http://www.cecill.info". 

As a counterpart to the access to the source code and  rights to copy,
modify and redistribute granted by the license, users are provided only
with a limited warranty  and the software's author,  the holder of the
economic rights,  and the successive licensors  have only  limited
liability. 

In this respect, the user's attention is drawn to the risks associated
with loading,  using,  modifying and/or developing or reproducing the
software by the user in light of its specific status of free software,
that may mean  that it is complicated to manipulate,  and  that  also
therefore means  that it is reserved for developers  and  experienced
professionals having in-depth computer knowledge. Users are therefore
encouraged to load and test the software's suitability as regards their
requirements in conditions enabling the security of their systems and/or 
data to be ensured and,  more generally, to use and operate it in the 
same conditions as regards security. 

The fact that you are presently reading this means that you have had
knowledge of the CeCILL license and that you accept its terms.

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
ComputeQEforRow(const int j,
                const double Hsmallr,
                const int Hnx,
                const int Hnxt,
                const int Hnyt,
                const int Hnxyt,
                const int Hnvar,
                const int slices, const int Hstep, 
		double *restrict uold, double *restrict q, double *restrict e
  ) {
  int i, s;
  double eken;

#define IDX(i,j,k)    ( ((i)*Hstep*Hnxyt) + ((j)*Hnxyt) + (k) )
#define IDXE(i,j)     ( ((i)*Hnxyt) + (j) )

#define IHV(i, j, v)  ((i) + Hnxt * ((j) + Hnyt * (v)))

#pragma acc parallel pcopy(q[0:Hnvar*Hstep*Hnxyt]) pcopyout(e[0:Hstep*Hnxyt]) pcopyin(uold[0:Hnvar*Hnxt*Hnyt])
#pragma acc loop gang
  for (s = 0; s < slices; s++) {
#pragma acc loop vector
    for (i = 0; i < Hnx; i++) {
      int idxuID = IHV(i + ExtraLayer, j + s, ID);
      int idxuIU = IHV(i + ExtraLayer, j + s, IU);
      int idxuIV = IHV(i + ExtraLayer, j + s, IV);
      int idxuIP = IHV(i + ExtraLayer, j + s, IP);
      q[IDX(ID,s,i)] = MAX(uold[idxuID], Hsmallr);
      q[IDX(IU,s,i)] = uold[idxuIU] / q[IDX(ID,s,i)];
      q[IDX(IV,s,i)] = uold[idxuIV] / q[IDX(ID,s,i)];
      eken = half * (Square(q[IDX(IU,s,i)]) + Square(q[IDX(IV,s,i)]));
      q[IDX(IP,s,i)] = uold[idxuIP] / q[IDX(ID,s,i)] - eken;
      e[IDXE(s,i)] = q[IDX(IP,s,i)];
    }
  }
#undef IHV
#undef IHVW
#undef IDX
#undef IDXE
}

static void
courantOnXY(double *restrict cournox,
	    double *restrict cournoy,
            const int Hnx,
            const int Hnxyt,
            const int Hnvar, const int slices, const int Hstep, 
	    double *restrict c, double *restrict q
  ) {
  double dcournox = *cournox, dcournoy = *cournoy; 
  int i, s;
  // double maxValC = zero;
  double tmp1, tmp2;

#define IDX(i,j,k)    ( ((i)*Hstep*Hnxyt) + ((j)*Hnxyt) + (k) )
#define IDXE(i,j)     ( ((i)*Hnxyt) + (j) )

  // #define IHVW(i,v) ((i) + (v) * nxyt)
  //     maxValC = c[0];
  //     for (i = 0; i < Hnx; i++) {
  //         maxValC = MAX(maxValC, c[i]);
  //     }
  //     for (i = 0; i < Hnx; i++) {
  //         *cournox = MAX(*cournox, maxValC + DABS(q[IU][i]));
  //         *cournoy = MAX(*cournoy, maxValC + DABS(q[IV][i]));
  //     }

#pragma acc parallel pcopyin(q[0:Hnvar*Hstep*Hnxyt], c[0:Hstep*Hnxyt]) 
#pragma acc loop reduction(max:dcournox,dcournoy) 
  for (s = 0; s < slices; s++) {
    for (i = 0; i < Hnx; i++) {
      tmp1 = c[IDXE(s,i)] + DABS(q[IDX(IU,s,i)]);
      tmp2 = c[IDXE(s,i)] + DABS(q[IDX(IV,s,i)]);
      dcournox = fmax(dcournox, tmp1);
      dcournoy = fmax(dcournoy, tmp2);
    }
  }

  *cournox = dcournox;
  *cournoy = dcournoy;

#undef IHVW
#undef IDX
#undef IDXE
}
void
compute_deltat(double *dt, const hydroparam_t H, hydrowork_t * Hw, hydrovar_t * Hv, hydrovarwork_t * Hvw) {
  double cournox, cournoy;
  int j, jend, slices, Hstep, Hmin, Hmax;
  double *restrict e;
  double *restrict c;
  double *restrict q;
  double *restrict uold;
  WHERE("compute_deltat");

  //   compute time step on grid interior
  cournox = zero;
  cournoy = zero;
  Hvw->q = (double (*)) calloc(H.nvar * H.nxystep * H.nxyt, sizeof(double));
  Hw->e = (double (*)) malloc((H.nxyt) * H.nxystep * sizeof(double));
  Hw->c = (double (*)) malloc((H.nxyt) * H.nxystep * sizeof(double));

  c = Hw->c;
  e = Hw->e;
  q = Hvw->q;
  uold = Hv->uold;

  {
  Hstep = H.nxystep;
  Hmin = H.jmin + ExtraLayer;
  Hmax = H.jmax - ExtraLayer;

#pragma acc data create(q[0:H.nvar*Hstep*H.nxyt], e[0:Hstep*H.nxyt], c[0:Hstep*H.nxyt]) \
                 present(uold[0:H.nvar*H.nxt*H.nyt]) 
  for (j = Hmin; j < Hmax; j += Hstep) {
    jend = j + Hstep;
    if (jend >= Hmax)
      jend = Hmax;
    slices = jend - j;          // numbre of slices to compute
    ComputeQEforRow(j, H.smallr, H.nx, H.nxt, H.nyt, H.nxyt, H.nvar, slices, Hstep, Hv->uold, q, e);
    equation_of_state(0, H.nx, H.nxyt, H.nvar, H.smallc, H.gamma, slices, Hstep, e, q, c);
    courantOnXY(&cournox, &cournoy, H.nx, H.nxyt, H.nvar, slices, Hstep, c, q);

#ifdef FLOPS
    flops += 10;
#endif /*  */
  }
  }
  Free(Hvw->q);
  Free(Hw->e);
  Free(Hw->c);
  *dt = H.courant_factor * H.dx / MAX(cournox, MAX(cournoy, H.smallc));

#ifdef FLOPS
  flops += 2;

#endif /*  */

  // fprintf(stdout, "%g %g %g %g\n", cournox, cournoy, H.smallc, H.courant_factor);
}                               // compute_deltat

//EOF
