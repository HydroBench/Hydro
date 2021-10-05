/*
  A simple 2D hydro code
  (C) Romain Teyssier : CEA/IRFU           -- original F90 code
  (C) Pierre-Francois Lavallee : IDRIS      -- original F90 code
  (C) Guillaume Colin de Verdiere : CEA/DAM -- for the C version
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

#include "parametres.h"
#include "compute_deltat.h"
#include "utils.h"
#include "equation_of_state.h"

#define DABS(x) (double) fabs((x))

void
ComputeQEforRow(const long j, double *uold, double *q, double *e,
                const double Hsmallr, const long Hnx, const long Hnxt, const long Hnyt, const long nxyt)
{
  long i;
  double eken;

#define IHV(i, j, v)  ((i) + Hnxt * ((j) + Hnyt * (v)))
#define IHVW(i, v) ((i) + (v) * nxyt)
  for (i = 0; i < Hnx; i++) {
    long idxuID = IHV(i + ExtraLayer, j, ID);
    long idxuIU = IHV(i + ExtraLayer, j, IU);
    long idxuIV = IHV(i + ExtraLayer, j, IV);
    long idxuIP = IHV(i + ExtraLayer, j, IP);
    q[IHVW(i, ID)] = MAX(uold[idxuID], Hsmallr);
    q[IHVW(i, IU)] = uold[idxuIU] / q[IHVW(i, ID)];
    q[IHVW(i, IV)] = uold[idxuIV] / q[IHVW(i, ID)];
    eken = half * (Square(q[IHVW(i, IU)]) + Square(q[IHVW(i, IV)]));
    q[IHVW(i, IP)] = uold[idxuIP] / q[IHVW(i, ID)] - eken;
    e[i] = q[IHVW(i, IP)];
    MFLOPS(3, 3, 1, 0);
  }
#undef IHV
#undef IHVW
} void
courantOnXY(double *cournox, double *cournoy, const long Hnx, const long nxyt, double *c, double *q)
{
  long i;
  double tmp1, tmp2;

  *cournox = *cournoy = 0.0;

#define IHVW(i,v) ((i) + (v) * nxyt)
//     maxValC = c[0];
//     for (i = 0; i < Hnx; i++) {
//         maxValC = MAX(maxValC, c[i]);
//         MFLOPS(0, 0, 1, 0);
//     }
//     for (i = 0; i < Hnx; i++) {
//         *cournox = MAX(*cournox, maxValC + DABS(q[IHVW(i, IU)]));
//         *cournoy = MAX(*cournoy, maxValC + DABS(q[IHVW(i, IV)]));
//         MFLOPS(2, 0, 4, 0);
//     }

  for (i = 0; i < Hnx; i++) {
    tmp1 = c[i] + DABS(q[IHVW(i, IU)]);
    tmp2 = c[i] + DABS(q[IHVW(i, IV)]);
    *cournox = MAX(*cournox, tmp1);
    *cournoy = MAX(*cournoy, tmp2);
  }
#undef IHVW
}
void
compute_deltat(double *dt, const hydroparam_t H, hydrowork_t * Hw, hydrovar_t * Hv, hydrovarwork_t * Hvw)
{
  double cournox, cournoy;
  long j;
  WHERE("compute_deltat");

#define IHVW(i,v) ((i) + (v) * nxyt)

  //   compute time step on grid interior
  cournox = zero;
  cournoy = zero;
  Hvw->q = (double *) calloc(H.nvar * H.nxyt, sizeof(double));
  Hw->e = (double *) malloc(H.nx * sizeof(double));
  Hw->c = (double *) malloc(H.nx * sizeof(double));
  for (j = H.jmin + ExtraLayer; j < H.jmax - ExtraLayer; j++) {
    ComputeQEforRow(j, Hv->uold, Hvw->q, Hw->e, H.smallr, H.nx, H.nxt, H.nyt, H.nxyt);
    equation_of_state(&Hvw->q[IHvw(0, ID)], Hw->e, &Hvw->q[IHvw(0, IP)], Hw->c, 0, H.nx, H.smallc, H.gamma);
    courantOnXY(&cournox, &cournoy, H.nx, H.nxyt, Hw->c, Hvw->q);
  }
  Free(Hvw->q);
  Free(Hw->e);
  Free(Hw->c);
  *dt = H.courant_factor * H.dx / MAX(cournox, MAX(cournoy, H.smallc));
  MFLOPS(1, 1, 2, 0);
  // fprintf(stdout, "%g %g %g %g\n", cournox, cournoy, H.smallc, H.courant_factor);
#undef IHVW
}                               // compute_deltat


//EOF
