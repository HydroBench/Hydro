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

#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <stdio.h>

#include "parametres.h"
#include "utils.h"
#include "trace.h"

#ifndef HMPP
#define CFLOPS(c)               /* {flops+=c;} */
#define IDX(i,j,k) ( (i*Hstep*Hnxyt) + (j*Hnxyt) + k )
#define IDXE(i,j) ( (i*Hnxyt) + j )

void
trace(const double dtdx,
      const int n,
      const int Hscheme,
      const int Hnvar,
      const int Hnxyt,
      const int slices, const int Hstep,
      double *q,
      double *dq,
      double *c, double *qxm, double *qxp) {
  //double q[Hnvar][Hstep][Hnxyt],
  //double dq[Hnvar][Hstep][Hnxyt], double c[Hstep][Hnxyt], double qxm[Hnvar][Hstep][Hnxyt],
  //double qxp[Hnvar][Hstep][Hnxyt]) {
  int ijmin, ijmax;
  int i, IN, s;
  double cc, csq, r, u, v, p, a;
  double dr, du, dv, dp, da;
  double alpham, alphap, alpha0r, alpha0v;
  double spminus, spplus, spzero;
  double apright, amright, azrright, azv1right, acmpright;
  double apleft, amleft, azrleft, azv1left, acmpleft;
  double zerol = 0.0, zeror = 0.0, project = 0.;

  WHERE("trace");
  ijmin = 0;
  ijmax = n;

  // if (strcmp(Hscheme, "muscl") == 0) {       // MUSCL-Hancock method
  if (Hscheme == HSCHEME_MUSCL) {       // MUSCL-Hancock method
    zerol = -hundred / dtdx;
    zeror = hundred / dtdx;
    project = one;
    CFLOPS(2);
  }
  // if (strcmp(Hscheme, "plmde") == 0) {       // standard PLMDE
  if (Hscheme == HSCHEME_PLMDE) {       // standard PLMDE
    zerol = zero;
    zeror = zero;
    project = one;
  }
  // if (strcmp(Hscheme, "collela") == 0) {     // Collela's method
  if (Hscheme == HSCHEME_COLLELA) {     // Collela's method
    zerol = zero;
    zeror = zero;
    project = zero;
  }

#pragma acc parallel pcopyin(q[0:Hnvar*Hstep*Hnxyt], dq[0:Hnvar*Hstep*Hnxyt], c[0:Hstep*Hnxyt]) pcopyout(qxm[0:Hnvar*Hstep*Hnxyt], qxp[0:Hnvar*Hstep*Hnxyt])
#pragma acc loop gang
  for (s = 0; s < slices; s++) {
#pragma acc loop vector
    for (i = ijmin + 1; i < ijmax - 1; i++) {
      cc = c[IDXE(s,i)];
      csq = Square(cc);
      r = q[IDX(ID,s,i)];
      u = q[IDX(IU,s,i)];
      v = q[IDX(IV,s,i)];
      p = q[IDX(IP,s,i)];
      dr = dq[IDX(ID,s,i)];
      du = dq[IDX(IU,s,i)];
      dv = dq[IDX(IV,s,i)];
      dp = dq[IDX(IP,s,i)];
      alpham = half * (dp / (r * cc) - du) * r / cc;
      alphap = half * (dp / (r * cc) + du) * r / cc;
      alpha0r = dr - dp / csq;
      alpha0v = dv;

      // Right state
      spminus = (u - cc) * dtdx + one;
      spplus = (u + cc) * dtdx + one;
      spzero = u * dtdx + one;
      if ((u - cc) >= zeror) {
        spminus = project;
      }
      if ((u + cc) >= zeror) {
        spplus = project;
      }
      if (u >= zeror) {
        spzero = project;
      }
      apright = -half * spplus * alphap;
      amright = -half * spminus * alpham;
      azrright = -half * spzero * alpha0r;
      azv1right = -half * spzero * alpha0v;
      qxp[IDX(ID,s,i)] = r + (apright + amright + azrright);
      qxp[IDX(IU,s,i)] = u + (apright - amright) * cc / r;
      qxp[IDX(IV,s,i)] = v + (azv1right);
      qxp[IDX(IP,s,i)] = p + (apright + amright) * csq;

      // Left state
      spminus = (u - cc) * dtdx - one;
      spplus = (u + cc) * dtdx - one;
      spzero = u * dtdx - one;
      if ((u - cc) <= zerol) {
        spminus = -project;
      }
      if ((u + cc) <= zerol) {
        spplus = -project;
      }
      if (u <= zerol) {
        spzero = -project;
      }
      apleft = -half * spplus * alphap;
      amleft = -half * spminus * alpham;
      azrleft = -half * spzero * alpha0r;
      azv1left = -half * spzero * alpha0v;
      qxm[IDX(ID,s,i)] = r + (apleft + amleft + azrleft);
      qxm[IDX(IU,s,i)] = u + (apleft - amleft) * cc / r;
      qxm[IDX(IV,s,i)] = v + (azv1left);
      qxm[IDX(IP,s,i)] = p + (apleft + amleft) * csq;

      CFLOPS(78);
    }
  }

  if (Hnvar > IP) {
#pragma acc parallel pcopyin(q[0:Hnvar*Hstep*Hnxyt], dq[0:Hnvar*Hstep*Hnxyt]) pcopyout(qxp[0:Hnvar*Hstep*Hnxyt], qxm[0:Hnvar*Hstep*Hnxyt])
#pragma acc loop gang collapse(2)
    for (IN = IP + 1; IN < Hnvar; IN++) {
      for (s = 0; s < slices; s++) {
#pragma acc loop vector
        for (i = ijmin + 1; i < ijmax - 1; i++) {
          u = q[IDX(IU,s,i)];
          a = q[IDX(IN,s,i)];
          da = dq[IDX(IN,s,i)];

          // Right state
          spzero = u * dtdx + one;
          if (u >= zeror) {
            spzero = project;
          }
          acmpright = -half * spzero * da;
          qxp[IDX(IN,s,i)] = a + acmpright;

          // Left state
          spzero = u * dtdx - one;
          if (u <= zerol) {
            spzero = -project;
          }
          acmpleft = -half * spzero * da;
          qxm[IDX(IN,s,i)] = a + acmpleft;

          CFLOPS(10);
        }
      }
    }
  }
}                               // trace

#undef IDX
#undef IDXE

#endif /* HMPP */

//EOF
