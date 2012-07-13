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

#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <stdio.h>

#include "parametres.h"
#include "utils.h"
#include "trace.h"
void
trace(double *RESTRICT q, double *RESTRICT dq, double *RESTRICT c,
      double *RESTRICT qxm, double *RESTRICT qxp,
      const double dtdx, const long n, const long Hscheme, const long Hnvar, const long Hnxyt)
{
  long ijmin, ijmax;
  long i, IN;
  double cc, csq, r, u, v, p, a;
  double dr, du, dv, dp, da;
  double alpham, alphap, alpha0r, alpha0v;
  double spminus, spplus, spzero;
  double apright, amright, azrright, azv1right, acmpright;
  double apleft, amleft, azrleft, azv1left, acmpleft;
  double zerol = 0.0, zeror = 0.0, project = 0.;

#define IHVW(i, v) ((i) + (v) * Hnxyt)

  WHERE("trace");
  ijmin = 0;
  ijmax = n;

  // if (strcmp(Hscheme, "muscl") == 0) {       // MUSCL-Hancock method
  if (Hscheme == HSCHEME_MUSCL) {       // MUSCL-Hancock method
    zerol = -hundred / dtdx;
    zeror = hundred / dtdx;
    project = one;
    MFLOPS(0, 2, 0, 0);
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

  for (i = ijmin + 1; i < ijmax - 1; i++) {
    cc = c[i];
    csq = Square(cc);
    r = q[IHVW(i, ID)];
    u = q[IHVW(i, IU)];
    v = q[IHVW(i, IV)];
    p = q[IHVW(i, IP)];
    dr = dq[IHVW(i, ID)];
    du = dq[IHVW(i, IU)];
    dv = dq[IHVW(i, IV)];
    dp = dq[IHVW(i, IP)];
    alpham = half * (dp / (r * cc) - du) * r / cc;
    alphap = half * (dp / (r * cc) + du) * r / cc;
    alpha0r = dr - dp / csq;
    alpha0v = dv;
    MFLOPS(9, 5, 0, 0);

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
    qxp[IHVW(i, ID)] = r + (apright + amright + azrright);
    qxp[IHVW(i, IU)] = u + (apright - amright) * cc / r;
    qxp[IHVW(i, IV)] = v + (azv1right);
    qxp[IHVW(i, IP)] = p + (apright + amright) * csq;
    MFLOPS(27, 1, 0, 0);

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
    qxm[IHVW(i, ID)] = r + (apleft + amleft + azrleft);
    qxm[IHVW(i, IU)] = u + (apleft - amleft) * cc / r;
    qxm[IHVW(i, IV)] = v + (azv1left);
    qxm[IHVW(i, IP)] = p + (apleft + amleft) * csq;
    MFLOPS(26, 1, 0, 0);
  }
  if (Hnvar > IP + 1) {
    for (IN = IP + 1; IN < Hnvar; IN++) {
      for (i = ijmin + 1; i < ijmax - 1; i++) {
        u = q[IHVW(i, IU)];
        a = q[IHVW(i, IN)];
        da = dq[IHVW(i, IN)];

        // Right state
        spzero = u * dtdx + one;
        if (u >= zeror) {
          spzero = project;
        }
        acmpright = -half * spzero * da;
        qxp[IHVW(i, IN)] = a + acmpright;

        // Left state
        spzero = u * dtdx - one;
        if (u <= zerol) {
          spzero = -project;
        }
        acmpleft = -half * spzero * da;
        qxm[IHVW(i, IN)] = a + acmpleft;
        MFLOPS(10, 0, 0, 0);
      }
    }
  }
}                               // trace

#undef IHVW

//EOF
