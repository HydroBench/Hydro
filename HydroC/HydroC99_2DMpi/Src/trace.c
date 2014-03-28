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
#include "perfcnt.h"

#ifndef HMPP

void
trace(const real_t dtdx,
      const int n,
      const int Hscheme,
      const int Hnvar,
      const int Hnxyt,
      const int slices, const int Hstep,
      real_t q[Hnvar][Hstep][Hnxyt],
      real_t dq[Hnvar][Hstep][Hnxyt], real_t c[Hstep][Hnxyt], real_t qxm[Hnvar][Hstep][Hnxyt],
      real_t qxp[Hnvar][Hstep][Hnxyt]) {
  int ijmin, ijmax;
  int i, IN, s;
  real_t zerol = 0.0, zeror = 0.0, project = 0.;

  WHERE("trace");
  ijmin = 0;
  ijmax = n;

  // if (strcmp(Hscheme, "muscl") == 0) {       // MUSCL-Hancock method
  if (Hscheme == HSCHEME_MUSCL) {       // MUSCL-Hancock method
    zerol = -hundred / dtdx;
    zeror = hundred / dtdx;
    project = one;
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

#pragma omp parallel for private(s,i), shared(qxp, qxm) 
  for (s = 0; s < slices; s++) {
    for (i = ijmin + 1; i < ijmax - 1; i++) {
      real_t cc, csq, r, u, v, p;
      real_t dr, du, dv, dp;
      real_t alpham, alphap, alpha0r, alpha0v;
      real_t spminus, spplus, spzero;
      real_t apright, amright, azrright, azv1right;
      real_t apleft, amleft, azrleft, azv1left;

      real_t upcc, umcc, upccx, umccx, ux;
      real_t rOcc, OrOcc, dprcc;

      cc = c[s][i];
      csq = Square(cc);
      r = q[ID][s][i];
      u = q[IU][s][i];
      v = q[IV][s][i];
      p = q[IP][s][i];
      dr = dq[ID][s][i];
      du = dq[IU][s][i];
      dv = dq[IV][s][i];
      dp = dq[IP][s][i];
      rOcc = r / cc;
      OrOcc = cc / r;
      dprcc = dp / (r * cc);
      alpham = half * (dprcc - du) * rOcc;
      alphap = half * (dprcc + du) * rOcc;
      alpha0r = dr - dp / csq;
      alpha0v = dv;
      upcc = u + cc;
      umcc = u - cc;
      upccx = upcc * dtdx;
      umccx = umcc * dtdx;
      ux = u * dtdx;

      // Right state
      spminus = (umcc >= zeror) ? (project) : umccx + one;
      spplus  = (upcc >= zeror) ? (project) : upccx + one;
      spzero  = (u >= zeror)    ? (project) : ux + one;
      apright = -half * spplus * alphap;
      amright = -half * spminus * alpham;
      azrright = -half * spzero * alpha0r;
      azv1right = -half * spzero * alpha0v;
      qxp[ID][s][i] = r + (apright + amright + azrright);
      qxp[IU][s][i] = u + (apright - amright) * OrOcc;
      qxp[IV][s][i] = v + (azv1right);
      qxp[IP][s][i] = p + (apright + amright) * csq;

      // Left state
      spminus = (umcc <= zerol) ? (-project) : umccx - one;
      spplus  = (upcc <= zerol) ? (-project) : upccx - one;
      spzero  = (u <= zerol)    ? (-project) : ux - one;
      apleft = -half * spplus * alphap;
      amleft = -half * spminus * alpham;
      azrleft = -half * spzero * alpha0r;
      azv1left = -half * spzero * alpha0v;
      qxm[ID][s][i] = r + (apleft + amleft + azrleft);
      qxm[IU][s][i] = u + (apleft - amleft) * OrOcc;
      qxm[IV][s][i] = v + (azv1left);
      qxm[IP][s][i] = p + (apleft + amleft) * csq;
    }
  }

  { 
    int nops = slices * ((ijmax - 1) - (ijmin + 1));
    FLOPS(77 * nops, 7 * nops, 0 * nops, 0 * nops);
  }

  if (Hnvar > IP) {
    for (IN = IP + 1; IN < Hnvar; IN++) {
      for (s = 0; s < slices; s++) {
        for (i = ijmin + 1; i < ijmax - 1; i++) {
	  real_t u, a;
	  real_t da;
	  real_t spzero;
	  real_t acmpright;
	  real_t acmpleft;
          u = q[IU][s][i];
          a = q[IN][s][i];
          da = dq[IN][s][i];

          // Right state
          spzero = u * dtdx + one;
          if (u >= zeror) {
            spzero = project;
          }
          acmpright = -half * spzero * da;
          qxp[IN][s][i] = a + acmpright;

          // Left state
          spzero = u * dtdx - one;
          if (u <= zerol) {
            spzero = -project;
          }
          acmpleft = -half * spzero * da;
          qxm[IN][s][i] = a + acmpleft;
        }
      }
    }
  }
}                               // trace

#endif /* HMPP */

//EOF
