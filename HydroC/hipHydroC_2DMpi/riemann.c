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
#include "riemann.h"

#define DABS(x) (double) fabs((x))

void
riemann(double *RESTRICT qleft, double *RESTRICT qright,
        double *RESTRICT qgdnv, double *RESTRICT rl,
        double *RESTRICT ul, double *RESTRICT pl, double *RESTRICT cl,
        double *RESTRICT wl, double *RESTRICT rr, double *RESTRICT ur,
        double *RESTRICT pr, double *RESTRICT cr, double *RESTRICT wr,
        double *RESTRICT ro, double *RESTRICT uo, double *RESTRICT po,
        double *RESTRICT co, double *RESTRICT wo,
        double *RESTRICT rstar, double *RESTRICT ustar,
        double *RESTRICT pstar, double *RESTRICT cstar,
        long *RESTRICT sgnm, double *RESTRICT spin,
        double *RESTRICT spout, double *RESTRICT ushock,
        double *RESTRICT frac, double *RESTRICT scr,
        double *RESTRICT delp, double *RESTRICT pold,
        long *RESTRICT ind, long *RESTRICT ind2,
        const long narray,
        const double Hsmallr,
        const double Hsmallc, const double Hgamma, const long Hniter_riemann, const long Hnvar, const long Hnxyt)
{

  // Local variables
  double smallp, gamma6, ql, qr, usr, usl, wwl, wwr, smallpp;
  long i, invar, iter, nface;
#define IHVW(i, v) ((i) + (v) * Hnxyt)

  WHERE("riemann");

  // Constants
  nface = narray;
  smallp = Square(Hsmallc) / Hgamma;
  smallpp = Hsmallr * smallp;
  gamma6 = (Hgamma + one) / (two * Hgamma);

  // Pressure, density and velocity
  for (i = 0; i < nface; i++) {
    rl[i] = MAX(qleft[IHVW(i, ID)], Hsmallr);
    ul[i] = qleft[IHVW(i, IU)];
    pl[i] = MAX(qleft[IHVW(i, IP)], (double) (rl[i] * smallp));
    rr[i] = MAX(qright[IHVW(i, ID)], Hsmallr);
    ur[i] = qright[IHVW(i, IU)];
    pr[i] = MAX(qright[IHVW(i, IP)], (double) (rr[i] * smallp));
    // Lagrangian sound speed
    cl[i] = Hgamma * pl[i] * rl[i];
    cr[i] = Hgamma * pr[i] * rr[i];
    // First guess
    wl[i] = sqrt(cl[i]);
    wr[i] = sqrt(cr[i]);
    pstar[i] = ((wr[i] * pl[i] + wl[i] * pr[i]) + wl[i] * wr[i] * (ul[i] - ur[i])) / (wl[i] + wr[i]);
    pstar[i] = MAX(pstar[i], 0.0);
    pold[i] = pstar[i];
    // ind est un masque de traitement pour le newton
    ind[i] = 1;                 // toutes les cellules sont a traiter
  }

  // Newton-Raphson iterations to find pstar at the required accuracy
  for (iter = 0; iter < Hniter_riemann; iter++) {
    double precision = 1.e-6;
    for (i = 0; i < nface; i++) {
      if (ind[i] == 1) {
        wwl = sqrt(cl[i] * (one + gamma6 * (pold[i] - pl[i]) / pl[i]));
        wwr = sqrt(cr[i] * (one + gamma6 * (pold[i] - pr[i]) / pr[i]));
        ql = two * wwl * Square(wwl) / (Square(wwl) + cl[i]);
        qr = two * wwr * Square(wwr) / (Square(wwr) + cr[i]);
        usl = ul[i] - (pold[i] - pl[i]) / wwl;
        usr = ur[i] + (pold[i] - pr[i]) / wwr;
        delp[i] = MAX((double) (qr * ql / (qr + ql) * (usl - usr)), (double) (-pold[i]));
        pold[i] = pold[i] + delp[i];
        uo[i] = DABS(delp[i] / (pold[i] + smallpp));
        if (uo[i] <= precision) {
          ind[i] = 0;           // cellule qui n'est plus a considerer
          MFLOPS(28, 8, 1, 0);
        }
      }
    }
  }                             // iter_riemann

  for (i = 0; i < nface; i++) {
    pstar[i] = pold[i];
  }
  for (i = 0; i < nface; i++) {
    wl[i] = sqrt(cl[i] * (one + gamma6 * (pstar[i] - pl[i]) / pl[i]));
    wr[i] = sqrt(cr[i] * (one + gamma6 * (pstar[i] - pr[i]) / pr[i]));
    MFLOPS(8, 4, 0, 0);
  }
  for (i = 0; i < nface; i++) {
    ustar[i] = half * (ul[i] + (pl[i] - pstar[i]) / wl[i] + ur[i] - (pr[i] - pstar[i]) / wr[i]);
    sgnm[i] = (ustar[i] > 0) ? 1 : -1;
    if (sgnm[i] == 1) {
      ro[i] = rl[i];
      uo[i] = ul[i];
      po[i] = pl[i];
      wo[i] = wl[i];
    } else {
      ro[i] = rr[i];
      uo[i] = ur[i];
      po[i] = pr[i];
      wo[i] = wr[i];
    }
    co[i] = MAX(Hsmallc, sqrt(DABS(Hgamma * po[i] / ro[i])));
    rstar[i] = ro[i] / (one + ro[i] * (po[i] - pstar[i]) / Square(wo[i]));
    rstar[i] = MAX(rstar[i], Hsmallr);
    cstar[i] = MAX(Hsmallc, sqrt(DABS(Hgamma * pstar[i] / rstar[i])));
    spout[i] = co[i] - sgnm[i] * uo[i];
    spin[i] = cstar[i] - sgnm[i] * ustar[i];
    ushock[i] = wo[i] / ro[i] - sgnm[i] * uo[i];
    if (pstar[i] >= po[i]) {
      spin[i] = ushock[i];
      spout[i] = ushock[i];
    }
    scr[i] = MAX((double) (spout[i] - spin[i]), (double) (Hsmallc + DABS(spout[i] + spin[i])));
    frac[i] = (one + (spout[i] + spin[i]) / scr[i]) * half;
    frac[i] = MAX(zero, (double) (MIN(one, frac[i])));
    MFLOPS(24, 10, 8, 0);
  }

  for (i = 0; i < nface; i++) {
    qgdnv[IHVW(i, ID)] = frac[i] * rstar[i] + (one - frac[i]) * ro[i];
    if (spout[i] < zero) {
      qgdnv[IHVW(i, ID)] = ro[i];
    }
    if (spin[i] > zero) {
      qgdnv[IHVW(i, ID)] = rstar[i];
    }
    MFLOPS(4, 0, 0, 0);
  }

  for (i = 0; i < nface; i++) {
    qgdnv[IHVW(i, IU)] = frac[i] * ustar[i] + (one - frac[i]) * uo[i];
    if (spout[i] < zero) {
      qgdnv[IHVW(i, IU)] = uo[i];
    }
    if (spin[i] > zero) {
      qgdnv[IHVW(i, IU)] = ustar[i];
    }
    MFLOPS(4, 0, 0, 0);
  }


  for (i = 0; i < nface; i++) {
    qgdnv[IHVW(i, IP)] = frac[i] * pstar[i] + (one - frac[i]) * po[i];
    if (spout[i] < zero) {
      qgdnv[IHVW(i, IP)] = po[i];
    }
    if (spin[i] > zero) {
      qgdnv[IHVW(i, IP)] = pstar[i];
    }
    MFLOPS(4, 0, 0, 0);
  }

// transverse velocity
  for (i = 0; i < nface; i++) {
    if (sgnm[i] == 1) {
      qgdnv[IHVW(i, IV)] = qleft[IHVW(i, IV)];
    }
  }
  for (i = 0; i < nface; i++) {
    if (sgnm[i] != 1) {
      qgdnv[IHVW(i, IV)] = qright[IHVW(i, IV)];
    }
  }

// other passive variables
  if (Hnvar > IP + 1) {
    for (invar = IP + 1; invar < Hnvar; invar++) {
      for (i = 0; i < nface; i++) {
        if (sgnm[i] == 1) {
          qgdnv[IHVW(i, invar)] = qleft[IHVW(i, invar)];
        }
      }
      for (i = 0; i < nface; i++) {
        if (sgnm[i] != 1) {
          qgdnv[IHVW(i, invar)] = qright[IHVW(i, invar)];
        }
      }
    }
  }
}                               // riemann


//EOF
