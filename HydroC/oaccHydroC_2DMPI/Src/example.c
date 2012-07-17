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

#ifndef HMPP
#include "parametres.h"
#include "constoprim.h"
#include "utils.h"

#define CFLOPS(c)               /* {flops+=c;} */

void
constoprim(double *RESTRICT u, double *RESTRICT q, double *RESTRICT e,  /* 0...2 */
           const int n, const int Hnxyt, const int Hnvar, const double Hsmallr) {       /* 3...6 */
  int ijmin, ijmax, IN, i;
  double eken;
  const int nxyt = Hnxyt;
  WHERE("constoprim");
  ijmin = 0;
  ijmax = n;

#define IHVW(i,v) ((i) + (v) * nxyt)

  for (i = ijmin; i < ijmax; i++) {
    double qid = MAX(u[IHVW(i, ID)], Hsmallr);
    q[IHVW(i, ID)] = qid;

    double qiu = u[IHVW(i, IU)] / qid;
    double qiv = u[IHVW(i, IV)] / qid;
    q[IHVW(i, IU)] = qiu;
    q[IHVW(i, IV)] = qiv;

    eken = half * (Square(qiu) + Square(qiv));

    double qip = u[IHVW(i, IP)] / qid - eken;
    q[IHVW(i, IP)] = qip;
    e[i] = qip;

    CFLOPS(10);
  }

  if (Hnvar > IP) {
    for (IN = IP + 1; IN < Hnvar; IN++) {
      for (i = ijmin; i < ijmax; i++) {
        q[IHVW(i, IN)] = u[IHVW(i, IN)] / q[IHVW(i, IN)];
        CFLOPS(1);
      }
    }
  }
}                               // constoprim


#undef IHVW
#endif
//EOF
