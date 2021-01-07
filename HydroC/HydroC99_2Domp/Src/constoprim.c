#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <stdio.h>

#include "parametres.h"
#include "constoprim.h"
#include "perfcnt.h"
#include "utils.h"

void
constoprim(const int n,
           const int Hnxyt,
           const int Hnvar,
           const real_t Hsmallr,
           const int slices, const int Hstep,
           real_t u[Hnvar][Hstep][Hnxyt], real_t q[Hnvar][Hstep][Hnxyt], real_t e[Hstep][Hnxyt]) {
  int ijmin, ijmax, IN, i, s;
  real_t eken;
  // const int nxyt = Hnxyt;
  WHERE("constoprim");
  ijmin = 0;
  ijmax = n;

#ifdef TARGETON
#pragma message "TARGET on CONSTOPRIM"
#pragma omp target				\
	map(to:u[0:Hnvar][0:Hstep][ijmin:ijmax])	\
	map(from:q[0:Hnvar][0:Hstep][ijmin:ijmax])	\
	map(from:e[0:Hstep][ijmin:ijmax])
#pragma omp teams distribute parallel for default(none) private(s, i), shared(u, q, e) collapse(2)
#else
#pragma omp parallel for private(i, s, eken), shared(q,e) COLLAPSE
#endif
  for (s = 0; s < slices; s++) {
    for (i = ijmin; i < ijmax; i++) {
      real_t qid = MAX(u[ID][s][i], Hsmallr);
      q[ID][s][i] = qid;

      real_t qiu = u[IU][s][i] / qid;
      real_t qiv = u[IV][s][i] / qid;
      q[IU][s][i] = qiu;
      q[IV][s][i] = qiv;

      eken = half * (Square(qiu) + Square(qiv));

      real_t qip = u[IP][s][i] / qid - eken;
      q[IP][s][i] = qip;
      e[s][i] = qip;
    }
  }
  { 
    int nops = slices * ((ijmax) - (ijmin));
    FLOPS(5 * nops, 3 * nops, 1 * nops, 0 * nops);
  }

  if (Hnvar > IP) {
    for (IN = IP + 1; IN < Hnvar; IN++) {
      for (s = 0; s < slices; s++) {
        for (i = ijmin; i < ijmax; i++) {
          q[IN][s][i] = u[IN][s][i] / q[IN][s][i];
        }
      }
    }
  }
}                               // constoprim


#undef IHVW
//EOF
