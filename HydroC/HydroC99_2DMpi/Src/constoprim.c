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

#ifndef HMPP
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

#pragma omp parallel for private(i, s, eken), shared(q,e) COLLAPSE
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
#endif
//EOF
