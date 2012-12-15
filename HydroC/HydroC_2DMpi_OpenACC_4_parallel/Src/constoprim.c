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

#ifndef HMPP
#include "parametres.h"
#include "constoprim.h"
#include "utils.h"

#define CFLOPS(c)               /* {flops+=c;} */
#define IDX(i,j,k)    ( (i*Hstep*Hnxyt) + (j*Hnxyt) + k )
#define IDXE(i,j)     ( (i*Hnxyt) + j )


void
constoprim(const int n,
           const int Hnxyt,
           const int Hnvar,
           const double Hsmallr,
           const int slices, const int Hstep,
           double *u, double *q, double *e) {
  //double u[Hnvar][Hstep][Hnxyt], double q[Hnvar][Hstep][Hnxyt], double e[Hstep][Hnxyt]) {
  int ijmin, ijmax, IN, i, s;
  double eken;
  // const int nxyt = Hnxyt;
  WHERE("constoprim");
  ijmin = 0;
  ijmax = n;

#pragma acc parallel pcopyin(u[0:Hnvar*Hstep*Hnxyt]) pcopyout(q[0:Hnvar*Hstep*Hnxyt], e[0:Hstep*Hnxyt])
#pragma acc loop gang 
  for (s = 0; s < slices; s++) {
#pragma acc loop vector
    for (i = ijmin; i < ijmax; i++) {
      double qid = MAX(u[IDX(ID,s,i)], Hsmallr);
      q[IDX(ID,s,i)] = qid;

      double qiu = u[IDX(IU,s,i)] / qid;
      double qiv = u[IDX(IV,s,i)] / qid;
      q[IDX(IU,s,i)] = qiu;
      q[IDX(IV,s,i)] = qiv;

      eken = half * (Square(qiu) + Square(qiv));

      double qip = u[IDX(IP,s,i)] / qid - eken;
      q[IDX(IP,s,i)] = qip;
      e[IDXE(s,i)] = qip;

      CFLOPS(9);
    }
  }

  if (Hnvar > IP) {
#pragma acc parallel pcopyin(u[0:Hnvar*Hstep*Hnxyt]) pcopy(q[0:Hnvar*Hstep*Hnxyt])
#pragma acc loop gang collapse(2)
    for (IN = IP + 1; IN < Hnvar; IN++) {
      for (s = 0; s < slices; s++) {
#pragma acc loop vector
        for (i = ijmin; i < ijmax; i++) {
          q[IDX(IN,s,i)] = u[IDX(IN,s,i)] / q[IDX(IN,s,i)];
          CFLOPS(1);
        }
      }
    }
  }
}                               // constoprim


#undef IHVW
#undef IDX
#undef IDXE
#endif
//EOF
