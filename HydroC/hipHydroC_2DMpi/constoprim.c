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
#include "constoprim.h"
#include "utils.h"
void
constoprim(double *RESTRICT u, double *RESTRICT q, double *RESTRICT e,
           const long n, const long Hnxyt, const long Hnvar, const double Hsmallr)
{
  long ijmin, ijmax, IN, i;
  double eken;
  const long nxyt = Hnxyt;
  WHERE("constoprim");
  ijmin = 0;
  ijmax = n;

#define IHVW(i,v) ((i) + (v) * nxyt)
  for (i = ijmin; i < ijmax; i++) {
    q[IHVW(i, ID)] = MAX(u[IHVW(i, ID)], Hsmallr);
    MFLOPS(0, 0, 1, 0);
  }

  for (i = ijmin; i < ijmax; i++) {
    q[IHVW(i, IU)] = u[IHVW(i, IU)] / q[IHVW(i, ID)];
    MFLOPS(0, 1, 0, 0);
  }
  for (i = ijmin; i < ijmax; i++) {
    q[IHVW(i, IV)] = u[IHVW(i, IV)] / q[IHVW(i, ID)];
    MFLOPS(0, 1, 0, 0);
  }
  for (i = ijmin; i < ijmax; i++) {
    eken = half * (Square(q[IHVW(i, IU)]) + Square(q[IHVW(i, IV)]));
    q[IHVW(i, IP)] = u[IHVW(i, IP)] / q[IHVW(i, ID)] - eken;
    MFLOPS(0, 1, 0, 0);
  }
  if (Hnvar > IP + 1) {
    for (IN = IP + 1; IN < Hnvar; IN++) {
      for (i = ijmin; i < ijmax; i++) {
        q[IHVW(i, IN)] = u[IHVW(i, IN)] / q[IHVW(i, IN)];
        MFLOPS(0, 1, 0, 0);
      }
    }
  }
  for (i = ijmin; i < ijmax; i++) {
    e[i] = q[IHVW(i, IP)];
  }
}                               // constoprim


#undef IHVW
//EOF
