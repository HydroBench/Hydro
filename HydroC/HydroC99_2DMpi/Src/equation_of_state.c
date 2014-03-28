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

// #include <stdlib.h>
// #include <unistd.h>
#include <math.h>
#include <stdio.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#ifndef HMPP
#include "equation_of_state.h"
#include "parametres.h"
#include "perfcnt.h"
#include "utils.h"

void
equation_of_state(int imin,
                  int imax,
                  const int Hnxyt,
                  const int Hnvar,
                  const real_t Hsmallc,
                  const real_t Hgamma,
                  const int slices, const int Hstep,
                  real_t eint[Hstep][Hnxyt], real_t q[Hnvar][Hstep][Hnxyt], real_t c[Hstep][Hnxyt]) {
  int k, s;
	int inpar = 0;
  real_t smallp;

  WHERE("equation_of_state");
  smallp = Square(Hsmallc) / Hgamma;
   FLOPS(1, 1, 0, 0);

  // printf("EOS: %d %d %d %d %g %g %d %d\n", imin, imax, Hnxyt, Hnvar, Hsmallc, Hgamma, slices, Hstep);
#ifdef _OPENMP
	inpar = omp_in_parallel();
	//#pragma omp parallel for if (!inpar) schedule(auto) private(s,k), shared(c,q), collapse(2)
#pragma omp parallel for  private(s,k), shared(c,q) COLLAPSE
#endif
  for (s = 0; s < slices; s++) {
    for (k = imin; k < imax; k++) {
      register real_t rhok = q[ID][s][k];
      register real_t base = (Hgamma - one) * rhok * eint[s][k];
      base = MAX(base, (real_t) (rhok * smallp));

      q[IP][s][k] = base;
      c[s][k] = sqrt(Hgamma * base / rhok);
    }
  }
  { 
    int nops = slices * (imax - imin);
    FLOPS(5 * nops, 2 * nops, 1 * nops, 0 * nops);
  }
}                               // equation_of_state


#endif
// EOF
