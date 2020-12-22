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
#include "slope.h"
#include "perfcnt.h"

#ifndef HMPP

#define DABS(x) (real_t) fabs((x))

#ifdef TARGETON
#pragma message "TARGET on SLOPE begin declare"
#endif

void
slope (const int n,
       const int Hnvar,
       const int Hnxyt,
       const real_t Hslope_type,
       const int slices, const int Hstep, real_t q[Hnvar][Hstep][Hnxyt],
       real_t dq[Hnvar][Hstep][Hnxyt])
{
	int nbv, i, ijmin, ijmax, s;
	// long ihvwin, ihvwimn, ihvwipn;
	// #define IHVW(i, v) ((i) + (v) * Hnxyt)

	WHERE ("slope");
	ijmin = 0;
	ijmax = n;
	// for (s = 0; s < slices; s++) {
	// 	for (nbv = 0; nbv < Hnvar; nbv++) {
	// 		for (i = ijmin + 1; i < ijmax - 1; i++) {
	// 			printf("ORG Hslope_type, qi, qim1, qip1, dq: %lg %lg %lg %lg %lg \n", Hslope_type, q[nbv][s][i], q[nbv][s][i - 1], q[nbv][s][i + 1], dq[nbv][s][i]);
	// 		}
	// 	}
	// }
// #ifdef TARGETON
// #pragma omp target teams distribute parallel for private(nbv, s, i) shared(dq) map(alloc: dq[0:Hnvar][0:Hstep][ijmin + 1:ijmax - 1])
// 	for (s = 0; s < slices; s++) {
// 		for (nbv = 0; nbv < Hnvar; nbv++) {
// 			for (i = ijmin + 1; i < ijmax - 1; i++) {
// 				dq[nbv][s][i] = 0.0;
// 			}
// 		}
// 	}
// #endif
#ifdef TARGETON
#pragma omp target teams distribute parallel for private(nbv, s, i) shared(q, dq) \
	map(to: q[0:Hnvar][0:Hstep][ijmin + 1:ijmax - 1]) \
	map(from: dq[0:Hnvar][0:Hstep][ijmin + 1:ijmax - 1]) collapse(3)
	{
// #pragma omp teams distribute parallel for private(nbv, s, i) shared(dq) 
#else
#pragma omp parallel for private(nbv, s, i) shared(dq) collapse(3) // COLLAPSE
		{
#endif
	for (s = 0; s < slices; s++)
	{
		for (nbv = 0; nbv < Hnvar; nbv++)
		{
			for (i = ijmin + 1; i < ijmax - 1; i++)
			{
				real_t dlft, drgt, dcen, dsgn, slop, dlim;
				int llftrgt = 0;
				real_t t1;
				// printf("GPU Hslope_type, qi, qim1, qip1: %lg %lg %lg %lg %lg \n", Hslope_type, q[nbv][s][i], q[nbv][s][i - 1], q[nbv][s][i + 1], dq[nbv][s][i]);
				dlft = Hslope_type * (q[nbv][s][i] - q[nbv][s][i - 1]);
				drgt = Hslope_type * (q[nbv][s][i + 1] - q[nbv][s][i]);
				dcen = half * (dlft + drgt) / Hslope_type;
				dsgn = (dcen > zero) ? (real_t) 1.0 : (real_t) - 1.0;	// sign(one, dcen);
				llftrgt = ((dlft * drgt) <= zero);
				t1 = fmin (fabs (dlft), fabs (drgt));
				dq[nbv][s][i] = dsgn * fmin ((1 - llftrgt) * t1, fabs (dcen));
			}
		}
	}
		}
	{
		int nops = Hnvar * slices * ((ijmax - 1) - (ijmin + 1));
		FLOPS (8 * nops, 1 * nops, 6 * nops, 0 * nops);
	}
}				// slope

#ifdef TARGET
#pragma message "TARGET on SLOPE end declare"
#pragma omp end declare target
#endif
#undef IHVW

#endif /* HMPP */
//EOF
