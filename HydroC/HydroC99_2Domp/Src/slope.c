#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <stdio.h>

#include "parametres.h"
#include "utils.h"
#include "slope.h"
#include "perfcnt.h"

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

//EOF
