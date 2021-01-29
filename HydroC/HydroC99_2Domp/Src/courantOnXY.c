#include <stdio.h>
#include <string.h>
#include <malloc.h>
#include <math.h>

#include "parametres.h"
#include "compute_deltat.h"
#include "utils.h"
#include "perfcnt.h"
#include "courantOnXY.h"

#define DABS(x) (real_t) fabs((x))

void
courantOnXY(real_t * cournox,
	    real_t * cournoy,
	    const int Hnx,
	    const int Hnxyt,
	    const int Hnvar, const int slices, const int Hstep,
	    real_t c[Hstep][Hnxyt], real_t q[Hnvar][Hstep][Hnxyt],
	    real_t * tmpm1, real_t * tmpm2)
{
    int s, i;
    // real_t maxValC = zero;
    real_t tmp1 = *cournox, tmp2 = *cournoy;
#ifdef TARGETON
#ifdef TRACKDATA
    fprintf(stderr, "Moving courantOnXY IN\n");
#endif

#pragma omp target \
	map(tmp1, tmp2)\
	map(c[0:Hstep][0:Hnxyt])		\
	map(q[0:Hnvar][0:Hstep][0:Hnxyt])
#define TD teams distribute
#else
#define TD
#endif
#pragma omp TD parallel for \
    default(none)			 \
    firstprivate(slices, Hnx)		 \
    private(s,i)			 \
    shared(q,c)			 \
    reduction(max:tmp1, tmp2)
    for (s = 0; s < slices; s++) {
#pragma omp simd reduction(max:tmp1, tmp2)
	for (i = 0; i < Hnx; i++) {
	    tmp1 = MAX(tmp1, c[s][i] + DABS(q[IU][s][i]));
	    tmp2 = MAX(tmp2, c[s][i] + DABS(q[IV][s][i]));
	}
    }
    *cournox = tmp1;
    *cournoy = tmp2;
#ifdef TRACKDATA
    fprintf(stderr, "Moving courantOnXY OUT\n");
#endif
    {
	int nops = (slices) * Hnx;
	FLOPS(2 * nops, 0 * nops, 2 * nops, 0 * nops);
    }
#undef IHVW
}
