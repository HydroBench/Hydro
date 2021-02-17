#include <malloc.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "compute_deltat.h"
#include "courantOnXY.h"
#include "parametres.h"
#include "perfcnt.h"
#include "utils.h"

void courantOnXY(real_t *cournox, real_t *cournoy, const int Hnx, const int Hnxyt, const int Hnvar,
                 const int slices, const int Hstep, real_t c[Hstep][Hnxyt],
                 real_t q[Hnvar][Hstep][Hnxyt], real_t *tmpm1, real_t *tmpm2) {
    int s, i;
    // real_t maxValC = zero;
    real_t tmp1 = *cournox, tmp2 = *cournoy;
#define NTNT
    
#ifdef TARGETON
#ifdef TRACKDATA
    fprintf(stderr, "Moving courantOnXY IN\n");
#endif // TRACKDATA

#pragma omp target map(tmp1, tmp2) map(c [0:Hstep] [0:Hnxyt]) map(q [0:Hnvar] [0:Hstep] [0:Hnxyt])

#ifdef __INTEL_LLVM_COMPILER
#undef NTNT
    // as of version 2021 1.0 the reduction needs those parameters. Why ??²
#define NTNT num_teams(24) num_threads(16)
#endif // __INTEL_LLVM_COMPILER
#endif // TARGETON
    
#ifdef LOOPFORM
#pragma omp teams loop bind(teams) private(s, i) reduction(max : tmp1, tmp2) collapse(2)
#else
#pragma omp TEAMSDIS parallel for default(none) firstprivate(slices, Hnx, Hnxyt, Hstep) private(s, i) \
    shared(q, c) reduction(max                                                                     \
                           : tmp1, tmp2) NTNT collapse(2)
#endif // LOOPFORM

    for (s = 0; s < slices; s++) {
        for (i = 0; i < Hnx; i++) {
            tmp1 = MAX(tmp1, c[s][i] + FABS(q[IU][s][i]));
            tmp2 = MAX(tmp2, c[s][i] + FABS(q[IV][s][i]));
        }
    }
    *cournox = tmp1;
    *cournoy = tmp2;
#ifdef TRACKDATA
    fprintf(stderr, "Moving courantOnXY OUT\n");
#endif
    int nops = (slices)*Hnx;
    FLOPS(2 * nops, 0 * nops, 2 * nops, 0 * nops);
}
