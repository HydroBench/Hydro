#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "cclock.h"
#include "parametres.h"
#include "perfcnt.h"
#include "slope.h"
#include "utils.h"

void slope(const int n, const int Hnvar, const int Hnxyt, const real_t Hslope_type,
           const int slices, const int Hstep, real_t q[Hnvar][Hstep][Hnxyt],
           real_t dq[Hnvar][Hstep][Hnxyt]) {
    struct timespec start, end;
    int nbv, i, ijmin, ijmax, s;
    // long ihvwin, ihvwimn, ihvwipn;
    // #define IHVW(i, v) ((i) + (v) * Hnxyt)

    WHERE("slope");
#ifdef TRACKDATA
    fprintf(stderr, "Moving slope IN\n");
#endif
    start = cclock();
    ijmin = 0;
    ijmax = n;
#ifdef TARGETON
#pragma omp target map(q [0:Hnvar] [0:Hstep] [0:Hnxyt], dq [0:Hnvar] [0:Hstep] [0:Hnxyt])
#endif

#ifdef LOOPFORM
#pragma omp teams loop bind(teams) private(nbv, s, i) collapse(3)
#else
#pragma omp TEAMSDIS parallel for default(none) collapse(3) private(nbv, s, i) shared(q, dq)       \
    firstprivate(Hslope_type, ijmin, ijmax, slices, Hnvar)
#endif
    for (s = 0; s < slices; s++) {
        for (nbv = 0; nbv < Hnvar; nbv++) {
            for (i = ijmin + 1; i < ijmax - 1; i++) {
                real_t dlft, drgt, dcen, dsgn, slop, dlim;
                int llftrgt = 0;
                real_t t1;
                dlft = Hslope_type * (q[nbv][s][i] - q[nbv][s][i - 1]);
                drgt = Hslope_type * (q[nbv][s][i + 1] - q[nbv][s][i]);
                dcen = half * (dlft + drgt) / Hslope_type;
                dsgn = (dcen > zero) ? one : -one; // sign(one, dcen);
                llftrgt = ((dlft * drgt) <= zero);
                t1 = FMIN(FABS(dlft), FABS(drgt));
                dq[nbv][s][i] = dsgn * FMIN((real_t)(1 - llftrgt) * t1, FABS(dcen));
            }
        }
    }
    {
        int nops = Hnvar * slices * ((ijmax - 1) - (ijmin + 1));
        FLOPS(8 * nops, 1 * nops, 6 * nops, 0 * nops);
    }

    end = cclock();
    functim[TIM_SLOPE] += ccelaps(start, end);
#ifdef TRACKDATA
    fprintf(stderr, "Moving slope OUT\n");
#endif
} // slope

#undef IHVW

// EOF
