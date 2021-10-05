#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "cclock.h"
#include "parametres.h"
#include "qleftright.h"
#include "utils.h"

void qleftright(const int idim, const int Hnx, const int Hny, const int Hnxyt, const int Hnvar,
                const int slices, const int Hstep, real_t qxm[Hnvar][Hstep][Hnxyt],
                real_t qxp[Hnvar][Hstep][Hnxyt], real_t qleft[Hnvar][Hstep][Hnxyt],
                real_t qright[Hnvar][Hstep][Hnxyt]) {
    int nvar, i, s;
    int bmax;
    // struct timespec start, end;

    WHERE("qleftright");
#ifdef TRACKDATA
    fprintf(stderr, "Moving qleftright IN\n");
#endif
    // start = cclock();

    if (idim == 1) {
        bmax = Hnx + 1;
    } else {
        bmax = Hny + 1;
    }

#ifdef TARGETON
#pragma omp target map(qxm [0:Hnvar] [0:Hstep] [0:Hnxyt]), map(qxp [0:Hnvar] [0:Hstep] [0:Hnxyt]), \
    map(qleft [0:Hnvar] [0:Hstep] [0:Hnxyt]), map(qright [0:Hnvar] [0:Hstep] [0:Hnxyt])
#endif
#ifdef LOOPFORM
#pragma omp teams loop bind(teams) private(s, i, nvar) collapse(3)
#else
#pragma omp TEAMSDIS parallel for default(none) private(s, i, nvar),                               \
    FIRSTP(slices, Hnvar, bmax, Hnxyt, Hstep) shared(qleft, qright, qxm, qxp) collapse(3)
#endif
    for (nvar = 0; nvar < Hnvar; nvar++) {
        for (s = 0; s < slices; s++) {
            // #pragma omp simd
            for (i = 0; i < bmax; i++) {
                qleft[nvar][s][i] = qxm[nvar][s][i + 1];
                qright[nvar][s][i] = qxp[nvar][s][i + 2];
            }
        }
    }
    // end = cclock();
    // functim[TIM_QLEFTR] += ccelaps(start, end);
#ifdef TRACKDATA
    fprintf(stderr, "Moving qleftright OUT\n");
#endif
    return;
}
// EOF
