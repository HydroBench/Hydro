#include <malloc.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "ComputeQEforRow.h"
#include "compute_deltat.h"
#include "parametres.h"
#include "perfcnt.h"
#include "utils.h"
void ComputeQEforRow(const int j, const real_t Hsmallr, const int Hnx, const int Hnxt,
                     const int Hnyt, const int Hnxyt, const int Hnvar, const int slices,
                     const int Hstep, real_t *uold, real_t q[Hnvar][Hstep][Hnxyt],
                     real_t e[Hstep][Hnxyt]) {
    int i, s;

#ifdef TRACKDATA
    fprintf(stderr, "Moving ComputeQEforRow IN\n");
#endif

#ifdef TARGETON
#pragma omp target map(uold [0:Hnvar * Hnxt * Hnyt]) map(e [0:Hstep] [0:Hnxyt])                    \
    map(q [0:Hnvar] [0:Hstep] [0:Hnxyt])
#endif
#ifdef LOOPFORM
#pragma omp teams loop bind(teams) private(s, i) collapse(2)
#else
#pragma omp TEAMSDIS parallel for default(none) shared(q, e, uold)                                 \
    FIRSTP(Hsmallr, slices, Hnx, Hnxt, Hnyt, j, Hnxyt, Hstep) private(s, i) collapse(2)
#endif
    for (s = 0; s < slices; s++) {
        for (i = 0; i < Hnx; i++) {
            real_t eken;
            real_t tmp;
            int idxuID = IHV(i + ExtraLayer, j + s, ID);
            int idxuIU = IHV(i + ExtraLayer, j + s, IU);
            int idxuIV = IHV(i + ExtraLayer, j + s, IV);
            int idxuIP = IHV(i + ExtraLayer, j + s, IP);
            q[ID][s][i] = MAX(uold[idxuID], Hsmallr);
            q[IU][s][i] = uold[idxuIU] / q[ID][s][i];
            q[IV][s][i] = uold[idxuIV] / q[ID][s][i];
            eken = half * (Square(q[IU][s][i]) + Square(q[IV][s][i]));
            tmp = uold[idxuIP] / q[ID][s][i] - eken;
            q[IP][s][i] = tmp;
            e[s][i] = tmp;
        }
    }
#ifdef TRACKDATA
    fprintf(stderr, "Moving ComputeQEforRow OUT\n");
#endif
    int nops = slices * Hnx;
    FLOPS(5 * nops, 3 * nops, 1 * nops, 0 * nops);
}
