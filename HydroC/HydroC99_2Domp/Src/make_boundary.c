#ifdef MPI
#include <mpi.h>
#endif
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <unistd.h>

#include "cclock.h"
#include "make_boundary.h"
#include "mpibounds.h"
#include "parametres.h"
#include "perfcnt.h"
#include "utils.h"

#undef TARGETONS

void make_boundary(int idim, const hydroparam_t H, hydrovar_t *Hv) {
    int i, ivar, i0, j, j0;
    real_t sign;
#ifdef MPI
    static real_t *sendbufru = 0, *sendbufld = 0; // [ExtraLayerTot * H.nxyt * H.nvar]
    static real_t *recvbufru = 0, *recvbufld = 0;
#endif
    struct timespec start, end;
    real_t * uold = Hv->uold;

    WHERE("make_boundary");
#ifdef MPI
    if (sendbufru == 0) {
        long lgr = ExtraLayer * H.nvar * H.nxyt;
        sendbufru = (real_t *)malloc(sizeof(sendbufru[0]) * lgr);
        sendbufld = (real_t *)malloc(sizeof(sendbufld[0]) * lgr);
        recvbufru = (real_t *)malloc(sizeof(recvbufru[0]) * lgr);
        recvbufld = (real_t *)malloc(sizeof(recvbufld[0]) * lgr);
    }
#endif
    start = cclock();
    {

        if (idim == 1) {
#ifdef MPI
            mpileftright(idim, H, Hv, sendbufru, sendbufld, recvbufru, recvbufld);
#endif
            if (H.boundary_left > 0) {
                // Left boundary
#ifdef TARGETON
#pragma omp target map(uold [0:H.nvar * H.nxt * H.nyt])
#endif
#ifdef LOOPFORM
#pragma omp teams loop bind(teams) private(ivar, i, j, i0, sign) collapse(2)
#else
#pragma omp TEAMSDIS parallel for default(none),                                                   \
    firstprivate(H) private(ivar, i, j, i0, sign) shared(uold) collapse(2)
#endif
                for (ivar = 0; ivar < H.nvar; ivar++) {
                    for (i = 0; i < ExtraLayer; i++) {
                        sign = 1.0;
                        if (H.boundary_left == 1) {
                            i0 = ExtraLayerTot - i - 1;
                            if (ivar == IU) {
                                sign = -1.0;
                            }
                        } else if (H.boundary_left == 2) {
                            i0 = 2;
                        } else {
                            i0 = H.nx + i;
                        }
                        for (j = H.jmin + ExtraLayer; j < H.jmax - ExtraLayer; j++) {
                            uold[IHv(i, j, ivar)] = uold[IHv(i0, j, ivar)] * sign;
                        }
                    }
                }
                {
                    int nops =
                        H.nvar * ExtraLayer * ((H.jmax - ExtraLayer) - (H.jmin + ExtraLayer));
                    FLOPS(1 * nops, 0 * nops, 0 * nops, 0 * nops);
                }
            }

            if (H.boundary_right > 0) {
                // Right boundary
#ifdef TARGETON
#pragma omp target map(uold [0:H.nvar * H.nxt * H.nyt])
#endif
#ifdef LOOPFORM
#pragma omp teams loop bind(teams) private(ivar, i, j, i0, sign) collapse(2)
#else
#pragma omp TEAMSDIS parallel for default(none) firstprivate(H) private(ivar, i, j, i0, sign)      \
    shared(uold) collapse(2)
#endif
                for (ivar = 0; ivar < H.nvar; ivar++) {
                    for (i = H.nx + ExtraLayer; i < H.nx + ExtraLayerTot; i++) {
                        sign = 1.0;
                        if (H.boundary_right == 1) {
                            i0 = 2 * H.nx + ExtraLayerTot - i - 1;
                            if (ivar == IU) {
                                sign = -1.0;
                            }
                        } else if (H.boundary_right == 2) {
                            i0 = H.nx + ExtraLayer;
                        } else {
                            i0 = i - H.nx;
                        }
                        for (j = H.jmin + ExtraLayer; j < H.jmax - ExtraLayer; j++) {
                            uold[IHv(i, j, ivar)] = uold[IHv(i0, j, ivar)] * sign;
                        } // for j
                    }     // for i
                }
                {
                    int nops = H.nvar * ((H.jmax - ExtraLayer) - (H.jmin + ExtraLayer)) *
                               ((H.nx + ExtraLayerTot) - (H.nx + ExtraLayer));
                    FLOPS(1 * nops, 0 * nops, 0 * nops, 0 * nops);
                }
            }
        } else {
#ifdef MPI
            mpiupdown(idim, H, Hv, sendbufru, sendbufld, recvbufru, recvbufld);
#endif
            // Lower boundary
            if (H.boundary_down > 0) {
                j0 = 0;
#ifdef TARGETON
#pragma omp target map(uold [0:H.nvar * H.nxt * H.nyt])
#endif
#ifdef LOOPFORM
#pragma omp teams loop bind(teams) private(ivar, i, j, j0, sign) collapse(2)
#else
#pragma omp TEAMSDIS parallel for default(none) firstprivate(H) private(ivar, i, j, j0, sign)      \
    shared(uold) collapse(2)
#endif
                for (ivar = 0; ivar < H.nvar; ivar++) {
                    for (j = 0; j < ExtraLayer; j++) {
                        sign = 1.0;
                        if (H.boundary_down == 1) {
                            j0 = ExtraLayerTot - j - 1;
                            if (ivar == IV) {
                                sign = -1.0;
                            }
                        } else if (H.boundary_down == 2) {
                            j0 = ExtraLayerTot;
                        } else {
                            j0 = H.ny + j;
                        }
                        for (i = H.imin + ExtraLayer; i < H.imax - ExtraLayer; i++) {
                            uold[IHv(i, j, ivar)] = uold[IHv(i, j0, ivar)] * sign;
                        }
                    }
                }
                {
                    int nops = H.nvar * ((ExtraLayer) - (0)) *
                               ((H.imax - ExtraLayer) - (H.imin + ExtraLayer));
                    FLOPS(1 * nops, 0 * nops, 0 * nops, 0 * nops);
                }
            }
            // Upper boundary
            if (H.boundary_up > 0) {
#ifdef TARGETON
#pragma omp target map(uold [0:H.nvar * H.nxt * H.nyt])
#endif
#ifdef LOOPFORM
#pragma omp teams loop bind(teams) private(ivar, i, j, j0, sign) collapse(2)
#else
#pragma omp TEAMSDIS parallel for default(none) firstprivate(H) private(ivar, i, j, j0, sign)      \
    shared(uold) collapse(2)
#endif
                for (ivar = 0; ivar < H.nvar; ivar++) {
                    for (j = H.ny + ExtraLayer; j < H.ny + ExtraLayerTot; j++) {
                        sign = 1.0;
                        if (H.boundary_up == 1) {
                            j0 = 2 * H.ny + ExtraLayerTot - j - 1;
                            if (ivar == IV) {
                                sign = -1.0;
                            }
                        } else if (H.boundary_up == 2) {
                            j0 = H.ny + 1;
                        } else {
                            j0 = j - H.ny;
                        }
                        for (i = H.imin + ExtraLayer; i < H.imax - ExtraLayer; i++) {
                            uold[IHv(i, j, ivar)] = uold[IHv(i, j0, ivar)] * sign;
                        }
                    }
                }
                {
                    int nops = H.nvar * ((H.ny + ExtraLayerTot) - (H.ny + ExtraLayer)) *
                               ((H.imax - ExtraLayer) - (H.imin + ExtraLayer));
                    FLOPS(1 * nops, 0 * nops, 0 * nops, 0 * nops);
                }
            }
        }
    } // end of the omp target data
    end = cclock();
    functim[TIM_MAKBOU] += ccelaps(start, end);
}

// make_boundary
// EOF
