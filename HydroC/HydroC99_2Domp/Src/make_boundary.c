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
    int32_t Hnvar = H.nvar, Hnxt = H.nxt, Hnyt = H.nyt;
    int32_t Hnx = H.nx, Hny = H.ny, Himin = H.imin, Himax = H.imax, Hjmin = H.jmin, Hjmax = H.jmax;
    int32_t Hboundup = H.boundary_up, Hbounddown = H.boundary_down, Hboundleft = H.boundary_left, Hboundright = H.boundary_right;
#ifdef TRACKDATA
    fprintf(stderr, "Moving make_boundary IN\n");
#endif
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
            if (H.nproc > 1) mpileftright(idim, H, Hv, sendbufru, sendbufld, recvbufru, recvbufld);
#endif
            if (Hboundleft > 0) {
                // Left boundary
#ifdef TARGETON
#pragma omp target map(uold [0:Hnvar * Hnxt * Hnyt])
#endif
#ifdef LOOPFORM
#pragma omp teams loop bind(teams) private(ivar, i, j, i0, sign) collapse(2)
#else
#pragma omp TEAMSDIS parallel for default(none),                                                   \
    firstprivate(Hnvar, Hnxt, Hnyt, Hnx, Hboundleft, Hjmin, Hjmax) private(ivar, i, j, i0, sign) shared(uold) collapse(2)
#endif
                for (ivar = 0; ivar < Hnvar; ivar++) {
                    for (i = 0; i < ExtraLayer; i++) {
                        sign = 1.0;
                        if (Hboundleft == 1) {
                            i0 = ExtraLayerTot - i - 1;
                            if (ivar == IU) {
                                sign = -1.0;
                            }
                        } else if (Hboundleft == 2) {
                            i0 = 2;
                        } else {
                            i0 = Hnx + i;
                        }
                        for (j = Hjmin + ExtraLayer; j < Hjmax - ExtraLayer; j++) {
                            uold[IHV(i, j, ivar)] = uold[IHV(i0, j, ivar)] * sign;
                        }
                    }
                }
                {
                    int nops =
                        Hnvar * ExtraLayer * ((Hjmax - ExtraLayer) - (Hjmin + ExtraLayer));
                    FLOPS(1 * nops, 0 * nops, 0 * nops, 0 * nops);
                }
            }

            if (Hboundright > 0) {
                // Right boundary
#ifdef TARGETON
#pragma omp target map(uold [0:Hnvar * Hnxt * Hnyt])
#endif
#ifdef LOOPFORM
#pragma omp teams loop bind(teams) private(ivar, i, j, i0, sign) collapse(2)
#else
#pragma omp TEAMSDIS parallel for default(none) firstprivate(Hnvar, Hnxt, Hnyt, Hnx, Hboundright, Hjmin, Hjmax) private(ivar, i, j, i0, sign)      \
    shared(uold) collapse(2)
#endif
                for (ivar = 0; ivar < Hnvar; ivar++) {
                    for (i = Hnx + ExtraLayer; i < Hnx + ExtraLayerTot; i++) {
                        sign = 1.0;
                        if (Hboundright == 1) {
                            i0 = 2 * Hnx + ExtraLayerTot - i - 1;
                            if (ivar == IU) {
                                sign = -1.0;
                            }
                        } else if (Hboundright == 2) {
                            i0 = Hnx + ExtraLayer;
                        } else {
                            i0 = i - Hnx;
                        }
                        for (j = Hjmin + ExtraLayer; j < Hjmax - ExtraLayer; j++) {
                            uold[IHV(i, j, ivar)] = uold[IHV(i0, j, ivar)] * sign;
                        } // for j
                    }     // for i
                }
                {
                    int nops = Hnvar * ((Hjmax - ExtraLayer) - (Hjmin + ExtraLayer)) *
                               ((Hnx + ExtraLayerTot) - (Hnx + ExtraLayer));
                    FLOPS(1 * nops, 0 * nops, 0 * nops, 0 * nops);
                }
            }
        } else {
#ifdef MPI
            if (H.nproc > 1) mpiupdown(idim, H, Hv, sendbufru, sendbufld, recvbufru, recvbufld);
#endif
            // Lower boundary
            if (Hbounddown > 0) {
                j0 = 0;
#ifdef TARGETON
#pragma omp target map(uold [0:Hnvar * Hnxt * Hnyt])
#endif
#ifdef LOOPFORM
#pragma omp teams loop bind(teams) private(ivar, i, j, j0, sign) collapse(2)
#else
#pragma omp TEAMSDIS parallel for default(none) firstprivate(Hnvar, Hnxt, Hnyt, Hny, Hbounddown, Himin, Himax) private(ivar, i, j, j0, sign)      \
    shared(uold) collapse(2)
#endif
                for (ivar = 0; ivar < Hnvar; ivar++) {
                    for (j = 0; j < ExtraLayer; j++) {
                        sign = 1.0;
                        if (Hbounddown == 1) {
                            j0 = ExtraLayerTot - j - 1;
                            if (ivar == IV) {
                                sign = -1.0;
                            }
                        } else if (Hbounddown == 2) {
                            j0 = ExtraLayerTot;
                        } else {
                            j0 = Hny + j;
                        }
                        for (i = Himin + ExtraLayer; i < Himax - ExtraLayer; i++) {
                            uold[IHV(i, j, ivar)] = uold[IHV(i, j0, ivar)] * sign;
                        }
                    }
                }
                {
                    int nops = Hnvar * ((ExtraLayer) - (0)) *
                               ((Himax - ExtraLayer) - (Himin + ExtraLayer));
                    FLOPS(1 * nops, 0 * nops, 0 * nops, 0 * nops);
                }
            }
            // Upper boundary
            if (Hboundup > 0) {
#ifdef TARGETON
#pragma omp target map(uold [0:Hnvar * Hnxt * Hnyt])
#endif
#ifdef LOOPFORM
#pragma omp teams loop bind(teams) private(ivar, i, j, j0, sign) collapse(2)
#else
#pragma omp TEAMSDIS parallel for default(none) firstprivate(Hnvar, Hnxt, Hnyt, Hny, Hboundup, Himin, Himax) private(ivar, i, j, j0, sign)      \
    shared(uold) collapse(2)
#endif
                for (ivar = 0; ivar < Hnvar; ivar++) {
                    for (j = Hny + ExtraLayer; j < Hny + ExtraLayerTot; j++) {
                        sign = 1.0;
                        if (Hboundup == 1) {
                            j0 = 2 * Hny + ExtraLayerTot - j - 1;
                            if (ivar == IV) {
                                sign = -1.0;
                            }
                        } else if (Hboundup == 2) {
                            j0 = Hny + 1;
                        } else {
                            j0 = j - Hny;
                        }
                        for (i = Himin + ExtraLayer; i < Himax - ExtraLayer; i++) {
                            uold[IHV(i, j, ivar)] = uold[IHV(i, j0, ivar)] * sign;
                        }
                    }
                }
                {
                    int nops = Hnvar * ((Hny + ExtraLayerTot) - (Hny + ExtraLayer)) *
                               ((Himax - ExtraLayer) - (Himin + ExtraLayer));
                    FLOPS(1 * nops, 0 * nops, 0 * nops, 0 * nops);
                }
            }
        }
    } // end of the omp target data
    end = cclock();
    functim[TIM_MAKBOU] += ccelaps(start, end);
#ifdef TRACKDATA
    fprintf(stderr, "Moving make_boundary OUT\n");
#endif
}

// make_boundary
// EOF
