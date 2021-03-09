
#include "Domain.hpp"
#include "ParallelInfo.hpp"

#ifdef MPI_ON
#include <mpi.h>
#endif


#include <cmath>

void Domain::dumpOneArray(FILE *f, Matrix2<real_t> &p) {
    int32_t x, y, xmin, xmax, ymin, ymax, n = 1;
    getExtends(TILE_INTERIOR, xmin, xmax, ymin, ymax);

#ifndef MPI_ON
    y = (ymax - ymin) / 2;
    for (x = xmin, n = 1; x < xmax; x++) {
        fprintf(f, "%4d, %17.10le\n", n++, (double)p(x, ymin + y));
    }
    fprintf(f, "\n");

#else  // MPI_ON
    MPI_Request request[1];
    MPI_Status status[1];
    int err;
    double *gbuf = new double[m_globNx];
    double *buf = new double[m_globNx];
    int32_t itsbox[MAXBOX_D];
    int32_t i, d, nbreq;
    int *nd = new int[m_nProc];
    assert(nd != NULL);

    y = m_globNy / 2;
    for (i = 0; i < m_globNx; i++)
        gbuf[i] = 0.0;

    CalcSubSurface(0, m_globNx, 0, m_globNy, 0, m_nProc - 1, itsbox, m_myPe);
    n = 0;
    if ((itsbox[YMIN_D] <= y) && (y < itsbox[YMAX_D])) {
        n = 0;
        for (x = xmin; x < xmax; x++) {
            buf[n++] = (double)p(x, ymin + (y - itsbox[YMIN_D]));
        }
        if (m_myPe == 0) {
            // no transfer needed
            // fprintf(stderr, "n=%d\n", n);
            for (i = 0; i < n; i++) {
                gbuf[itsbox[XMIN_D] + i] = buf[i];
            }
        }
    }

    nd[m_myPe] = n;
    // send/recv the number of double to transfer
    nbreq = 0;
    for (d = 1; d < m_nProc; d++) {
        CalcSubSurface(0, m_globNx, 0, m_globNy, 0, m_nProc - 1, itsbox, d);
        if ((itsbox[YMIN_D] <= y) && (y < itsbox[YMAX_D])) {
            if (m_myPe == 0) {
                MPI_Recv(&nd[d], 1, MPI_INT, d, 887 + 2 * d, MPI_COMM_WORLD,
                         status); //, &request[0]);
            } else {
                if (m_myPe == d) {
                    MPI_Send(&nd[d], 1, MPI_INT, 0, 887 + 2 * d, MPI_COMM_WORLD); // , &request[0]);
                }
            }
        }
    }
    // send/recv the data
    nbreq = 0;
    for (d = 1; d < m_nProc; d++) {
        CalcSubSurface(0, m_globNx, 0, m_globNy, 0, m_nProc - 1, itsbox, d);
        if ((itsbox[YMIN_D] <= y) && (y < itsbox[YMAX_D])) {
            if (m_myPe == 0) {
                MPI_Recv(&gbuf[itsbox[XMIN_D]], nd[d], MPI_DOUBLE, d, 887 + 2 * d + 1,
                         MPI_COMM_WORLD, status);
            } else {
                if (m_myPe == d) {
                    MPI_Send(buf, nd[d], MPI_DOUBLE, 0, 887 + 2 * d + 1, MPI_COMM_WORLD);
                }
            }
        } // if y is inside
    }

    // final trace output
    if (m_myPe == 0) {
        for (x = 0, n = 1; x < m_globNx; x++) {
            fprintf(f, "%4d, %17.10le\n", n++, gbuf[x]);
        }
        fprintf(f, "\n");
    }

    delete[] nd;
    delete[] buf;
    delete[] gbuf;
#endif // MPI_ON
}

void Domain::dumpLineArray(FILE *f, Matrix2<real_t> &p, char *name, char *ext) {
    char fname[256];
    if (ParallelInfo::mype() == 0) {
        sprintf(fname, "%s%s_%06d_%s.lst", "DUMPLINE", name, m_iter, ext);
        f = fopen(fname, "w");
        fprintf(f, "#  X     %s\n", name);
    }
    dumpOneArray(f, p);
    if (f != stderr)
        fclose(f);
}

void Domain::dumpLine(void) {
    Matrix2<real_t> &puold = *(*m_uold)(IP_VAR);
    Matrix2<real_t> &duold = *(*m_uold)(ID_VAR);
    Matrix2<real_t> &uuold = *(*m_uold)(IU_VAR);
    Matrix2<real_t> &vuold = *(*m_uold)(IV_VAR);
    FILE *f = stderr;
    char ext[256];
    char *pvar;

    if (ParallelInfo::mype() == 0) {
        if (ParallelInfo::nb_procs() > 1) {
            strcpy(ext, "PAR");
        } else {
            strcpy(ext, "SEQ");
        }
        pvar = getenv("HYDROC_DUMPEXT");
        if (pvar != NULL) {
            strncat(ext, pvar, 250);
        }
    }
    dumpLineArray(f, puold, (char *)"P", ext);
    dumpLineArray(f, duold, (char *)"D", ext);
    dumpLineArray(f, uuold, (char *)"U", ext);
    dumpLineArray(f, vuold, (char *)"V", ext);

    Matrix2<real_t> speed(uuold);
    for (int32_t j = 0; j < speed.getH(); j++) {
        for (int32_t i = 0; i < speed.getW(); i++) {
            real_t valU = uuold(i, j);
            real_t valV = vuold(i, j);
            speed(i, j) = sqrt(valU * valU + valV * valV);
        }
    }
    dumpLineArray(f, speed, (char *)"S", ext);
}
