//
// (C) Guillaume.Colin-de-Verdiere at CEA.Fr
//
//
#include "Domain.hpp"
#include "ParallelInfo.hpp"

#ifdef MPI_ON
#include <mpi.h>
#endif

#include <cmath>

void Domain::createTestCase() {
    int32_t xmin, xmax, ymin, ymax;
    int32_t i, j, x, y;

    int nProc = ParallelInfo::nb_procs();
    int myPe = ParallelInfo::mype();

    getExtends(TILE_FULL, xmin, xmax, ymin, ymax);

    m_uold = new Soa(NB_VAR, (xmax + xmin + 1), (ymax - ymin + 1));

    Matrix2<real_t> &uoldIP = *(*m_uold)(IP_VAR);
    Matrix2<real_t> &uoldID = *(*m_uold)(ID_VAR);
    Matrix2<real_t> &uoldIU = *(*m_uold)(IU_VAR);
    Matrix2<real_t> &uoldIV = *(*m_uold)(IV_VAR);

    uoldIP.clear();
    uoldID.clear();
    uoldIU.clear();
    uoldIV.clear();

    for (j = ymin + m_ExtraLayer; j < ymax - m_ExtraLayer; j++)
        for (i = xmin + m_ExtraLayer; i < xmax - m_ExtraLayer; i++) {
            uoldID(i, j) = one;
        }

    for (j = ymin + m_ExtraLayer; j < ymax - m_ExtraLayer; j++)
        for (i = xmin + m_ExtraLayer; i < xmax - m_ExtraLayer; i++) {
            uoldIU(i, j) = zero;
        }

    for (j = ymin + m_ExtraLayer; j < ymax - m_ExtraLayer; j++)
        for (i = xmin + m_ExtraLayer; i < xmax - m_ExtraLayer; i++) {
            uoldIV(i, j) = zero;
        }

    for (j = ymin + m_ExtraLayer; j < ymax - m_ExtraLayer; j++)
        for (i = xmin + m_ExtraLayer; i < xmax - m_ExtraLayer; i++) {
            uoldIP(i, j) = 1e-5;
        }

    if (m_testcase == 0) {
        if (nProc == 1) {
            x = (xmax - xmin) / 2 + m_ExtraLayer * 0;
            y = (ymax - ymin) / 2 + m_ExtraLayer * 0;
            uoldIP(x, y) = one / m_dx / m_dx;
            if (m_stats > 0)
                printf("Centered test case : %d %d\n", x, y);
        } else {
            x = ((m_globNx) / 2);
            y = ((m_globNy) / 2);
            if ((x >= m_box[XMIN_D]) && (x < m_box[XMAX_D]) && (y >= m_box[YMIN_D]) &&
                (y < m_box[YMAX_D])) {
                x = x - m_box[XMIN_D] + m_ExtraLayer;
                y = y - m_box[YMIN_D] + m_ExtraLayer;
                uoldIP(x, y) = one / m_dx / m_dx;
                if (m_stats > 0)
                    printf("Centered test case : [%d] %d %d\n", myPe, x, y);
            }
        }
    }

    if (m_testcase == 1) {
        x = m_ExtraLayer;
        y = m_ExtraLayer;
        if (nProc == 1) {
            uoldIP(x, y) = one / m_dx / m_dx;
            if (m_stats > 0)
                printf("Lower corner test case : %d %d\n", x, y);
        } else {
            if ((x >= m_box[XMIN_D]) && (x < m_box[XMAX_D]) && (y >= m_box[YMIN_D]) &&
                (y < m_box[YMAX_D])) {
                uoldIP(x, y) = one / m_dx / m_dx;
                if (m_stats > 0)
                    printf("Lower corner test case : [%d] %d %d\n", myPe, x, y);
            }
        }
    }

    if (m_testcase == 2) {
        if (nProc == 1) {
            x = m_ExtraLayer;
            y = m_ExtraLayer;
            for (j = y; j < ymax; j++) {
                uoldIP(x, j) = one / m_dx / m_dx;
            }
            if (m_stats > 0)
                printf("SOD tube test case\n");
        } else {
            x = m_ExtraLayer;
            y = m_ExtraLayer;
            for (j = 0; j < m_globNy; j++) {
                if ((x >= m_box[XMIN_D]) && (x < m_box[XMAX_D]) && (j >= m_box[YMIN_D]) &&
                    (j < m_box[YMAX_D])) {
                    y = j - m_box[YMIN_D] + m_ExtraLayer;
                    uoldIP(x, y) = one / m_dx / m_dx;
                }
            }
            if (m_stats > 0)
                printf("SOD tube test case in //\n");
        }
    }
    if (m_testcase == 3) {
        for (j = 0; j < m_globNy; j++) {
            double osc = double(j + m_globNy * 0.5) / double(m_globNy) * 16. * 3.14159;
            double oscmx = 0.1 * (m_globNx - m_ExtraLayer);
            int xm;
            osc = sin(osc) * oscmx;
            xm = 2 * m_ExtraLayer + osc + oscmx;
            x = xm;
            // for (x = m_ExtraLayer; x < xm; x++) {
            if ((x >= m_box[XMIN_D]) && (x < m_box[XMAX_D]) && (j >= m_box[YMIN_D]) &&
                (j < m_box[YMAX_D])) {
                x = x - m_box[XMIN_D] + m_ExtraLayer;
                y = j - m_box[YMIN_D] + m_ExtraLayer;
                uoldIP(x, y) = one / m_dx / m_dx;
            }
            //}
        }
        printf("\n");
    }
    if (m_testcase > 3) {
        printf("Test case not implemented -- aborting !\n");
        abort();
    }
    fflush(stdout);
}

// EOF
