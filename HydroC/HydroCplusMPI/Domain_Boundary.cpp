#ifdef MPI_ON
#include <mpi.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <unistd.h>

using namespace std;

//
#include "Domain.hpp"
#include "EnumDefs.hpp"
#include "cclock.hpp"

void Domain::boundary_init() {
    int32_t size, ivar, i, j, i0, j0;
    double start, end, startio, elaps;
    start = Custom_Timer::dcclock();
    int sign;
#ifdef MPI_ON
    MPI_Request requests[4];
    MPI_Status status[4];
    MPI_Datatype mpiFormat = MPI_DOUBLE;
    int err = 0, reqcnt = 0;
    int64_t bytesMoved = 0;
#endif

#ifdef MPI_ON
    if (sizeof(real_t) == sizeof(float))
        mpiFormat = MPI_FLOAT;
#endif

    if (m_scan == X_SCAN) {
#ifdef MPI_ON
        if (m_nProc > 1) {
            size = pack_arrayv(m_ExtraLayer, m_sendbufld);
            size = pack_arrayv(m_nx, m_sendbufru);

            startio = dcclock();

            if (m_box[RIGHT_D] != -1) {
                MPI_Isend(m_sendbufru, size, mpiFormat, m_box[RIGHT_D], 123, MPI_COMM_WORLD,
                          &requests[reqcnt]);
                reqcnt++;
                bytesMoved += size;
            }
            if (m_box[LEFT_D] != -1) {
                MPI_Isend(m_sendbufld, size, mpiFormat, m_box[LEFT_D], 246, MPI_COMM_WORLD,
                          &requests[reqcnt]);
                reqcnt++;
                bytesMoved += size;
            }
            if (m_box[RIGHT_D] != -1) {
                MPI_Irecv(m_recvbufru, size, mpiFormat, m_box[RIGHT_D], 246, MPI_COMM_WORLD,
                          &requests[reqcnt]);
                reqcnt++;
                bytesMoved += size;
            }
            if (m_box[LEFT_D] != -1) {
                MPI_Irecv(m_recvbufld, size, mpiFormat, m_box[LEFT_D], 123, MPI_COMM_WORLD,
                          &requests[reqcnt]);
                reqcnt++;
                bytesMoved += size;
            }
            err = MPI_Waitall(reqcnt, requests, status);
            assert(err == MPI_SUCCESS);

            elaps = dcclock() - startio;
            double bandwith = bytesMoved / (1024.0 * 1024.0) / elaps;
            m_mainTimer.set(BOUNDINITBW, bandwith);
            // if (m_myPe == 0) cerr << "X " << bandwith << " ";
        }
#endif
    } // X_SCAN

    if (m_scan == Y_SCAN) {
#ifdef MPI_ON
        if (m_nProc > 1) {
            size = pack_arrayh(m_ExtraLayer, m_sendbufld);
            size = pack_arrayh(m_ny, m_sendbufru);

            startio = dcclock();

            if (m_box[DOWN_D] != -1) {
                MPI_Isend(m_sendbufld, size, mpiFormat, m_box[DOWN_D], 123, MPI_COMM_WORLD,
                          &requests[reqcnt]);
                reqcnt++;
                bytesMoved += size;
            }
            if (m_box[UP_D] != -1) {
                MPI_Isend(m_sendbufru, size, mpiFormat, m_box[UP_D], 246, MPI_COMM_WORLD,
                          &requests[reqcnt]);
                reqcnt++;
                bytesMoved += size;
            }
            if (m_box[DOWN_D] != -1) {
                MPI_Irecv(m_recvbufld, size, mpiFormat, m_box[DOWN_D], 246, MPI_COMM_WORLD,
                          &requests[reqcnt]);
                reqcnt++;
                bytesMoved += size;
            }
            if (m_box[UP_D] != -1) {
                MPI_Irecv(m_recvbufru, size, mpiFormat, m_box[UP_D], 123, MPI_COMM_WORLD,
                          &requests[reqcnt]);
                reqcnt++;
                bytesMoved += size;
            }
            err = MPI_Waitall(reqcnt, requests, status);
            assert(err == MPI_SUCCESS);

            elaps = dcclock() - startio;
            double bandwith = bytesMoved / (1024.0 * 1024.0) / elaps;
            m_mainTimer.set(BOUNDINITBW, bandwith);
            // if (m_myPe == 0) cerr << "Y " << bandwith << " ";
        }
#endif
    } // Y_SCAN
    elaps = Custom_Timer::dcclock() - start;
    m_mainTimer.add(BOUNDINIT, elaps);

} // boundary_init

void Domain::boundary_process() {
    int32_t xmin, xmax, ymin, ymax;
    int32_t size, ivar, i, j, i0, j0;
    int sign;
    double start, end;
    start = Custom_Timer::dcclock();
#ifdef MPI_ON
    MPI_Request requests[4];
    MPI_Status status[4];
    MPI_Datatype mpiFormat = MPI_DOUBLE;
#endif

    getExtends(TILE_FULL, xmin, xmax, ymin, ymax);

#ifdef MPI_ON
    if (sizeof(real_t) == sizeof(float))
        mpiFormat = MPI_FLOAT;
#endif

    if (m_scan == X_SCAN) {
#ifdef MPI_ON
        if (m_box[RIGHT_D] != -1) {
            size = unpack_arrayv(m_nx + m_ExtraLayer, m_recvbufru);
        }
        if (m_box[LEFT_D] != -1) {
            size = unpack_arrayv(0, m_recvbufld);
        }
#endif

<<<<<<< Upstream, based on origin/master
	if (m_boundary_left > 0) {
	    // Left boundary
#pragma omp parallel for \
    default(none),\
    firstprivate(ymin, ymax),\
    private(ivar, i, j, i0, sign) shared(m_uold) collapse(2)
	    for (ivar = 0; ivar < NB_VAR; ivar++) {
		for (i = 0; i < m_ExtraLayer; i++) {
		    Matrix2 < real_t > &uold = *(*m_uold) (ivar);
		    sign = 1.0;
		    if (m_boundary_left == 1) {
			i0 = 2 * m_ExtraLayer - i - 1;	// CL reflexion
			if (ivar == IU_VAR) {
			    sign = -1;
			}
		    } else if (m_boundary_left == 2) {
			i0 = m_ExtraLayer;	// CL absorbante
		    } else {
			i0 = m_nx + i;	// CL periodique
		    }
		    for (j = ymin + m_ExtraLayer; j < ymax - m_ExtraLayer; j++) {
			uold(i, j) = uold(i0, j) * sign;
		    }
		}
	    }
	}
=======
        if (m_boundary_left > 0) {
            // Left boundary
            for (ivar = 0; ivar < NB_VAR; ivar++) {
                Matrix2<real_t> &uold = *(*m_uold)(ivar);
                for (i = 0; i < m_ExtraLayer; i++) {
                    sign = 1.0;
                    if (m_boundary_left == 1) {
                        i0 = 2 * m_ExtraLayer - i - 1; // CL reflexion
                        if (ivar == IU_VAR) {
                            sign = -1;
                        }
                    } else if (m_boundary_left == 2) {
                        i0 = m_ExtraLayer; // CL absorbante
                    } else {
                        i0 = m_nx + i; // CL periodique
                    }
#pragma ivdep
                    for (j = ymin + m_ExtraLayer; j < ymax - m_ExtraLayer; j++) {
                        uold(i, j) = uold(i0, j) * sign;
                    }
                }
            }
        }
>>>>>>> 1ae5822 New indent format and change of utility to clang-format instead of indent. Better for C++

<<<<<<< Upstream, based on origin/master
	if (m_boundary_right > 0) {
	    // Right boundary
#pragma omp parallel for \
    default(none) \
    firstprivate(ymin, ymax),					\
    private(ivar, i, j, i0, sign) shared(m_uold) collapse(2)
	    for (ivar = 0; ivar < NB_VAR; ivar++) {
		for (i = m_nx + m_ExtraLayer; i < m_nx + 2 * m_ExtraLayer; i++) {
		    Matrix2 < real_t > &uold = *(*m_uold) (ivar);
		    sign = 1.0;
		    if (m_boundary_right == 1) {
			i0 = 2 * m_nx + 2 * m_ExtraLayer - i - 1;
			if (ivar == IU_VAR) {
			    sign = -1;
			}
		    } else if (m_boundary_right == 2) {
			i0 = m_nx + m_ExtraLayer;
		    } else {
			i0 = i - m_nx;
		    }
		    for (j = ymin + m_ExtraLayer; j < ymax - m_ExtraLayer; j++) {
			uold(i, j) = uold(i0, j) * sign;
		    }
		}
	    }
	}
    }				// X_SCAN
=======
        if (m_boundary_right > 0) {
            // Right boundary
            for (ivar = 0; ivar < NB_VAR; ivar++) {
                Matrix2<real_t> &uold = *(*m_uold)(ivar);
                for (i = m_nx + m_ExtraLayer; i < m_nx + 2 * m_ExtraLayer; i++) {
                    sign = 1.0;
                    if (m_boundary_right == 1) {
                        i0 = 2 * m_nx + 2 * m_ExtraLayer - i - 1;
                        if (ivar == IU_VAR) {
                            sign = -1;
                        }
                    } else if (m_boundary_right == 2) {
                        i0 = m_nx + m_ExtraLayer;
                    } else {
                        i0 = i - m_nx;
                    }
#pragma ivdep
                    for (j = ymin + m_ExtraLayer; j < ymax - m_ExtraLayer; j++) {
                        uold(i, j) = uold(i0, j) * sign;
                    }
                }
            }
        }
    } // X_SCAN
>>>>>>> 1ae5822 New indent format and change of utility to clang-format instead of indent. Better for C++

    if (m_scan == Y_SCAN) {
#ifdef MPI_ON
        if (m_box[DOWN_D] != -1) {
            unpack_arrayh(0, m_recvbufld);
        }
        if (m_box[UP_D] != -1) {
            unpack_arrayh(m_ny + m_ExtraLayer, m_recvbufru);
        }
#endif
<<<<<<< Upstream, based on origin/master
	// Lower boundary
	if (m_boundary_down > 0) {
	    j0 = 0;
#pragma omp parallel for \
    default(none) \
    firstprivate(xmin, xmax),					\
    private(ivar, i, j, j0, sign) shared(m_uold) collapse(2)
	    for (ivar = 0; ivar < NB_VAR; ivar++) {
		for (j = 0; j < m_ExtraLayer; j++) {
		    Matrix2 < real_t > &uold = *(*m_uold) (ivar);
		    sign = 1;
		    if (m_boundary_down == 1) {
			j0 = 2 * m_ExtraLayer - j - 1;
			if (ivar == IV_VAR) {
			    sign = -1;
			}
		    } else if (m_boundary_down == 2) {
			j0 = m_ExtraLayer;
		    } else {
			j0 = m_ny + j;
		    }
		    for (i = xmin + m_ExtraLayer; i < xmax - m_ExtraLayer; i++) {
			uold(i, j) = uold(i, j0) * sign;
		    }
		}
	    }
	}
	// Upper boundary
	if (m_boundary_up > 0) {
#pragma omp parallel for \
    default(none)						\
    firstprivate(xmin, xmax),					\
    private(ivar, i, j, j0, sign) shared(m_uold) collapse(2)
	    for (ivar = 0; ivar < NB_VAR; ivar++) {
		for (j = m_ny + m_ExtraLayer; j < m_ny + 2 * m_ExtraLayer; j++) {
		    Matrix2 < real_t > &uold = *(*m_uold) (ivar);
		    sign = 1;
		    if (m_boundary_up == 1) {
			j0 = 2 * m_ny + 2 * m_ExtraLayer - j - 1;
			if (ivar == IV_VAR) {
			    sign = -1;
			}
		    } else if (m_boundary_up == 2) {
			j0 = m_ny + 1;
		    } else {
			j0 = j - m_ny;
		    }
		    for (i = xmin + m_ExtraLayer; i < xmax - m_ExtraLayer; i++) {
			uold(i, j) = uold(i, j0) * sign;
		    }
		}
	    }
	}
    }				// Y_SCAN
    Matrix2 < real_t > &uold = *(*m_uold) (IP_VAR);
=======
        // Lower boundary
        if (m_boundary_down > 0) {
            j0 = 0;
            for (ivar = 0; ivar < NB_VAR; ivar++) {
                Matrix2<real_t> &uold = *(*m_uold)(ivar);
                for (j = 0; j < m_ExtraLayer; j++) {
                    sign = 1;
                    if (m_boundary_down == 1) {
                        j0 = 2 * m_ExtraLayer - j - 1;
                        if (ivar == IV_VAR) {
                            sign = -1;
                        }
                    } else if (m_boundary_down == 2) {
                        j0 = m_ExtraLayer;
                    } else {
                        j0 = m_ny + j;
                    }
#pragma ivdep
                    for (i = xmin + m_ExtraLayer; i < xmax - m_ExtraLayer; i++) {
                        uold(i, j) = uold(i, j0) * sign;
                    }
                }
            }
        }
        // Upper boundary
        if (m_boundary_up > 0) {
            for (ivar = 0; ivar < NB_VAR; ivar++) {
                Matrix2<real_t> &uold = *(*m_uold)(ivar);
                for (j = m_ny + m_ExtraLayer; j < m_ny + 2 * m_ExtraLayer; j++) {
                    sign = 1;
                    if (m_boundary_up == 1) {
                        j0 = 2 * m_ny + 2 * m_ExtraLayer - j - 1;
                        if (ivar == IV_VAR) {
                            sign = -1;
                        }
                    } else if (m_boundary_up == 2) {
                        j0 = m_ny + 1;
                    } else {
                        j0 = j - m_ny;
                    }
#pragma ivdep
                    for (i = xmin + m_ExtraLayer; i < xmax - m_ExtraLayer; i++) {
                        uold(i, j) = uold(i, j0) * sign;
                    }
                }
            }
        }
    } // Y_SCAN
    Matrix2<real_t> &uold = *(*m_uold)(IP_VAR);
>>>>>>> 1ae5822 New indent format and change of utility to clang-format instead of indent. Better for C++
    if (m_prt)
        std::cout << "uold boundary_process" << uold;
    double elaps = Custom_Timer::dcclock() - start;
    m_mainTimer.add(BOUNDEXEC, elaps);
} // boundary_process

int32_t Domain::pack_arrayv(int32_t xoffset, Preal_t buffer) {
    int32_t xmin, xmax, ymin, ymax;
    int32_t ivar, i, j;
    getExtends(TILE_FULL, xmin, xmax, ymin, ymax);

#pragma omp parallel for default(none) private(ivar, i, j) \
    firstprivate(ymin, ymax, xoffset, m_ExtraLayer), shared(buffer, m_uold) collapse(3)
    for (ivar = 0; ivar < NB_VAR; ivar++) {
<<<<<<< Upstream, based on origin/master
	for (j = ymin; j < ymax; j++) {
	    for (i = xoffset; i < xoffset + m_ExtraLayer; i++) {
		Matrix2 < real_t > &uold = *(*m_uold) (ivar);
		int32_t p1 = (i - xoffset) + (j - ymin) * (m_ExtraLayer) + ivar * (m_ExtraLayer * (ymax - ymin));
		buffer[p1] = uold(i, j);
	    }
	}
=======
        Matrix2<real_t> &uold = *(*m_uold)(ivar);
        for (j = ymin; j < ymax; j++) {
            for (i = xoffset; i < xoffset + m_ExtraLayer; i++) {
                v = uold(i, j);
                buffer[p++] = v;
            }
        }
>>>>>>> 1ae5822 New indent format and change of utility to clang-format instead of indent. Better for C++
    }
    return NB_VAR * (m_ExtraLayer * (ymax - ymin));
}

int32_t Domain::unpack_arrayv(int32_t xoffset, Preal_t buffer) {
    int32_t xmin, xmax, ymin, ymax;
    int32_t ivar, i, j;
    getExtends(TILE_FULL, xmin, xmax, ymin, ymax);

#pragma omp parallel for default(none) private(ivar, i, j) \
    firstprivate(ymin, ymax, xoffset, m_ExtraLayer), shared(buffer, m_uold) collapse(3)
    for (ivar = 0; ivar < NB_VAR; ivar++) {
<<<<<<< Upstream, based on origin/master
	for (j = ymin; j < ymax; j++) {
	    for (i = xoffset; i < xoffset + m_ExtraLayer; i++) {
		Matrix2 < real_t > &uold = *(*m_uold) (ivar);
		int32_t p1 = (i - xoffset) + (j - ymin) * (m_ExtraLayer) + ivar * (m_ExtraLayer * (ymax - ymin));
		uold(i, j) = buffer[p1];
	    }
	}
=======
        Matrix2<real_t> &uold = *(*m_uold)(ivar);
        for (j = ymin; j < ymax; j++) {
            for (i = xoffset; i < xoffset + m_ExtraLayer; i++) {
                uold(i, j) = buffer[p++];
            }
        }
>>>>>>> 1ae5822 New indent format and change of utility to clang-format instead of indent. Better for C++
    }
    return NB_VAR * (m_ExtraLayer * (ymax - ymin));
}

int32_t Domain::pack_arrayh(int32_t yoffset, Preal_t buffer) {
    int32_t xmin, xmax, ymin, ymax;
    int32_t ivar, i, j;
    getExtends(TILE_FULL, xmin, xmax, ymin, ymax);

#pragma omp parallel for default(none) private(ivar, i, j) \
    firstprivate(xmin, xmax, yoffset, m_ExtraLayer), shared(buffer, m_uold) collapse(3)
    for (ivar = 0; ivar < NB_VAR; ivar++) {
<<<<<<< Upstream, based on origin/master
	for (j = yoffset; j < yoffset + m_ExtraLayer; j++) {
	    for (i = xmin; i < xmax; i++) {
		Matrix2 < real_t > &uold = *(*m_uold) (ivar);
		int32_t p1 = (i - xmin) + (j - yoffset) * (xmax - xmin) + ivar * m_ExtraLayer * (xmax - xmin);
		buffer[p1] = uold(i, j);
	    }
	}
=======
        Matrix2<real_t> &uold = *(*m_uold)(ivar);
        for (j = yoffset; j < yoffset + m_ExtraLayer; j++) {
            for (i = xmin; i < xmax; i++) {
                buffer[p++] = uold(i, j);
            }
        }
>>>>>>> 1ae5822 New indent format and change of utility to clang-format instead of indent. Better for C++
    }
    return (xmax - xmin) * m_ExtraLayer * NB_VAR;
}

int32_t Domain::unpack_arrayh(int32_t yoffset, Preal_t buffer) {
    int32_t xmin, xmax, ymin, ymax;
    int32_t ivar, i, j;
    getExtends(TILE_FULL, xmin, xmax, ymin, ymax);

#pragma omp parallel for default(none) private(ivar, i, j) \
    firstprivate(xmin, xmax, yoffset, m_ExtraLayer), shared(buffer, m_uold) collapse(3)
    for (ivar = 0; ivar < NB_VAR; ivar++) {
<<<<<<< Upstream, based on origin/master
	for (j = yoffset; j < yoffset + m_ExtraLayer; j++) {
	    for (i = xmin; i < xmax; i++) {
		Matrix2 < real_t > &uold = *(*m_uold) (ivar);
		int32_t p1 = (i - xmin) + (j - yoffset) * (xmax - xmin) + ivar * m_ExtraLayer * (xmax - xmin);
		uold(i, j) = buffer[p1];
	    }
	}
=======
        Matrix2<real_t> &uold = *(*m_uold)(ivar);
        for (j = yoffset; j < yoffset + m_ExtraLayer; j++) {
            for (i = xmin; i < xmax; i++) {
                uold(i, j) = buffer[p++];
            }
        }
>>>>>>> 1ae5822 New indent format and change of utility to clang-format instead of indent. Better for C++
    }
    return (xmax - xmin) * m_ExtraLayer * NB_VAR;
}
