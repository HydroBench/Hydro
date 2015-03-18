#ifdef MPI_ON
#include <mpi.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

#include <cstring>
#include <cstdlib>
#include <unistd.h>
#include <cmath>
#include <cstdio>
#include <climits>
#include <cerrno>
#include <iostream>
#include <iomanip>

#include <strings.h>
#include <unistd.h>
#include <malloc.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <float.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <stdarg.h>

using namespace std;

//
#include "EnumDefs.hpp"
#include "Domain.hpp"
#include "cclock.h"

void Domain::boundary_init()
{
	int32_t size, ivar, i, j, i0, j0;
	int sign;
#ifdef MPI_ON
	MPI_Request requests[4];
	MPI_Status status[4];
	MPI_Datatype mpiFormat = MPI_DOUBLE;
	int err = 0, reqcnt = 0;
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

			if (m_box[RIGHT_D] != -1) {
				MPI_Isend(m_sendbufru, size, mpiFormat, m_box[RIGHT_D], 123, MPI_COMM_WORLD, &requests[reqcnt]);
				reqcnt++;
			}
			if (m_box[LEFT_D] != -1) {
				MPI_Isend(m_sendbufld, size, mpiFormat, m_box[LEFT_D], 246, MPI_COMM_WORLD, &requests[reqcnt]);
				reqcnt++;
			}
			if (m_box[RIGHT_D] != -1) {
				MPI_Irecv(m_recvbufru, size, mpiFormat, m_box[RIGHT_D], 246, MPI_COMM_WORLD, &requests[reqcnt]);
				reqcnt++;
			}
			if (m_box[LEFT_D] != -1) {
				MPI_Irecv(m_recvbufld, size, mpiFormat, m_box[LEFT_D], 123, MPI_COMM_WORLD, &requests[reqcnt]);
				reqcnt++;
			}
		}
#endif
	}			// X_SCAN

	if (m_scan == Y_SCAN) {
#ifdef MPI_ON
		if (m_nProc > 1) {
			size = pack_arrayh(m_ExtraLayer, m_sendbufld);
			size = pack_arrayh(m_ny, m_sendbufru);

			if (m_box[DOWN_D] != -1) {
				MPI_Isend(m_sendbufld, size, mpiFormat, m_box[DOWN_D], 123, MPI_COMM_WORLD, &requests[reqcnt]);
				reqcnt++;
			}
			if (m_box[UP_D] != -1) {
				MPI_Isend(m_sendbufru, size, mpiFormat, m_box[UP_D], 246, MPI_COMM_WORLD, &requests[reqcnt]);
				reqcnt++;
			}
			if (m_box[DOWN_D] != -1) {
				MPI_Irecv(m_recvbufld, size, mpiFormat, m_box[DOWN_D], 246, MPI_COMM_WORLD, &requests[reqcnt]);
				reqcnt++;
			}
			if (m_box[UP_D] != -1) {
				MPI_Irecv(m_recvbufru, size, mpiFormat, m_box[UP_D], 123, MPI_COMM_WORLD, &requests[reqcnt]);
				reqcnt++;
			}
		}
#endif
	}			// Y_SCAN
#ifdef MPI_ON
	if (m_nProc > 1) {
		err = MPI_Waitall(reqcnt, requests, status);
		assert(err == MPI_SUCCESS);
	}			// Y_SCAN
#endif
}

void Domain::boundary_process()
{
	int32_t xmin, xmax, ymin, ymax;
	int32_t size, ivar, i, j, i0, j0;
	int sign;
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

		if (m_boundary_left > 0) {
			// Left boundary
			for (ivar = 0; ivar < NB_VAR; ivar++) {
				Matrix2 < real_t > &uold = *(*m_uold) (ivar);
				for (i = 0; i < m_ExtraLayer; i++) {
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
#pragma ivdep
					for (j = ymin + m_ExtraLayer; j < ymax - m_ExtraLayer; j++) {
						uold(i, j) = uold(i0, j) * sign;
					}
				}
			}
		}

		if (m_boundary_right > 0) {
			// Right boundary
			for (ivar = 0; ivar < NB_VAR; ivar++) {
				Matrix2 < real_t > &uold = *(*m_uold) (ivar);
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
	}			// X_SCAN

	if (m_scan == Y_SCAN) {
#ifdef MPI_ON
		if (m_box[DOWN_D] != -1) {
			unpack_arrayh(0, m_recvbufld);
		}
		if (m_box[UP_D] != -1) {
			unpack_arrayh(m_ny + m_ExtraLayer, m_recvbufru);
		}
#endif
		// Lower boundary
		if (m_boundary_down > 0) {
			j0 = 0;
			for (ivar = 0; ivar < NB_VAR; ivar++) {
				Matrix2 < real_t > &uold = *(*m_uold) (ivar);
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
				Matrix2 < real_t > &uold = *(*m_uold) (ivar);
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
	}			// Y_SCAN
	Matrix2 < real_t > &uold = *(*m_uold) (IP_VAR);
	if (m_prt)
		uold.printFormatted("uold boundary_process");
}

int32_t Domain::pack_arrayv(int32_t xoffset, real_t * buffer)
{
	int32_t xmin, xmax, ymin, ymax;
	int32_t ivar, i, j, p = 0;
	real_t v;
	getExtends(TILE_FULL, xmin, xmax, ymin, ymax);

	for (ivar = 0; ivar < NB_VAR; ivar++) {
		Matrix2 < real_t > &uold = *(*m_uold) (ivar);
		for (j = ymin; j < ymax; j++) {
			for (i = xoffset; i < xoffset + m_ExtraLayer; i++) {
				v = uold(i, j);
				buffer[p++] = v;
			}
		}
	}
	return p;
}

int32_t Domain::unpack_arrayv(int32_t xoffset, real_t * buffer)
{
	int32_t xmin, xmax, ymin, ymax;
	int32_t ivar, i, j, p = 0;
	getExtends(TILE_FULL, xmin, xmax, ymin, ymax);

	for (ivar = 0; ivar < NB_VAR; ivar++) {
		Matrix2 < real_t > &uold = *(*m_uold) (ivar);
		for (j = ymin; j < ymax; j++) {
			for (i = xoffset; i < xoffset + m_ExtraLayer; i++) {
				uold(i, j) = buffer[p++];
			}
		}
	}
	return p;
}

int32_t Domain::pack_arrayh(int32_t yoffset, real_t * buffer)
{
	int32_t xmin, xmax, ymin, ymax;
	int32_t ivar, i, j, p = 0;
	getExtends(TILE_FULL, xmin, xmax, ymin, ymax);

	for (ivar = 0; ivar < NB_VAR; ivar++) {
		Matrix2 < real_t > &uold = *(*m_uold) (ivar);
		for (j = yoffset; j < yoffset + m_ExtraLayer; j++) {
			for (i = xmin; i < xmax; i++) {
				buffer[p++] = uold(i, j);
			}
		}
	}
	return p;
}

int32_t Domain::unpack_arrayh(int32_t yoffset, real_t * buffer)
{
	int32_t xmin, xmax, ymin, ymax;
	int32_t ivar, i, j, p = 0;
	getExtends(TILE_FULL, xmin, xmax, ymin, ymax);

	for (ivar = 0; ivar < NB_VAR; ivar++) {
		Matrix2 < real_t > &uold = *(*m_uold) (ivar);
		for (j = yoffset; j < yoffset + m_ExtraLayer; j++) {
			for (i = xmin; i < xmax; i++) {
				uold(i, j) = buffer[p++];
			}
		}
	}
	return p;
}
