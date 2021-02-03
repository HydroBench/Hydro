//
// (C) Guillaume.Colin-de-Verdiere at CEA.Fr
//
#ifdef MPI
#include <mpi.h>
#endif
#include <stdio.h>
#include <string.h>
#include <strings.h>
#include <unistd.h>
#include <stdlib.h>
#include <malloc.h>
#include <assert.h>
#include <limits.h>
#include <sys/time.h>
#include <float.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif

//
#include "mpibounds.h"

static int
pack_arrayv(const int xmin, const hydroparam_t H, hydrovar_t * Hv,
	    real_t * buffer);
static int unpack_arrayv(const int xmin, const hydroparam_t H, hydrovar_t * Hv,
			 real_t * buffer);
static int pack_arrayh(const int xmin, const hydroparam_t H, hydrovar_t * Hv,
		       real_t * buffer);
static int unpack_arrayh(const int xmin, const hydroparam_t H, hydrovar_t * Hv,
			 real_t * buffer);

int
pack_arrayv(const int xmin, const hydroparam_t H, hydrovar_t * Hv,
	    real_t * buffer)
{
    int ivar, i, j, p = 0;
    for (ivar = 0; ivar < H.nvar; ivar++) {
	for (j = 0; j < H.nyt; j++) {
	    // #warning "GATHER to vectorize ?"
	    for (i = xmin; i < xmin + ExtraLayer; i++) {
		buffer[p++] = Hv->uold[IHv(i, j, ivar)];
	    }
	}
    }
    return p;
}

int
unpack_arrayv(const int xmin, const hydroparam_t H, hydrovar_t * Hv,
	      real_t * buffer)
{
    int ivar, i, j, p = 0;
    for (ivar = 0; ivar < H.nvar; ivar++) {
	for (j = 0; j < H.nyt; j++) {
	    // #warning "SCATTER to vectorize ?"
	    for (i = xmin; i < xmin + ExtraLayer; i++) {
		Hv->uold[IHv(i, j, ivar)] = buffer[p++];
	    }
	}
    }
    return p;
}

int
pack_arrayh(const int ymin, const hydroparam_t H, hydrovar_t * Hv,
	    real_t * buffer)
{
    int ivar, i, j, p = 0;
    for (ivar = 0; ivar < H.nvar; ivar++) {
	for (j = ymin; j < ymin + ExtraLayer; j++) {
	    // #warning "GATHER to vectorize ?"
	    // #pragma simd
	    for (i = 0; i < H.nxt; i++) {
		buffer[p++] = Hv->uold[IHv(i, j, ivar)];
	    }
	}
    }
    return p;
}

int
unpack_arrayh(const int ymin, const hydroparam_t H, hydrovar_t * Hv,
	      real_t * buffer)
{
    int ivar, i, j, p = 0;
    for (ivar = 0; ivar < H.nvar; ivar++) {
	for (j = ymin; j < ymin + ExtraLayer; j++) {
	    // #warning "SCATTER to vectorize ?"
	    for (i = 0; i < H.nxt; i++) {
		Hv->uold[IHv(i, j, ivar)] = buffer[p++];
	    }
	}
    }
    return p;
}

#define VALPERLINE 11
int
print_bufferh(FILE * fic, const int ymin, const hydroparam_t H, hydrovar_t * Hv,
	      real_t * buffer)
{
    int ivar, i, j, p = 0, nbr = 1;
    for (ivar = 3; ivar < H.nvar; ivar++) {
	fprintf(fic, "BufferH v=%d\n", ivar);
	for (j = ymin; j < ymin + ExtraLayer; j++) {
	    for (i = 0; i < H.nxt; i++) {
		fprintf(fic, "%13.6le ", buffer[p++]);
		nbr++;
		if (nbr == VALPERLINE) {
		    fprintf(fic, "\n");
		    nbr = 1;
		}
	    }
	}
	if (nbr != 1)
	    fprintf(fic, "\n");
    }
    return p;
}

void mpileftright(int idim, const hydroparam_t H, hydrovar_t * Hv,
		  real_t * sendbufru, real_t * sendbufld, real_t * recvbufru,
		  real_t * recvbufld)
{
    int i, ivar, i0, j, j0, err, size;
    real_t sign;
    int reqcnt = 0;

#ifdef MPI
    MPI_Request requests[4];
    MPI_Status status[4];
    MPI_Datatype mpiFormat = MPI_DOUBLE;

    if (sizeof(real_t) == sizeof(float))
	mpiFormat = MPI_FLOAT;

    i = ExtraLayer;
    size = pack_arrayv(i, H, Hv, sendbufld);
    i = H.nx;
    size = pack_arrayv(i, H, Hv, sendbufru);
    if (H.box[RIGHT_BOX] != -1) {
	MPI_Isend(sendbufru, size, mpiFormat, H.box[RIGHT_BOX], 123,
		  MPI_COMM_WORLD, &requests[reqcnt]);
	reqcnt++;
    }
    if (H.box[LEFT_BOX] != -1) {
	MPI_Isend(sendbufld, size, mpiFormat, H.box[LEFT_BOX], 246,
		  MPI_COMM_WORLD, &requests[reqcnt]);
	reqcnt++;
    }
    if (H.box[RIGHT_BOX] != -1) {
	MPI_Irecv(recvbufru, size, mpiFormat, H.box[RIGHT_BOX], 246,
		  MPI_COMM_WORLD, &requests[reqcnt]);
	reqcnt++;
    }
    if (H.box[LEFT_BOX] != -1) {
	MPI_Irecv(recvbufld, size, mpiFormat, H.box[LEFT_BOX], 123,
		  MPI_COMM_WORLD, &requests[reqcnt]);
	reqcnt++;
    }
    err = MPI_Waitall(reqcnt, requests, status);
    assert(err == MPI_SUCCESS);

    if (H.box[RIGHT_BOX] != -1) {
	i = H.nx + ExtraLayer;
	size = unpack_arrayv(i, H, Hv, recvbufru);
    }

    if (H.box[LEFT_BOX] != -1) {
	i = 0;
	size = unpack_arrayv(i, H, Hv, recvbufld);
    }
#endif
}

void mpiupdown(int idim, const hydroparam_t H, hydrovar_t * Hv,
	       real_t * sendbufru, real_t * sendbufld, real_t * recvbufru,
	       real_t * recvbufld)
{
    int i, ivar, i0, j, j0, err, size;
    real_t sign;
    int reqcnt = 0;

#ifdef MPI
    MPI_Request requests[4];
    MPI_Status status[4];
    MPI_Datatype mpiFormat = MPI_DOUBLE;

    if (sizeof(real_t) == sizeof(float))
	mpiFormat = MPI_FLOAT;

    j = ExtraLayer;
    size = pack_arrayh(j, H, Hv, sendbufld);
    // fprintf(stderr, "%d prep %d\n", H.mype, j);
    j = H.ny;
    size = pack_arrayh(j, H, Hv, sendbufru);
    // fprintf(stderr, "%d prep %d (s=%d)\n", H.mype, j, size);
    if (H.box[DOWN_BOX] != -1) {
	MPI_Isend(sendbufld, size, mpiFormat, H.box[DOWN_BOX], 123,
		  MPI_COMM_WORLD, &requests[reqcnt]);
	reqcnt++;
    }
    if (H.box[UP_BOX] != -1) {
	MPI_Isend(sendbufru, size, mpiFormat, H.box[UP_BOX], 246,
		  MPI_COMM_WORLD, &requests[reqcnt]);
	reqcnt++;
    }
    if (H.box[DOWN_BOX] != -1) {
	MPI_Irecv(recvbufld, size, mpiFormat, H.box[DOWN_BOX], 246,
		  MPI_COMM_WORLD, &requests[reqcnt]);
	reqcnt++;
    }
    if (H.box[UP_BOX] != -1) {
	MPI_Irecv(recvbufru, size, mpiFormat, H.box[UP_BOX], 123,
		  MPI_COMM_WORLD, &requests[reqcnt]);
	reqcnt++;
    }
    err = MPI_Waitall(reqcnt, requests, status);
    assert(err == MPI_SUCCESS);

    if (H.box[DOWN_BOX] != -1) {
	j = 0;
	unpack_arrayh(j, H, Hv, recvbufld);
    }
    if (H.box[UP_BOX] != -1) {
	j = H.ny + ExtraLayer;
	unpack_arrayh(j, H, Hv, recvbufru);
    }
#endif
}

//EOF