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
    int ivar, i, j;
    long lgr = ExtraLayer * H.nvar * H.nyt;
#ifdef TARGETON
#pragma omp target map(Hv->uold[0:H.nvar *H.nxt * H.nyt]) map(tofrom: buffer[0:lgr])
#pragma omp teams distribute parallel for default(none) private(ivar, i, j) shared(buffer, H, Hv, xmin) collapse(3)
#else
#pragma omp parallel for default(none) private(ivar, i, j) shared(buffer, H, Hv, xmin) collapse(3)
#endif
    for (ivar = 0; ivar < H.nvar; ivar++) {
	for (j = 0; j < H.nyt; j++) {
	    for (i = xmin; i < xmin + ExtraLayer; i++) {
		long p =
		    (i - xmin) + j * (ExtraLayer) + ivar * (ExtraLayer * H.nyt);
		buffer[p] = Hv->uold[IHv(i, j, ivar)];
	    }
	}
    }
    return lgr;
}				// pack_arrayv

int
unpack_arrayv(const int xmin, const hydroparam_t H, hydrovar_t * Hv,
	      real_t * buffer)
{
    int ivar, i, j;
    long lgr = ExtraLayer * H.nvar * H.nyt;

#ifdef TARGETON
#pragma omp target map(Hv->uold[0:H.nvar *H.nxt * H.nyt]) map(tofrom: buffer[0:lgr])
#pragma omp teams distribute parallel for default(none) private(ivar, i, j) shared(buffer, H, Hv, xmin) collapse(3)
#else
#pragma omp parallel for default(none) private(ivar, i, j) shared(buffer, H, Hv, xmin) collapse(3)
#endif
    for (ivar = 0; ivar < H.nvar; ivar++) {
	for (j = 0; j < H.nyt; j++) {
	    for (i = xmin; i < xmin + ExtraLayer; i++) {
		long p =
		    (i - xmin) + j * (ExtraLayer) + ivar * (ExtraLayer * H.nyt);
		Hv->uold[IHv(i, j, ivar)] = buffer[p];
	    }
	}
    }
    return lgr;
}				// unpack_arrayv

int
pack_arrayh(const int ymin, const hydroparam_t H, hydrovar_t * Hv,
	    real_t * buffer)
{
    int ivar, i, j;
    long lgr = ExtraLayer * H.nvar * H.nxt;
#ifdef TARGETON
#pragma omp target map(Hv->uold[0:H.nvar *H.nxt * H.nyt]) map(tofrom: buffer[0:lgr])
#pragma omp teams distribute parallel for default(none) private(ivar, i, j) shared(buffer, H, Hv, ymin) collapse(3)
#else
#pragma omp parallel for default(none) private(ivar, i, j) shared(buffer, H, Hv, ymin) collapse(3)
#endif
    for (ivar = 0; ivar < H.nvar; ivar++) {
	for (j = ymin; j < ymin + ExtraLayer; j++) {
	    for (i = 0; i < H.nxt; i++) {
		long p =
		    (i) + (j - ymin) * (H.nxt) + ivar * (ExtraLayer * H.nxt);
		buffer[p] = Hv->uold[IHv(i, j, ivar)];
	    }
	}
    }
    return lgr;
}

int
unpack_arrayh(const int ymin, const hydroparam_t H, hydrovar_t * Hv,
	      real_t * buffer)
{
    long lgr = ExtraLayer * H.nvar * H.nxt;
    int ivar, i, j;
#ifdef TARGETON
#pragma omp target map(Hv->uold[0:H.nvar *H.nxt * H.nyt]) map(tofrom: buffer[0:lgr])
#pragma omp teams distribute parallel for default(none) private(ivar, i, j) shared(buffer, H, Hv, ymin) collapse(3)
#else
#pragma omp parallel for default(none) private(ivar, i, j) shared(buffer, H, Hv, ymin) collapse(3)
#endif
    for (ivar = 0; ivar < H.nvar; ivar++) {
	for (j = ymin; j < ymin + ExtraLayer; j++) {
	    for (i = 0; i < H.nxt; i++) {
		long p =
		    (i) + (j - ymin) * (H.nxt) + ivar * (ExtraLayer * H.nxt);
		Hv->uold[IHv(i, j, ivar)] = buffer[p];
	    }
	}
    }
    return lgr;
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
#ifdef TARGETON
    // get the buffer from the GPU
#pragma omp target update from (sendbufld [0:size])
#endif
    
    j = H.ny;
    size = pack_arrayh(j, H, Hv, sendbufru);
#ifdef TARGETON
    // get the buffer from the GPU
#pragma omp target update from (sendbufru [0:size])
#endif
    
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
#ifdef TARGETON
    // put the buffer on the GPU
#pragma omp target update to (recvbufld [0:size])
#endif
	j = 0;
	unpack_arrayh(j, H, Hv, recvbufld);
    }
    if (H.box[UP_BOX] != -1) {
#ifdef TARGETON
    // put the buffer on the GPU
#pragma omp target update to (recvbufru [0:size])
#endif
	j = H.ny + ExtraLayer;
	unpack_arrayh(j, H, Hv, recvbufru);
    }
#endif
}

//EOF
