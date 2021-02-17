//
// (C) Guillaume.Colin-de-Verdiere at CEA.Fr
//
#ifdef MPI
#include <mpi.h>
#endif
#include <assert.h>
#include <float.h>
#include <limits.h>
#include <malloc.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/time.h>
#include <unistd.h>
#ifdef _OPENMP
#include <omp.h>
#endif

//
#include "mpibounds.h"

// static int pack_arrayv(const int xmin, const hydroparam_t H, hydrovar_t *Hv, real_t *buffer);
// static int unpack_arrayv(const int xmin, const hydroparam_t H, hydrovar_t *Hv, real_t *buffer);
// static int pack_arrayh(const int xmin, const hydroparam_t H, hydrovar_t *Hv, real_t *buffer);
// static int unpack_arrayh(const int xmin, const hydroparam_t H, hydrovar_t *Hv, real_t *buffer);

int pack_arrayv(const int xmin, const hydroparam_t H, hydrovar_t *Hv, real_t *buffer) {
    int ivar, i, j;
    long lgr = ExtraLayer * H.nvar * H.nyt;
    real_t *uold = &Hv->uold[0];
    int32_t Hnvar = H.nvar, Hnxt = H.nxt, Hnyt = H.nyt;

#ifdef TRACKDATA
    fprintf(stderr, "Moving pack_arrayv IN %d\n", (int) sizeof(H));
#endif
    
#ifdef TARGETON
#pragma omp target map(uold [0:Hnvar * Hnxt * Hnyt]) map(from: buffer [0:lgr])
#endif
#pragma omp TEAMSDIS parallel for default(none) private(ivar, i, j) firstprivate(xmin, Hnvar, Hnxt, Hnyt), \
    shared(buffer, uold) collapse(3)
    for (ivar = 0; ivar < Hnvar; ivar++) {
        for (j = 0; j < Hnyt; j++) {
            for (i = xmin; i < xmin + ExtraLayer; i++) {
                long p = (i - xmin) + j * (ExtraLayer) + ivar * (ExtraLayer * Hnyt);
                buffer[p] = uold[IHV(i, j, ivar)];
            }
        }
    }
#ifdef TRACKDATA
    fprintf(stderr, "Moving pack_arrayv OUT\n");
#endif
    return lgr;
} // pack_arrayv

int unpack_arrayv(const int xmin, const hydroparam_t H, hydrovar_t *Hv, real_t *buffer) {
    int ivar, i, j;
    long lgr = ExtraLayer * H.nvar * H.nyt;
    real_t *uold = &Hv->uold[0];
    int32_t Hnvar = H.nvar, Hnxt = H.nxt, Hnyt = H.nyt;

#ifdef TRACKDATA
    fprintf(stderr, "Moving unpack_arrayv IN\n");
#endif
#ifdef TARGETON
#pragma omp target map(uold [0:Hnvar * Hnxt * Hnyt]) map(to: buffer [0:lgr])
#endif
#pragma omp TEAMSDIS parallel for default(none) private(ivar, i, j) firstprivate(xmin, Hnvar, Hnxt, Hnyt), \
    shared(buffer, uold) collapse(3)
    for (ivar = 0; ivar < Hnvar; ivar++) {
        for (j = 0; j < Hnyt; j++) {
            for (i = xmin; i < xmin + ExtraLayer; i++) {
                long p = (i - xmin) + j * (ExtraLayer) + ivar * (ExtraLayer * Hnyt);
                uold[IHV(i, j, ivar)] = buffer[p];
            }
        }
    }
#ifdef TRACKDATA
    fprintf(stderr, "Moving unpack_arrayv OUT\n");
#endif
    return lgr;
} // unpack_arrayv

int pack_arrayh(const int ymin, const hydroparam_t H, hydrovar_t *Hv, real_t *buffer) {
    int ivar, i, j;
    long lgr = ExtraLayer * H.nvar * H.nxt;
    real_t *uold = &Hv->uold[0];
    int32_t Hnvar = H.nvar, Hnxt = H.nxt, Hnyt = H.nyt;

#ifdef TRACKDATA
    fprintf(stderr, "Moving pack_arrayh IN\n");
#endif
#ifdef TARGETON
#pragma omp target map(uold [0:Hnvar * Hnxt * Hnyt]) map(from: buffer [0:lgr])
#endif
#pragma omp TEAMSDIS parallel for default(none) private(ivar, i, j) firstprivate(ymin, Hnvar, Hnxt, Hnyt), \
    shared(buffer, uold) collapse(3)
    for (ivar = 0; ivar < Hnvar; ivar++) {
        for (j = ymin; j < ymin + ExtraLayer; j++) {
            for (i = 0; i < Hnxt; i++) {
                long p = (i) + (j - ymin) * (Hnxt) + ivar * (ExtraLayer * Hnxt);
                buffer[p] = uold[IHV(i, j, ivar)];
            }
        }
    }
#ifdef TRACKDATA
    fprintf(stderr, "Moving pack_arrayh OUT\n");
#endif
    return lgr;
}

int unpack_arrayh(const int ymin, const hydroparam_t H, hydrovar_t *Hv, real_t *buffer) {
    long lgr = ExtraLayer * H.nvar * H.nxt;
    int ivar, i, j;
    real_t *uold = &Hv->uold[0];
    int32_t Hnvar = H.nvar, Hnxt = H.nxt, Hnyt = H.nyt;

#ifdef TRACKDATA
    fprintf(stderr, "Moving unpack_arrayh IN\n");
#endif
#ifdef TARGETON
#pragma omp target map(uold [0:Hnvar *Hnxt * Hnyt]) map(to: buffer [0:lgr])
#endif
#pragma omp TEAMSDIS parallel for default(none) private(ivar, i, j) firstprivate(ymin, Hnvar, Hnxt, Hnyt), \
    shared(buffer, uold) collapse(3)
    for (ivar = 0; ivar < Hnvar; ivar++) {
        for (j = ymin; j < ymin + ExtraLayer; j++) {
            for (i = 0; i < Hnxt; i++) {
                long p = (i) + (j - ymin) * (Hnxt) + ivar * (ExtraLayer * Hnxt);
                uold[IHV(i, j, ivar)] = buffer[p];
            }
        }
    }
#ifdef TRACKDATA
    fprintf(stderr, "Moving unpack_arrayh OUT\n");
#endif
    return lgr;
}

#define VALPERLINE 11
int print_bufferh(FILE *fic, const int ymin, const hydroparam_t H, hydrovar_t *Hv, real_t *buffer) {
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

void mpileftright(int idim, const hydroparam_t H, hydrovar_t *Hv, real_t *sendbufru,
                  real_t *sendbufld, real_t *recvbufru, real_t *recvbufld) {
    int i, ivar, i0, j, j0, err, size;
    real_t sign;
    int reqcnt = 0;

#ifdef MPI
    MPI_Request requests[4];
    MPI_Status status[4];
    MPI_Datatype mpiFormat = MPI_DOUBLE;

#ifdef TRACKDATA
    fprintf(stderr, "Moving mpileftright IN\n");
#endif
    if (sizeof(real_t) == sizeof(float))
        mpiFormat = MPI_FLOAT;

    i = ExtraLayer;
    size = pack_arrayv(i, H, Hv, sendbufld);
#ifdef TARGETON
    // get the buffer from the GPU
#ifdef TRACKDATA
    fprintf(stderr, "Moving mpileftright update from\n");
#endif
#pragma omp target update from(sendbufld [0:size])
#endif
    i = H.nx;
    size = pack_arrayv(i, H, Hv, sendbufru);
#ifdef TARGETON
    // get the buffer from the GPU
#ifdef TRACKDATA
    fprintf(stderr, "Moving mpileftright update from\n");
#endif
#pragma omp target update from(sendbufru [0:size])
#endif
    if (H.box[RIGHT_BOX] != -1) {
        MPI_Isend(sendbufru, size, mpiFormat, H.box[RIGHT_BOX], 123, MPI_COMM_WORLD,
                  &requests[reqcnt]);
        reqcnt++;
    }
    if (H.box[LEFT_BOX] != -1) {
        MPI_Isend(sendbufld, size, mpiFormat, H.box[LEFT_BOX], 246, MPI_COMM_WORLD,
                  &requests[reqcnt]);
        reqcnt++;
    }
    if (H.box[RIGHT_BOX] != -1) {
        MPI_Irecv(recvbufru, size, mpiFormat, H.box[RIGHT_BOX], 246, MPI_COMM_WORLD,
                  &requests[reqcnt]);
        reqcnt++;
    }
    if (H.box[LEFT_BOX] != -1) {
        MPI_Irecv(recvbufld, size, mpiFormat, H.box[LEFT_BOX], 123, MPI_COMM_WORLD,
                  &requests[reqcnt]);
        reqcnt++;
    }
    err = MPI_Waitall(reqcnt, requests, status);
    assert(err == MPI_SUCCESS);

    if (H.box[RIGHT_BOX] != -1) {
        i = H.nx + ExtraLayer;
#ifdef TARGETON
        // put the buffer on the GPU
#ifdef TRACKDATA
    fprintf(stderr, "Moving mpileftright update to\n");
#endif
#pragma omp target update to(recvbufru [0:size])
#endif
        size = unpack_arrayv(i, H, Hv, recvbufru);
    }

    if (H.box[LEFT_BOX] != -1) {
        i = 0;
#ifdef TARGETON
        // put the buffer on the GPU
#ifdef TRACKDATA
    fprintf(stderr, "Moving mpileftright update to\n");
#endif
#pragma omp target update to(recvbufld [0:size])
#endif
        size = unpack_arrayv(i, H, Hv, recvbufld);
    }
#ifdef TRACKDATA
    fprintf(stderr, "Moving mpileftright OUT\n");
#endif
#endif
}

void mpiupdown(int idim, const hydroparam_t H, hydrovar_t *Hv, real_t *sendbufru, real_t *sendbufld,
               real_t *recvbufru, real_t *recvbufld) {
    int i, ivar, i0, j, j0, err, size;
    real_t sign;
    int reqcnt = 0;

#ifdef MPI
    MPI_Request requests[4];
    MPI_Status status[4];
    MPI_Datatype mpiFormat = MPI_DOUBLE;

#ifdef TRACKDATA
    fprintf(stderr, "Moving mpiupdown IN\n");
#endif
    if (sizeof(real_t) == sizeof(float))
        mpiFormat = MPI_FLOAT;

    j = ExtraLayer;
    size = pack_arrayh(j, H, Hv, sendbufld);
#ifdef TARGETON
    // get the buffer from the GPU
#ifdef TRACKDATA
    fprintf(stderr, "Moving mpiupdown update from\n");
#endif
#pragma omp target update from(sendbufld [0:size])
#endif

    j = H.ny;
    size = pack_arrayh(j, H, Hv, sendbufru);
#ifdef TARGETON
    // get the buffer from the GPU
#ifdef TRACKDATA
    fprintf(stderr, "Moving mpiupdown update from\n");
#endif
#pragma omp target update from(sendbufru [0:size])
#endif

    if (H.box[DOWN_BOX] != -1) {
        MPI_Isend(sendbufld, size, mpiFormat, H.box[DOWN_BOX], 123, MPI_COMM_WORLD,
                  &requests[reqcnt]);
        reqcnt++;
    }
    if (H.box[UP_BOX] != -1) {
        MPI_Isend(sendbufru, size, mpiFormat, H.box[UP_BOX], 246, MPI_COMM_WORLD,
                  &requests[reqcnt]);
        reqcnt++;
    }
    if (H.box[DOWN_BOX] != -1) {
        MPI_Irecv(recvbufld, size, mpiFormat, H.box[DOWN_BOX], 246, MPI_COMM_WORLD,
                  &requests[reqcnt]);
        reqcnt++;
    }
    if (H.box[UP_BOX] != -1) {
        MPI_Irecv(recvbufru, size, mpiFormat, H.box[UP_BOX], 123, MPI_COMM_WORLD,
                  &requests[reqcnt]);
        reqcnt++;
    }
    err = MPI_Waitall(reqcnt, requests, status);
    assert(err == MPI_SUCCESS);

    if (H.box[DOWN_BOX] != -1) {
#ifdef TARGETON
        // put the buffer on the GPU
#ifdef TRACKDATA
    fprintf(stderr, "Moving mpiupdown update to\n");
#endif
#pragma omp target update to(recvbufld [0:size])
#endif
        j = 0;
        unpack_arrayh(j, H, Hv, recvbufld);
    }
    if (H.box[UP_BOX] != -1) {
#ifdef TARGETON
        // put the buffer on the GPU
#ifdef TRACKDATA
    fprintf(stderr, "Moving mpiupdown update to\n");
#endif
#pragma omp target update to(recvbufru [0:size])
#endif
        j = H.ny + ExtraLayer;
        unpack_arrayh(j, H, Hv, recvbufru);
    }
#ifdef TRACKDATA
    fprintf(stderr, "Moving mpiupdown OUT\n");
#endif
#endif
}

// EOF
