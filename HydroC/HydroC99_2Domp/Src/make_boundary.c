/*
  A simple 2D hydro code
  (C) Romain Teyssier : CEA/IRFU           -- original F90 code
  (C) Pierre-Francois Lavallee : IDRIS      -- original F90 code
  (C) Guillaume Colin de Verdiere : CEA/DAM -- for the C version
  (C) Adèle Villiermet : CINES            -- for FTI integration
*/
/*

This software is governed by the CeCILL license under French law and
abiding by the rules of distribution of free software.  You can  use, 
modify and/ or redistribute the software under the terms of the CeCILL
license as circulated by CEA, CNRS and INRIA at the following URL
"http://www.cecill.info". 

As a counterpart to the access to the source code and  rights to copy,
modify and redistribute granted by the license, users are provided only
with a limited warranty  and the software's author,  the holder of the
economic rights,  and the successive licensors  have only  limited
liability. 

In this respect, the user's attention is drawn to the risks associated
with loading,  using,  modifying and/or developing or reproducing the
software by the user in light of its specific status of free software,
that may mean  that it is complicated to manipulate,  and  that  also
therefore means  that it is reserved for developers  and  experienced
professionals having in-depth computer knowledge. Users are therefore
encouraged to load and test the software's suitability as regards their
requirements in conditions enabling the security of their systems and/or 
data to be ensured and,  more generally, to use and operate it in the 
same conditions as regards security. 

The fact that you are presently reading this means that you have had
knowledge of the CeCILL license and that you accept its terms.

*/

#ifdef MPI
#include <mpi.h>
#if FTI>0
#include <fti.h>
#endif
#endif
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <strings.h>
#include <assert.h>


#include "parametres.h"
#include "make_boundary.h"
#include "perfcnt.h"
#include "utils.h"

static int
pack_arrayv(const int xmin, const hydroparam_t H, hydrovar_t * Hv, real_t *buffer);
static int
unpack_arrayv(const int xmin, const hydroparam_t H, hydrovar_t * Hv, real_t *buffer);
static int
pack_arrayh(const int xmin, const hydroparam_t H, hydrovar_t * Hv, real_t *buffer);
static int
unpack_arrayh(const int xmin, const hydroparam_t H, hydrovar_t * Hv, real_t *buffer);

int
pack_arrayv(const int xmin, const hydroparam_t H, hydrovar_t * Hv, real_t *buffer) {
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
unpack_arrayv(const int xmin, const hydroparam_t H, hydrovar_t * Hv, real_t *buffer) {
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
pack_arrayh(const int ymin, const hydroparam_t H, hydrovar_t * Hv, real_t *buffer) {
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
unpack_arrayh(const int ymin, const hydroparam_t H, hydrovar_t * Hv, real_t *buffer) {
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
print_bufferh(FILE * fic, const int ymin, const hydroparam_t H, hydrovar_t * Hv, real_t *buffer) {
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

void
make_boundary(int idim, const hydroparam_t H, hydrovar_t * Hv) {

  // - - - - - - - - - - - - - - - - - - -
  // Cette portion de code est à vérifier
  // détail. J'ai des doutes sur la conversion
  // des index depuis fortran.
  // - - - - - - - - - - - - - - - - - - -
  int i, ivar, i0, j, j0, err, size;
  real_t sign;
  real_t sendbufld[ExtraLayerTot * H.nxyt * H.nvar];
  real_t sendbufru[ExtraLayerTot * H.nxyt * H.nvar];
  //   real_t *sendbufru, *sendbufld;
  real_t recvbufru[ExtraLayerTot * H.nxyt * H.nvar];
  real_t recvbufld[ExtraLayerTot * H.nxyt * H.nvar];
  //   real_t *recvbufru, *recvbufld;
#ifdef MPI
  MPI_Request requests[4];
  MPI_Status status[4];
  MPI_Datatype mpiFormat = MPI_DOUBLE;
#endif
  int reqcnt = 0;

  static FILE *fic = NULL;

  WHERE("make_boundary");

#ifdef MPI
  if (sizeof(real_t) == sizeof(float))  mpiFormat = MPI_FLOAT;
#endif

  if (idim == 1) {
#ifdef MPI
    i = ExtraLayer;
    size = pack_arrayv(i, H, Hv, sendbufld);
    i = H.nx;
    size = pack_arrayv(i, H, Hv, sendbufru);
#if FTI==0
    if (H.box[RIGHT_BOX] != -1) {
      MPI_Isend(sendbufru, size, mpiFormat, H.box[RIGHT_BOX], 123, MPI_COMM_WORLD, &requests[reqcnt]);
      reqcnt++;
    }
    if (H.box[LEFT_BOX] != -1) {
      MPI_Isend(sendbufld, size, mpiFormat, H.box[LEFT_BOX], 246, MPI_COMM_WORLD, &requests[reqcnt]);
      reqcnt++;
    }
    if (H.box[RIGHT_BOX] != -1) {
      MPI_Irecv(recvbufru, size, mpiFormat, H.box[RIGHT_BOX], 246, MPI_COMM_WORLD, &requests[reqcnt]);
      reqcnt++;
    }
    if (H.box[LEFT_BOX] != -1) {
      MPI_Irecv(recvbufld, size, mpiFormat, H.box[LEFT_BOX], 123, MPI_COMM_WORLD, &requests[reqcnt]);
      reqcnt++;
    }
#endif
#if FTI>0
    if (H.box[RIGHT_BOX] != -1) {
      MPI_Isend(sendbufru, size, mpiFormat, H.box[RIGHT_BOX], 123, FTI_COMM_WORLD, &requests[reqcnt]);
      reqcnt++;
    }
    if (H.box[LEFT_BOX] != -1) {
      MPI_Isend(sendbufld, size, mpiFormat, H.box[LEFT_BOX], 246, FTI_COMM_WORLD, &requests[reqcnt]);
      reqcnt++;
    }
    if (H.box[RIGHT_BOX] != -1) {
      MPI_Irecv(recvbufru, size, mpiFormat, H.box[RIGHT_BOX], 246, FTI_COMM_WORLD, &requests[reqcnt]);
      reqcnt++;
    }
    if (H.box[LEFT_BOX] != -1) {
      MPI_Irecv(recvbufld, size, mpiFormat, H.box[LEFT_BOX], 123, FTI_COMM_WORLD, &requests[reqcnt]);
      reqcnt++;
    }
#endif
    err = MPI_Waitall(reqcnt, requests, status);
    assert(err == MPI_SUCCESS);

    if (H.box[RIGHT_BOX] != -1) {
      {
        i = H.nx + ExtraLayer;
        size = unpack_arrayv(i, H, Hv, recvbufru);
      }
    }

    if (H.box[LEFT_BOX] != -1) {
      {
        i = 0;
        size = unpack_arrayv(i, H, Hv, recvbufld);
      }
    }
#endif

    if (H.boundary_left > 0) {
      // Left boundary
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
// #pragma simd
          for (j = H.jmin + ExtraLayer; j < H.jmax - ExtraLayer; j++) {
            Hv->uold[IHv(i, j, ivar)] = Hv->uold[IHv(i0, j, ivar)] * sign;
          }
        }
      }
      { 
	int nops = H.nvar * ExtraLayer * ((H.jmax - ExtraLayer) - (H.jmin + ExtraLayer));
	FLOPS(1 * nops, 0 * nops, 0 * nops, 0 * nops);
      }
    }

    if (H.boundary_right > 0) {
      // Right boundary
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
// #pragma simd
          for (j = H.jmin + ExtraLayer; j < H.jmax - ExtraLayer; j++) {
		  Hv->uold[IHv(i, j, ivar)] = Hv->uold[IHv(i0, j, ivar)] * sign;
          } // for j
        } // for i
      }
      { 
	int nops = H.nvar * ((H.jmax - ExtraLayer) - (H.jmin + ExtraLayer)) * ((H.nx + ExtraLayerTot) - (H.nx + ExtraLayer));
	FLOPS(1 * nops, 0 * nops, 0 * nops, 0 * nops);
      }
    }
  } else {
#ifdef MPI
    {
      if (fic) {
        fprintf(fic, "- = - = - = - Avant\n");
        printuoldf(fic, H, Hv);
      }
    }
    j = ExtraLayer;
    size = pack_arrayh(j, H, Hv, sendbufld);
    // fprintf(stderr, "%d prep %d\n", H.mype, j);
    if (fic) {
      fprintf(fic, "%d prep %d\n", H.mype, j);
      print_bufferh(fic, j, H, Hv, sendbufld);
    }
    j = H.ny;
    size = pack_arrayh(j, H, Hv, sendbufru);
    // fprintf(stderr, "%d prep %d (s=%d)\n", H.mype, j, size);
    if (fic) {
      fprintf(fic, "%d prep %d\n", H.mype, j);
      print_bufferh(fic, j, H, Hv, sendbufru);
    }
#if FTI==0
    if (H.box[DOWN_BOX] != -1) {
      MPI_Isend(sendbufld, size, mpiFormat, H.box[DOWN_BOX], 123, MPI_COMM_WORLD, &requests[reqcnt]);
      reqcnt++;
    }
    if (H.box[UP_BOX] != -1) {
      MPI_Isend(sendbufru, size, mpiFormat, H.box[UP_BOX], 246, MPI_COMM_WORLD, &requests[reqcnt]);
      reqcnt++;
    }
    if (H.box[DOWN_BOX] != -1) {
      MPI_Irecv(recvbufld, size, mpiFormat, H.box[DOWN_BOX], 246, MPI_COMM_WORLD, &requests[reqcnt]);
      reqcnt++;
    }
    if (H.box[UP_BOX] != -1) {
      MPI_Irecv(recvbufru, size, mpiFormat, H.box[UP_BOX], 123, MPI_COMM_WORLD, &requests[reqcnt]);
      reqcnt++;
    }
#endif
#if FTI>0
    if (H.box[DOWN_BOX] != -1) {
      MPI_Isend(sendbufld, size, mpiFormat, H.box[DOWN_BOX], 123, FTI_COMM_WORLD, &requests[reqcnt]);
      reqcnt++;
    }
    if (H.box[UP_BOX] != -1) {
      MPI_Isend(sendbufru, size, mpiFormat, H.box[UP_BOX], 246, FTI_COMM_WORLD, &requests[reqcnt]);
      reqcnt++;
    }
    if (H.box[DOWN_BOX] != -1) {
      MPI_Irecv(recvbufld, size, mpiFormat, H.box[DOWN_BOX], 246, FTI_COMM_WORLD, &requests[reqcnt]);
      reqcnt++;
    }
    if (H.box[UP_BOX] != -1) {
      MPI_Irecv(recvbufru, size, mpiFormat, H.box[UP_BOX], 123, FTI_COMM_WORLD, &requests[reqcnt]);
      reqcnt++;
    }
#endif
    err = MPI_Waitall(reqcnt, requests, status);
    assert(err == MPI_SUCCESS);

    if (H.box[DOWN_BOX] != -1) {
      {
        j = 0;
        unpack_arrayh(j, H, Hv, recvbufld);
        if (fic) {
          fprintf(fic, "%d down %d\n", H.mype, j);
          print_bufferh(fic, j, H, Hv, recvbufld);
        }
        // fprintf(stderr, "%d down %d\n", H.mype, j);
      }
    }
    if (H.box[UP_BOX] != -1) {
      {
        j = H.ny + ExtraLayer;
        unpack_arrayh(j, H, Hv, recvbufru);
        if (fic) {
          fprintf(fic, "%d up %d\n", H.mype, j);
          print_bufferh(fic, j, H, Hv, recvbufru);
        }
        // fprintf(stderr, "%d up %d\n", H.mype, j);
      }
    }
    // if (H.mype == 0) 
    {
      if (fic) {
        fprintf(fic, "- = - = - = - Apres\n");
        printuoldf(fic, H, Hv);
      }
    }
#endif
    // Lower boundary
    if (H.boundary_down > 0) {
      j0 = 0;
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
// #pragma simd
          for (i = H.imin + ExtraLayer; i < H.imax - ExtraLayer; i++) {
            Hv->uold[IHv(i, j, ivar)] = Hv->uold[IHv(i, j0, ivar)] * sign;
          }
        }
      }
      { 
	int nops = H.nvar * ((ExtraLayer) - (0)) * ((H.imax - ExtraLayer) - (H.imin + ExtraLayer));
	FLOPS(1 * nops, 0 * nops, 0 * nops, 0 * nops);
      }
    }
    // Upper boundary
    if (H.boundary_up > 0) {
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
// #pragma simd
          for (i = H.imin + ExtraLayer; i < H.imax - ExtraLayer; i++) {
            Hv->uold[IHv(i, j, ivar)] = Hv->uold[IHv(i, j0, ivar)] * sign;
          }
        }
      }
      { 
	int nops = H.nvar * ((H.ny + ExtraLayerTot) - (H.ny + ExtraLayer)) * ((H.imax - ExtraLayer) - (H.imin + ExtraLayer));
	FLOPS(1 * nops, 0 * nops, 0 * nops, 0 * nops);
      }
    }
  }
}

// make_boundary
//EOF
