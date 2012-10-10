/*
  A simple 2D hydro code
  (C) Romain Teyssier : CEA/IRFU           -- original F90 code
  (C) Pierre-Francois Lavallee : IDRIS      -- original F90 code
  (C) Guillaume Colin de Verdiere : CEA/DAM -- for the C version
  (C) Jeffrey Poznanovic : CSCS             -- for the OpenACC version
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

#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <strings.h>
#include <mpi.h>
#include <assert.h>

#include "parametres.h"
#include "make_boundary.h"
#include "utils.h"

static int
  pack_arrayv(const int xmin, const hydroparam_t H, hydrovar_t * Hv, double *restrict buffer);
static int
  unpack_arrayv(const int xmin, const hydroparam_t H, hydrovar_t * Hv, double *restrict buffer);
static int
  pack_arrayh(const int xmin, const hydroparam_t H, hydrovar_t * Hv, double *restrict buffer);
static int
  unpack_arrayh(const int xmin, const hydroparam_t H, hydrovar_t * Hv, double *restrict buffer);
static MPI_Request send_array(int to, double *buf, int size, int tag);
static MPI_Request recv_array(int from, double *buf, int size, int tag);
static void
  exch_boundary(Box_t dir, const hydroparam_t H, hydrovar_t * Hv, int tag);

MPI_Request
send_array(int to, double *buf, int size, int tag) {
  MPI_Request request = NULL;
  int err = 0;
  err = MPI_Send(buf, size, MPI_DOUBLE, to, tag, MPI_COMM_WORLD);
  assert(err == MPI_SUCCESS);
  return request;
}

MPI_Request
recv_array(int from, double *buf, int size, int tag) {
  MPI_Request request = NULL;
  MPI_Status st;
  int err = 0;

  err = MPI_Recv(buf, size, MPI_DOUBLE, from, tag, MPI_COMM_WORLD, &st);
  assert(err == MPI_SUCCESS);
  return request;
}

int
pack_arrayv(const int xmin, const hydroparam_t H, hydrovar_t * Hv, double *restrict buffer) {
  double *restrict uold = Hv->uold; 
  int ivar, i, j, p = 0;

#pragma acc parallel present(uold[0:H.nxt*H.nyt*H.nvar]) present(buffer[0:ExtraLayerTot*H.nxyt*H.nvar]) 
#pragma acc loop  reduction(+:p)
  for (ivar = 0; ivar < H.nvar; ivar++) {
#pragma acc loop 
    for (j = 0; j < H.nyt; j++) {
#pragma acc loop 
      for (i = xmin; i < xmin + ExtraLayer; i++) {
        buffer[p++] = uold[IHv(i, j, ivar)];
      }
    }
  }

  return p;
}

int
unpack_arrayv(const int xmin, const hydroparam_t H, hydrovar_t * Hv, double *restrict buffer) {
  double *restrict uold = Hv->uold; 
  int ivar, i, j, p = 0;
#pragma acc parallel present(uold[0:H.nxt*H.nyt*H.nvar]) present(buffer[0:ExtraLayerTot*H.nxyt*H.nvar]) 
#pragma acc loop  reduction(+:p)
  for (ivar = 0; ivar < H.nvar; ivar++) {
#pragma acc loop 
    for (j = 0; j < H.nyt; j++) {
#pragma acc loop 
      for (i = xmin; i < xmin + ExtraLayer; i++) {
        uold[IHv(i, j, ivar)] = buffer[p++];
      }
    }
  }
  return p;
}

int
pack_arrayh(const int ymin, const hydroparam_t H, hydrovar_t * Hv, double *buffer) {
  double *restrict uold = Hv->uold; 
  int ivar, i, j, p = 0;

#pragma acc parallel present(uold[0:H.nxt*H.nyt*H.nvar]) present(buffer[0:ExtraLayerTot*H.nxyt*H.nvar]) 
#pragma acc loop  reduction(+:p)
  for (ivar = 0; ivar < H.nvar; ivar++) {
#pragma acc loop 
    for (j = ymin; j < ymin + ExtraLayer; j++) {
#pragma acc loop 
      for (i = 0; i < H.nxt; i++) {
        buffer[p++] = uold[IHv(i, j, ivar)];
      }
    }
  }
  return p;
}

int
unpack_arrayh(const int ymin, const hydroparam_t H, hydrovar_t * Hv, double *buffer) {
  double *restrict uold = Hv->uold; 
  int ivar, i, j, p = 0;

#pragma acc parallel present(uold[0:H.nxt*H.nyt*H.nvar]) present(buffer[0:ExtraLayerTot*H.nxyt*H.nvar]) 
#pragma acc loop  reduction(+:p)
  for (ivar = 0; ivar < H.nvar; ivar++) {
#pragma acc loop 
    for (j = ymin; j < ymin + ExtraLayer; j++) {
#pragma acc loop 
      for (i = 0; i < H.nxt; i++) {
        uold[IHv(i, j, ivar)] = buffer[p++];
      }
    }
  }
  return p;
}

#define VALPERLINE 11
int
print_bufferh(FILE * fic, const int ymin, const hydroparam_t H, hydrovar_t * Hv, double *buffer) {
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
exch_boundary(Box_t dir, const hydroparam_t H, hydrovar_t * Hv, int tag) {
  int i, j, ivar, size, err;
  double sendbuf[ExtraLayerTot * H.nxyt * H.nvar];
  double recvbuf[ExtraLayerTot * H.nxyt * H.nvar];
  MPI_Status st;

  switch (dir) {
    case LEFT_BOX:
      i = ExtraLayer;
      size = pack_arrayv(i, H, Hv, sendbuf);
      fprintf(stderr, "%d WAIT requestW LEFT %d \n", H.mype, H.box[LEFT_BOX]);
      send_array(H.box[LEFT_BOX], sendbuf, size, LEFT_BOX);
      fprintf(stderr, "%d WAIT requestR LEFT %d \n", H.mype, H.box[LEFT_BOX]);
      recv_array(H.box[LEFT_BOX], recvbuf, size, LEFT_BOX);
      i = 0;
      size = unpack_arrayv(i, H, Hv, recvbuf);
      break;
    case RIGHT_BOX:
      i = H.nx;
      size = pack_arrayv(i, H, Hv, sendbuf);
      fprintf(stderr, "%d WAIT requestR RIGHT %d \n", H.mype, H.box[RIGHT_BOX]);
      recv_array(H.box[RIGHT_BOX], recvbuf, size, LEFT_BOX);
      fprintf(stderr, "%d WAIT requestW RIGHT %d \n", H.mype, H.box[RIGHT_BOX]);
      send_array(H.box[RIGHT_BOX], sendbuf, size, LEFT_BOX);
      i = H.nx + ExtraLayer;
      size = unpack_arrayv(i, H, Hv, recvbuf);
      break;
    case UP_BOX:
      //fprintf(stderr, "UP%d\n", H.mype);
      j = H.ny;
      size = pack_arrayh(j, H, Hv, sendbuf);
      fprintf(stderr, "%d WAIT requestW UP %d \n", H.mype, H.box[UP_BOX]);
      send_array(H.box[UP_BOX], sendbuf, size, UP_BOX);
      fprintf(stderr, "%d WAIT requestR UP %d \n", H.mype, H.box[UP_BOX]);
      recv_array(H.box[UP_BOX], recvbuf, size, UP_BOX);
      j = H.ny + ExtraLayer;
      size = unpack_arrayh(j, H, Hv, recvbuf);
      break;
    case DOWN_BOX:
      //fprintf(stderr, "DOWN%d\n", H.mype);
      j = ExtraLayer;
      size = pack_arrayh(j, H, Hv, sendbuf);
      fprintf(stderr, "%d WAIT requestR DOWN %d \n", H.mype, H.box[DOWN_BOX]);
      recv_array(H.box[DOWN_BOX], recvbuf, size, UP_BOX);
      fprintf(stderr, "%d WAIT requestW DOWN %d \n", H.mype, H.box[DOWN_BOX]);
      send_array(H.box[DOWN_BOX], sendbuf, size, UP_BOX);
      j = 0;
      size = unpack_arrayh(j, H, Hv, recvbuf);
      break;
  }
  MPI_Barrier(MPI_COMM_WORLD);
}

void
make_boundary(int idim, const hydroparam_t H, hydrovar_t * Hv) {

  // - - - - - - - - - - - - - - - - - - -
  // Cette portion de code est à vérifier
  // détail. J'ai des doutes sur la conversion
  // des index depuis fortran.
  // - - - - - - - - - - - - - - - - - - -
  int i, ivar, i0, j, j0, err, size;
  double sign;
  double sendbufld[ExtraLayerTot * H.nxyt * H.nvar];
  double sendbufru[ExtraLayerTot * H.nxyt * H.nvar];
  //   double *sendbufru, *sendbufld;
  double recvbufru[ExtraLayerTot * H.nxyt * H.nvar];
  double recvbufld[ExtraLayerTot * H.nxyt * H.nvar];
  //   double *recvbufru, *recvbufld;
  MPI_Status st;
  MPI_Win winld, winru;
  MPI_Request requests[4];
  MPI_Status status[4];
  int reqcnt = 0;

  static FILE *fic = NULL;
  char fname[256];

  //   if (fic == NULL) {
  //     sprintf(fname, "uold_%05d_%05d.txt", H.mype, H.nproc);
  //     fic = fopen(fname, "w");
  //     assert(fic != NULL);
  //   }

  //   err = MPI_Alloc_mem(ExtraLayerTot * H.nxyt * H.nvar * sizeof(double), MPI_INFO_NULL, &sendbufld);
  //   assert(err == MPI_SUCCESS);
  //   err = MPI_Alloc_mem(ExtraLayerTot * H.nxyt * H.nvar * sizeof(double), MPI_INFO_NULL, &sendbufru);
  //   assert(err == MPI_SUCCESS);
  //   err = MPI_Alloc_mem(ExtraLayerTot * H.nxyt * H.nvar * sizeof(double), MPI_INFO_NULL, &recvbufld);
  //   assert(err == MPI_SUCCESS);
  //   err = MPI_Alloc_mem(ExtraLayerTot * H.nxyt * H.nvar * sizeof(double), MPI_INFO_NULL, &recvbufru);
  //   assert(err == MPI_SUCCESS);


  WHERE("make_boundary");

  double *restrict uold = &(Hv->uold[0]);
#pragma acc data create(sendbufld[0:ExtraLayerTot * H.nxyt * H.nvar], \
                        sendbufru[0:ExtraLayerTot * H.nxyt * H.nvar], \
			recvbufru[0:ExtraLayerTot * H.nxyt * H.nvar], \
			recvbufld[0:ExtraLayerTot * H.nxyt * H.nvar]) \
                 present(uold[0:H.nxt*H.nyt*H.nvar])
{

  if (idim == 1) {
    i = ExtraLayer;
    size = pack_arrayv(i, H, Hv, sendbufld);
    //cudaThreadSynchronize();
    //acc_async_wait_all();
    i = H.nx;
    size = pack_arrayv(i, H, Hv, sendbufru);
    //acc_async_wait_all();

    if (H.box[RIGHT_BOX] != -1) {
      //cudaMemcpy(recvbufruDEV, recvbufru, size * sizeof(double), cudaMemcpyHostToDevice);
      #pragma acc update device(recvbufru[0:size])
      MPI_Isend(sendbufru, size, MPI_DOUBLE, H.box[RIGHT_BOX], 123, MPI_COMM_WORLD, &requests[reqcnt]);
      reqcnt++;
    }
    if (H.box[LEFT_BOX] != -1) {
      //cudaMemcpy(recvbufldDEV, recvbufld, size * sizeof(double), cudaMemcpyHostToDevice);
      #pragma acc update device(recvbufld[0:size])
      MPI_Isend(sendbufld, size, MPI_DOUBLE, H.box[LEFT_BOX], 246, MPI_COMM_WORLD, &requests[reqcnt]);
      reqcnt++;
    }
    if (H.box[RIGHT_BOX] != -1) {
      MPI_Irecv(recvbufru, size, MPI_DOUBLE, H.box[RIGHT_BOX], 246, MPI_COMM_WORLD, &requests[reqcnt]);
      reqcnt++;
    }
    if (H.box[LEFT_BOX] != -1) {
      MPI_Irecv(recvbufld, size, MPI_DOUBLE, H.box[LEFT_BOX], 123, MPI_COMM_WORLD, &requests[reqcnt]);
      reqcnt++;
    }

    err = MPI_Waitall(reqcnt, requests, status);
    assert(err == MPI_SUCCESS);

    if (H.box[RIGHT_BOX] != -1) {
      {
        i = H.nx + ExtraLayer;
	//cudaMemcpy(recvbufruDEV, recvbufru, size * sizeof(double), cudaMemcpyHostToDevice);
        #pragma acc update device(recvbufru[0:size])
        size = unpack_arrayv(i, H, Hv, recvbufru);
      }
    }

    if (H.box[LEFT_BOX] != -1) {
      {
        i = 0;
	//cudaMemcpy(recvbufldDEV, recvbufld, size * sizeof(double), cudaMemcpyHostToDevice);
        #pragma acc update device(recvbufld[0:size])
        size = unpack_arrayv(i, H, Hv, recvbufld);
      }
    }
    
    //acc_async_wait_all();

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
#if 1
          //#pragma acc parallel loop
          for (j = H.jmin + ExtraLayer; j < H.jmax - ExtraLayer; j++) {
            uold[IHv(i, j, ivar)] = uold[IHv(i0, j, ivar)] * sign;
          }
	  //acc_async_wait_all();
#else
#pragma acc host_data use_device(uold)
        Loop1KcuMakeBoundary <<< grid, block >>> (i, i0, sign, H.jmin, n, H.nxt, H.nyt, H.nvar, uold);
#endif
        }
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
	  //#pragma acc parallel loop
          for (j = H.jmin + ExtraLayer; j < H.jmax - ExtraLayer; j++) {
            uold[IHv(i, j, ivar)] = uold[IHv(i0, j, ivar)] * sign;
          }
	  //acc_async_wait_all();
        }
      }
    }
  } else {
    {
      if (fic) {
        fprintf(fic, "- = - = - = - Avant\n");
        printuoldf(fic, H, Hv);
      }
    }
    j = ExtraLayer;
    size = pack_arrayh(j, H, Hv, sendbufld);
    //acc_async_wait_all();

    // fprintf(stderr, "%d prep %d\n", H.mype, j);
    if (fic) {
      fprintf(fic, "%d prep %d\n", H.mype, j);
      print_bufferh(fic, j, H, Hv, sendbufld);
    }
    j = H.ny;
    size = pack_arrayh(j, H, Hv, sendbufru);
    //acc_async_wait_all();

    // fprintf(stderr, "%d prep %d (s=%d)\n", H.mype, j, size);
    if (fic) {
      fprintf(fic, "%d prep %d\n", H.mype, j);
      print_bufferh(fic, j, H, Hv, sendbufru);
    }

    if (H.box[DOWN_BOX] != -1) {
      //cudaMemcpy(sendbufld, sendbufldDEV, size * sizeof(double), cudaMemcpyDeviceToHost);
      #pragma acc update host(sendbufld[0:size])
      MPI_Isend(sendbufld, size, MPI_DOUBLE, H.box[DOWN_BOX], 123, MPI_COMM_WORLD, &requests[reqcnt]);
      reqcnt++;
    }
    if (H.box[UP_BOX] != -1) {
      //cudaMemcpy(sendbufru, sendbufruDEV, size * sizeof(double), cudaMemcpyDeviceToHost);
      #pragma acc update host(sendbufru[0:size])
      MPI_Isend(sendbufru, size, MPI_DOUBLE, H.box[UP_BOX], 246, MPI_COMM_WORLD, &requests[reqcnt]);
      reqcnt++;
    }
    if (H.box[DOWN_BOX] != -1) {
      MPI_Irecv(recvbufld, size, MPI_DOUBLE, H.box[DOWN_BOX], 246, MPI_COMM_WORLD, &requests[reqcnt]);
      reqcnt++;
    }
    if (H.box[UP_BOX] != -1) {
      MPI_Irecv(recvbufru, size, MPI_DOUBLE, H.box[UP_BOX], 123, MPI_COMM_WORLD, &requests[reqcnt]);
      reqcnt++;
    }

    err = MPI_Waitall(reqcnt, requests, status);
    assert(err == MPI_SUCCESS);

    if (H.box[DOWN_BOX] != -1) {
      {
        j = 0;
	//cudaMemcpy(recvbufldDEV, recvbufld, size * sizeof(double), cudaMemcpyHostToDevice);
        #pragma acc update device(recvbufld[0:size])
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
	//cudaMemcpy(recvbufruDEV, recvbufru, size * sizeof(double), cudaMemcpyHostToDevice);
        #pragma acc update device(recvbufru[0:size])
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
    //acc_async_wait_all();

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
	  //#pragma acc parallel loop
          for (i = H.imin + ExtraLayer; i < H.imax - ExtraLayer; i++) {
            uold[IHv(i, j, ivar)] = uold[IHv(i, j0, ivar)] * sign;
          }
	  //acc_async_wait_all();

        }
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
	  //#pragma acc parallel loop
          for (i = H.imin + ExtraLayer; i < H.imax - ExtraLayer; i++) {
            uold[IHv(i, j, ivar)] = uold[IHv(i, j0, ivar)] * sign;
          }
	  //acc_async_wait_all();

        }
      }
    }
  }

}
  //   MPI_Free_mem(sendbufld);
  //   MPI_Free_mem(sendbufru);
  //   MPI_Free_mem(recvbufld);
  //   MPI_Free_mem(recvbufru);
}

// make_boundary


//EOF
#ifdef NOTDEF
err = MPI_Get(sendbufld, size, MPI_DOUBLE, H.box[DOWN_BOX], 0, size, MPI_DOUBLE, winld);
err = MPI_Get(sendbufru, size, MPI_DOUBLE, H.box[UP_BOX], 0, size, MPI_DOUBLE, winru);
MPI_Win_fence(0, winru);

MPI_Win_free(&winld);
MPI_Win_free(&winru);
MPI_Win_fence(0, winld);

MPI_Win_fence(0, winru);
#endif
