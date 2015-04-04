/*
  A simple 2D hydro code
  (C) Romain Teyssier : CEA/IRFU           -- original F90 code
  (C) Pierre-Francois Lavallee : IDRIS      -- original F90 code
  (C) Guillaume Colin de Verdiere : CEA/DAM -- for the C version
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
#ifdef MPI
#include <mpi.h>
#endif
#include <assert.h>

#include "parametres.h"
#include "oclMakeBoundary.h"
#include "utils.h"
#include "oclInit.h"
#include "ocltools.h"

#define VALPERLINE 11
int
print_bufferh(FILE * fic, const int ymin, const hydroparam_t H, hydrovar_t * Hv, real_t *buffer) {
  int ivar, i, j, p = 0, nbr = 1;
  int Hnxt = H.nxt;
  fprintf(fic, "BufferH\n");
  for (ivar = 3; ivar < H.nvar; ivar++) {
    fprintf(fic, "BufferH v=%d\n", ivar);
    for (j = ymin; j < ymin + ExtraLayer; j++) {
      for (i = 0; i < Hnxt; i++) {
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
  fflush(fic);
  return p;
}

int
print_bufferv(FILE * fic, const int xmin, const hydroparam_t H, hydrovar_t * Hv, real_t *buffer, char *st) {
  int ivar, i, j, p = 0, nbr = 1;
  int Hnyt = H.nyt;
  fprintf(fic, "BufferV %s\n", st);
  for (ivar = 3; ivar < H.nvar; ivar++) {
    fprintf(fic, "BufferV v=%d\n", ivar);
    for (j = 0; j < Hnyt; j++) {
      for (i = xmin; i < xmin + ExtraLayer; i++) {
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
  fflush(fic);
  return p;
}

void
oclMakeBoundary(long idim, const hydroparam_t H, hydrovar_t * Hv, cl_mem uoldDEV) {
  cl_event event;
  cl_int error = 0;
  int i, i0, j, j0;
  int size;
  long n = 1;
  real_t sign;
  real_t *sendbufld;
  real_t *sendbufru;
  real_t *recvbufru;
  real_t *recvbufld;

  cl_mem recvbufruDEV, recvbufldDEV, sendbufldDEV, sendbufruDEV;
#ifdef MPI  
  MPI_Request requests[4];
  MPI_Status status[4];
#endif
  int reqcnt = 0;

  static FILE *fic = NULL;
  char fname[256];

  // if (fic == NULL) {
  //   sprintf(fname, "trace%04d.lst", H.mype);
  //   fic = fopen(fname, "w");
  // }

  if (H.nproc > 1) {
#ifdef MPI
    sendbufld = (real_t *) malloc(ExtraLayer * H.nxyt * H.nvar * sizeof(real_t));
    assert(sendbufld);
    sendbufru = (real_t *) malloc(ExtraLayer * H.nxyt * H.nvar * sizeof(real_t));
    assert(sendbufru);
    recvbufru = (real_t *) malloc(ExtraLayer * H.nxyt * H.nvar * sizeof(real_t));
    assert(recvbufru);
    recvbufld = (real_t *) malloc(ExtraLayer * H.nxyt * H.nvar * sizeof(real_t));
    assert(recvbufld);
    
    recvbufruDEV = clCreateBuffer(ctx, CL_MEM_READ_WRITE, ExtraLayer * H.nxyt * H.nvar * sizeof(real_t), NULL, &error);
    oclCheckErr(error, "");
    recvbufldDEV = clCreateBuffer(ctx, CL_MEM_READ_WRITE, ExtraLayer * H.nxyt * H.nvar * sizeof(real_t), NULL, &error);
    oclCheckErr(error, "");
    sendbufldDEV = clCreateBuffer(ctx, CL_MEM_READ_WRITE, ExtraLayer * H.nxyt * H.nvar * sizeof(real_t), NULL, &error);
    oclCheckErr(error, "");
    sendbufruDEV = clCreateBuffer(ctx, CL_MEM_READ_WRITE, ExtraLayer * H.nxyt * H.nvar * sizeof(real_t), NULL, &error);
    oclCheckErr(error, "");
#endif
  }

  WHERE("make_boundary");
  if (idim == 1) {
    if (H.nproc > 1) {
#ifdef MPI
      i = ExtraLayer;
      // fprintf(stderr, "make_boundary 01--%d\n", H.mype);
      OCLSETARG06(ker[kpack_arrayv], i, H.nxt, H.nyt, H.nvar, sendbufldDEV, uoldDEV);
      oclLaunchKernel(ker[kpack_arrayv], cqueue, H.nyt, THREADSSZ, __FILE__, __LINE__);
      i = H.nx;
      // fprintf(stderr, "make_boundary 02--%d\n", H.mype);
      OCLSETARG06(ker[kpack_arrayv], i, H.nxt, H.nyt, H.nvar, sendbufruDEV, uoldDEV);
      oclLaunchKernel(ker[kpack_arrayv], cqueue, H.nyt, THREADSSZ, __FILE__, __LINE__);

      size = ExtraLayer * H.nyt * H.nvar;
      // fprintf(stderr, "[%d] size pack_arrayv1 %d [%d %d %d %d]\n", H.mype, size, H.box[DOWN_BOX], H.box[UP_BOX], H.box[RIGHT_BOX], H.box[LEFT_BOX]);

      if (H.box[RIGHT_BOX] != -1) {
        error =
          clEnqueueReadBuffer(cqueue, sendbufruDEV, CL_TRUE, 0, size * sizeof(real_t), sendbufru, 0, NULL, &event);
        oclCheckErr(error, "");
	error = clReleaseEvent(event); oclCheckErr(error, "");
	// print_bufferv(fic, H.nx, H, Hv, sendbufru, "H.nx");

        MPI_Isend(sendbufru, size, (sizeof(real_t) == sizeof(double))? MPI_DOUBLE: MPI_FLOAT, H.box[RIGHT_BOX], 123, MPI_COMM_WORLD, &requests[reqcnt]);
        reqcnt++;
      }
      if (H.box[LEFT_BOX] != -1) {
        error =
          clEnqueueReadBuffer(cqueue, sendbufldDEV, CL_TRUE, 0, size * sizeof(real_t), sendbufld, 0, NULL, &event);
        oclCheckErr(error, "");
	error = clReleaseEvent(event); oclCheckErr(error, "");
	// print_bufferv(fic, ExtraLayer, H, Hv, sendbufld, "ExtraLayer");
        MPI_Isend(sendbufld, size, (sizeof(real_t) == sizeof(double))? MPI_DOUBLE: MPI_FLOAT, H.box[LEFT_BOX], 246, MPI_COMM_WORLD, &requests[reqcnt]);
        reqcnt++;
      }
      if (H.box[RIGHT_BOX] != -1) {
        MPI_Irecv(recvbufru, size, (sizeof(real_t) == sizeof(double))? MPI_DOUBLE: MPI_FLOAT, H.box[RIGHT_BOX], 246, MPI_COMM_WORLD, &requests[reqcnt]);
        reqcnt++;
      }
      if (H.box[LEFT_BOX] != -1) {
        MPI_Irecv(recvbufld, size, (sizeof(real_t) == sizeof(double))? MPI_DOUBLE: MPI_FLOAT, H.box[LEFT_BOX], 123, MPI_COMM_WORLD, &requests[reqcnt]);
        reqcnt++;
      }

      error = MPI_Waitall(reqcnt, requests, status);
      assert(error == MPI_SUCCESS);

      if (H.box[RIGHT_BOX] != -1) {
        i = H.nx + ExtraLayer;
        // fprintf(stderr, "make_boundary 03--%d\n", H.mype);
        error =
          clEnqueueWriteBuffer(cqueue, recvbufruDEV, CL_TRUE, 0, size * sizeof(real_t), recvbufru, 0, NULL, &event);
        oclCheckErr(error, "");
	error = clReleaseEvent(event); oclCheckErr(error, "");
        OCLSETARG06(ker[kunpack_arrayv], i, H.nxt, H.nyt, H.nvar, recvbufruDEV, uoldDEV);
        oclLaunchKernel(ker[kunpack_arrayv], cqueue, H.nyt, THREADSSZ, __FILE__, __LINE__);
      }

      if (H.box[LEFT_BOX] != -1) {
        i = 0;
        // fprintf(stderr, "make_boundary 04--%d\n", H.mype);
        error =
          clEnqueueWriteBuffer(cqueue, recvbufldDEV, CL_TRUE, 0, size * sizeof(real_t), recvbufld, 0, NULL, &event);
        oclCheckErr(error, "");
	error = clReleaseEvent(event); oclCheckErr(error, "");
        OCLSETARG06(ker[kunpack_arrayv], i, H.nxt, H.nyt, H.nvar, recvbufldDEV, uoldDEV);
        oclLaunchKernel(ker[kunpack_arrayv], cqueue, H.nyt, THREADSSZ, __FILE__, __LINE__);
      }
#endif
    }
    // Left boundary
    n = ((H.jmax - ExtraLayer) - (H.jmin + ExtraLayer));
    if (H.boundary_left > 0) {
      for (i = 0; i < ExtraLayer; i++) {
        sign = 1.0;
        if (H.boundary_left == 1) {
          i0 = ExtraLayerTot - i - 1;
	  // change of sign is in the kernel
          //         if (ivar == IU) {
          //           sign = -1.0;
          //         }
        } else if (H.boundary_left == 2) {
          i0 = 2;
        } else {
          i0 = H.nx + i;
        }
        OCLSETARG09(ker[Loop1KcuMakeBoundary], i, i0, sign, H.jmin, n, H.nxt, H.nyt, H.nvar, uoldDEV);
        oclLaunchKernel2D(ker[Loop1KcuMakeBoundary], cqueue, n, H.nvar, THREADSSZ, __FILE__, __LINE__);
      }
    }
    // Right boundary
    if (H.boundary_right > 0) {
      for (i = H.nx + ExtraLayer; i < H.nx + ExtraLayerTot; i++) {
        sign = 1.0;
        if (H.boundary_right == 1) {
          i0 = 2 * H.nx + ExtraLayerTot - i - 1;
          //         if (ivar == IU) {
          //           sign = -1.0;
          //         }
        } else if (H.boundary_right == 2) {
          i0 = H.nx + ExtraLayer;
        } else {
          i0 = i - H.nx;
        }
        OCLSETARG09(ker[Loop1KcuMakeBoundary], i, i0, sign, H.jmin, n, H.nxt, H.nyt, H.nvar, uoldDEV);
        oclLaunchKernel2D(ker[Loop1KcuMakeBoundary], cqueue, n, H.nvar, THREADSSZ, __FILE__, __LINE__);
      }
    }
  } else {
    if (H.nproc > 1) {
#ifdef MPI
      // SetBlockDims(H.nxt, THREADSSZ, block, grid);
      j = ExtraLayer;
      // fprintf(stderr, "make_boundary 05--%d\n", H.mype);
      OCLSETARG06(ker[kpack_arrayh], j, H.nxt, H.nyt, H.nvar, sendbufldDEV, uoldDEV);
      oclLaunchKernel(ker[kpack_arrayh], cqueue, H.nxt, THREADSSZ, __FILE__, __LINE__);
      j = H.ny;
      // fprintf(stderr, "make_boundary 06--%d\n", H.mype);
      OCLSETARG06(ker[kpack_arrayh], j, H.nxt, H.nyt, H.nvar, sendbufruDEV, uoldDEV);
      oclLaunchKernel(ker[kpack_arrayh], cqueue, H.nxt, THREADSSZ, __FILE__, __LINE__);

      size = ExtraLayer * H.nxt * H.nvar;
      // printf("[%d] size pack_arrayh1 %d [%d %d %d %d]\n", H.mype, size, H.box[DOWN_BOX], H.box[UP_BOX], H.box[RIGHT_BOX], H.box[LEFT_BOX]);

      if (H.box[DOWN_BOX] != -1) {
        error =
          clEnqueueReadBuffer(cqueue, sendbufldDEV, CL_TRUE, 0, size * sizeof(real_t), sendbufld, 0, NULL, &event);
        oclCheckErr(error, "");
	error = clReleaseEvent(event); oclCheckErr(error, "");
        // print_bufferh(stderr, ExtraLayer, H, Hv, sendbufld);
        MPI_Isend(sendbufld, size, (sizeof(real_t) == sizeof(double))? MPI_DOUBLE: MPI_FLOAT, H.box[DOWN_BOX], 123, MPI_COMM_WORLD, &requests[reqcnt]);
        reqcnt++;
      }
      if (H.box[UP_BOX] != -1) {
        error =
          clEnqueueReadBuffer(cqueue, sendbufruDEV, CL_TRUE, 0, size * sizeof(real_t), sendbufru, 0, NULL, &event);
        oclCheckErr(error, "");
	error = clReleaseEvent(event); oclCheckErr(error, "");
        // print_bufferh(stderr, j, H, Hv, sendbufru);
        MPI_Isend(sendbufru, size, (sizeof(real_t) == sizeof(double))? MPI_DOUBLE: MPI_FLOAT, H.box[UP_BOX], 246, MPI_COMM_WORLD, &requests[reqcnt]);
        reqcnt++;
      }

      if (H.box[DOWN_BOX] != -1) {
        MPI_Irecv(recvbufld, size, (sizeof(real_t) == sizeof(double))? MPI_DOUBLE: MPI_FLOAT, H.box[DOWN_BOX], 246, MPI_COMM_WORLD, &requests[reqcnt]);
        reqcnt++;
      }
      if (H.box[UP_BOX] != -1) {
        MPI_Irecv(recvbufru, size, (sizeof(real_t) == sizeof(double))? MPI_DOUBLE: MPI_FLOAT, H.box[UP_BOX], 123, MPI_COMM_WORLD, &requests[reqcnt]);
        reqcnt++;
      }

      error = MPI_Waitall(reqcnt, requests, status);
      assert(error == MPI_SUCCESS);

      if (H.box[DOWN_BOX] != -1) {
        j = 0;
        error =
          clEnqueueWriteBuffer(cqueue, recvbufldDEV, CL_TRUE, 0, size * sizeof(real_t), recvbufld, 0, NULL, &event);
        oclCheckErr(error, "");
	error = clReleaseEvent(event); oclCheckErr(error, "");
        // print_bufferh(stdout, j, H, Hv, recvbufld);
        // fprintf(stderr, "make_boundary 07--%d\n", H.mype);
        OCLSETARG06(ker[kunpack_arrayh], j, H.nxt, H.nyt, H.nvar, recvbufldDEV, uoldDEV);
        oclLaunchKernel(ker[kunpack_arrayh], cqueue, H.nxt, THREADSSZ, __FILE__, __LINE__);
      }
      if (H.box[UP_BOX] != -1) {
        j = H.ny + ExtraLayer;
        error =
          clEnqueueWriteBuffer(cqueue, recvbufruDEV, CL_TRUE, 0, size * sizeof(real_t), recvbufru, 0, NULL, &event);
        oclCheckErr(error, "");
	error = clReleaseEvent(event); oclCheckErr(error, "");
        // print_bufferh(stdout, j, H, Hv, recvbufru);
        // fprintf(stderr, "make_boundary 08--%d\n", H.mype);
        OCLSETARG06(ker[kunpack_arrayh], j, H.nxt, H.nyt, H.nvar, recvbufruDEV, uoldDEV);
        oclLaunchKernel(ker[kunpack_arrayh], cqueue, H.nxt, THREADSSZ, __FILE__, __LINE__);
      }
#endif      
    }
    n = ((H.imax - ExtraLayer) - (H.imin + ExtraLayer));
    // Lower boundary
    if (H.boundary_down > 0) {
      j0 = 0;
      for (j = 0; j < ExtraLayer; j++) {
        sign = 1.0;
        if (H.boundary_down == 1) {
          j0 = ExtraLayerTot - j - 1;
          //         if (ivar == IV) {
          //           sign = -1.0;
          //         }
        } else if (H.boundary_down == 2) {
          j0 = ExtraLayerTot;
        } else {
          j0 = H.ny + j;
        }
        OCLSETARG09(ker[Loop2KcuMakeBoundary], j, j0, sign, H.imin, n, H.nxt, H.nyt, H.nvar, uoldDEV);
        oclLaunchKernel2D(ker[Loop2KcuMakeBoundary], cqueue, n, H.nvar, THREADSSZ, __FILE__, __LINE__);
      }
    }
    // Upper boundary
    if (H.boundary_up > 0) {
      for (j = H.ny + ExtraLayer; j < H.ny + ExtraLayerTot; j++) {
        sign = 1.0;
        if (H.boundary_up == 1) {
          j0 = 2 * H.ny + ExtraLayerTot - j - 1;
	  // change of sign is in the kernel
          //         if (ivar == IV) {
          //           sign = -1.0;
          //         }
        } else if (H.boundary_up == 2) {
          j0 = H.ny + 1;
        } else {
          j0 = j - H.ny;
        }
        OCLSETARG09(ker[Loop2KcuMakeBoundary], j, j0, sign, H.imin, n, H.nxt, H.nyt, H.nvar, uoldDEV);
        oclLaunchKernel2D(ker[Loop2KcuMakeBoundary], cqueue, n, H.nvar, THREADSSZ, __FILE__, __LINE__);
      }
    }
  }
  if (H.nproc > 1) {
    error = clReleaseMemObject(sendbufruDEV);
    oclCheckErr(error, "make_boundary");
    error = clReleaseMemObject(sendbufldDEV);
    oclCheckErr(error, "make_boundary");
    error = clReleaseMemObject(recvbufldDEV);
    oclCheckErr(error, "make_boundary");
    error = clReleaseMemObject(recvbufruDEV);
    oclCheckErr(error, "make_boundary");
  }
  // fprintf(stderr, "make_boundary END\n");

  if (H.nproc > 1) {
    free(recvbufld);
    free(recvbufru);
    free(sendbufru);
    free(sendbufld);
  }
}                               // make_boundary


//EOF
