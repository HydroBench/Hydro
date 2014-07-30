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

#ifdef WITHMPI
 #ifdef SEEK_SET
  #undef SEEK_SET
  #undef SEEK_CUR
  #undef SEEK_END
 #endif
 #include <mpi.h>
#endif
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <strings.h>
#include <assert.h>

#include "parametres.h"
#include "cuMakeBoundary.h"
#include "gridfuncs.h"
#include "perfcnt.h"
#include "utils.h"

#ifdef IHv
#undef IHv
#endif

#define IHv(i,j,v) ((i) + (Hnxt * (Hnyt * (v) + (j))))
#define IHv2v(i,j,v) ((i) + (ExtraLayer * (Hnyt * (v) + (j))))
#define IHv2h(i,j,v) ((i) + (Hnxt * (ExtraLayer * (v) + (j))))

#define VALPERLINE 11
int
print_bufferh(FILE * fic, const int ymin, const hydroparam_t H, hydrovar_t * Hv, double *buffer) {
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
print_bufferv(FILE * fic, const int xmin, const hydroparam_t H, hydrovar_t * Hv, double *buffer) {
  int ivar, i, j, p = 0, nbr = 1;
  int Hnyt = H.nyt;
  fprintf(fic, "BufferV\n");
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

__global__ void
pack_arrayv(const int xmin, const long Hnxt, const long Hnyt, const long Hnvar, double *buffer, double *uold) {
  int ivar, i;
  int j = idx1d();
  if (j >= Hnyt)
    return;

  for (ivar = 0; ivar < Hnvar; ivar++) {
    for (i = xmin; i < xmin + ExtraLayer; i++) {
      buffer[IHv2v(i - xmin, j, ivar)] = uold[IHv(i, j, ivar)];
    }
  }
}

__global__ void
unpack_arrayv(const int xmin, const long Hnxt, const long Hnyt, const long Hnvar, double *buffer, double *uold) {
  int ivar, i;
  int j = idx1d();
  if (j >= Hnyt)
    return;

  for (ivar = 0; ivar < Hnvar; ivar++) {
    for (i = xmin; i < xmin + ExtraLayer; i++) {
      uold[IHv(i, j, ivar)] = buffer[IHv2v(i - xmin, j, ivar)];
    }
  }
}

__global__ void
pack_arrayh(const int ymin, const long Hnxt, const long Hnyt, const long Hnvar, double *buffer, double *uold) {
  int ivar, j;
  int i = idx1d();
  if (i >= Hnxt)
    return;

  for (ivar = 0; ivar < Hnvar; ivar++) {
    for (j = ymin; j < ymin + ExtraLayer; j++) {
      buffer[IHv2h(i, j - ymin, ivar)] = uold[IHv(i, j, ivar)];
    }
  }
}

__global__ void
unpack_arrayh(const int ymin, const long Hnxt, const long Hnyt, const long Hnvar, double *buffer, double *uold) {
  int ivar, j;
  int i = idx1d();
  if (i >= Hnxt)
    return;

  for (ivar = 0; ivar < Hnvar; ivar++) {
    for (j = ymin; j < ymin + ExtraLayer; j++) {
      uold[IHv(i, j, ivar)] = buffer[IHv2h(i, j - ymin, ivar)];
    }
  }
}

__global__ void
pack_arrayv1(const int xmin, const long Hnxt, const long Hnyt, const long Hnvar, double *buffer, double *uold) {
  int ivar, i, p = 0;
  int j = idx1d();
  if (j >= 1)
    return;

  for (ivar = 0; ivar < Hnvar; ivar++) {
    for (j = 0; j < Hnyt; j++) {
      for (i = xmin; i < xmin + ExtraLayer; i++) {
        buffer[p++] = uold[IHv(i, j, ivar)];
      }
    }
  }
  // printf("pack_arrayv1 %d \n", p);
}

__global__ void
unpack_arrayv1(const int xmin, const long Hnxt, const long Hnyt, const long Hnvar, double *buffer, double *uold) {
  int ivar, i, p = 0;
  int j = idx1d();
  if (j >= 1)
    return;
  for (ivar = 0; ivar < Hnvar; ivar++) {
    for (j = 0; j < Hnyt; j++) {
      for (i = xmin; i < xmin + ExtraLayer; i++) {
        uold[IHv(i, j, ivar)] = buffer[p++];
      }
    }
  }
}

__global__ void
pack_arrayh1(const int ymin, const long Hnxt, const long Hnyt, const long Hnvar, double *buffer, double *uold) {
  int ivar, j, p = 0;
  int i = idx1d();
  if (i >= 1)
    return;

  for (ivar = 0; ivar < Hnvar; ivar++) {
    for (j = ymin; j < ymin + ExtraLayer; j++) {
      for (i = 0; i < Hnxt; i++) {
        buffer[p++] = uold[IHv(i, j, ivar)];
      }
    }
  }
  // printf("pack_arrayh1 %d \n", p);
}

__global__ void
unpack_arrayh1(const int ymin, const long Hnxt, const long Hnyt, const long Hnvar, double *buffer, double *uold) {
  int ivar, j, p = 0;
  int i = idx1d();
  if (i >= 1)
    return;

  for (ivar = 0; ivar < Hnvar; ivar++) {
    for (j = ymin; j < ymin + ExtraLayer; j++) {
      for (i = 0; i < Hnxt; i++) {
        uold[IHv(i, j, ivar)] = buffer[p++];
      }
    }
  }
}

__global__ void
Loop1KcuMakeBoundary(const long i, const long i0, const double sign, const long Hjmin,
                     const long nx, const long Hnxt, const long Hnyt, const long Hnvar, double *uold) {
  int j, ivar;
  double vsign = sign;
  idx2d(j, ivar, nx);
  if (ivar >= Hnvar)
    return;

  // recuperation de la conditon qui etait dans la boucle
  if (ivar == IU)
    vsign = -1.0;

  j += (Hjmin + ExtraLayer);
  uold[IHv(i, j, ivar)] = uold[IHv(i0, j, ivar)] * vsign;
}

__global__ void
Loop2KcuMakeBoundary(const long j, const long j0, const double sign, const long Himin,
                     const long nx, const long Hnxt, const long Hnyt, const long Hnvar, double *uold) {
  int i, ivar;
  double vsign = sign;
  idx2d(i, ivar, nx);
  if (ivar >= Hnvar)
    return;

  // recuperation de la conditon qui etait dans la boucle
  if (ivar == IV)
    vsign = -1.0;

  i += (Himin + ExtraLayer);
  uold[IHv(i, j, ivar)] = uold[IHv(i, j0, ivar)] * vsign;
}

void
cuMakeBoundary(long idim, const hydroparam_t H, hydrovar_t * Hv, double *uoldDEV) {
  dim3 grid, block;
  long i, i0, j, j0;
  long n;
  double sign;
  int err, size;
  double sendbufld[ExtraLayer * H.nxyt * H.nvar];
  double sendbufru[ExtraLayer * H.nxyt * H.nvar];
  double recvbufru[ExtraLayer * H.nxyt * H.nvar];
  double recvbufld[ExtraLayer * H.nxyt * H.nvar];
  double *sendbufruDEV, *sendbufldDEV;
  double *recvbufruDEV, *recvbufldDEV;
  // MPI_Status st;
  // MPI_Win winld, winru;
#ifdef WITHMPI
  MPI_Request requests[4];
  MPI_Status status[4];
#endif
  int reqcnt = 0;
  int nops;

  static FILE *fic = NULL;
  char fname[256];

  if (H.nproc > 1) {
#ifdef WITHMPI
    cudaMalloc(&recvbufruDEV, ExtraLayer * H.nxyt * H.nvar * sizeof(double));
    CheckErr("recvbufruDEV");
    cudaMalloc(&recvbufldDEV, ExtraLayer * H.nxyt * H.nvar * sizeof(double));
    CheckErr("recvbufldDEV");
    cudaMalloc(&sendbufldDEV, ExtraLayer * H.nxyt * H.nvar * sizeof(double));
    CheckErr("recvbufldDEV");
    cudaMalloc(&sendbufruDEV, ExtraLayer * H.nxyt * H.nvar * sizeof(double));
    CheckErr("recvbufldDEV");
#endif
  }
  WHERE("make_boundary");
  if (idim == 1) {
    if (H.nproc > 1) {
#ifdef WITHMPI
      SetBlockDims(H.nyt, THREADSSZ, block, grid);
      i = ExtraLayer;
      pack_arrayv <<< grid, block >>> (i, H.nxt, H.nyt, H.nvar, sendbufldDEV, uoldDEV);
      CheckErr("pack_arrayv 1");
      cudaThreadSynchronize();
      CheckErr("cudaThreadSynchronize pack_arrayv 1");
      i = H.nx;
      pack_arrayv <<< grid, block >>> (i, H.nxt, H.nyt, H.nvar, sendbufruDEV, uoldDEV);
      CheckErr("pack_arrayv 2");
      cudaThreadSynchronize();
      CheckErr("cudaThreadSynchronize pack_arrayv 2");

      size = ExtraLayer * H.nyt * H.nvar;
      // fprintf(stderr, "[%d] size pack_arrayv1 %d [%d %d %d %d]\n", H.mype, size, H.box[DOWN_BOX], H.box[UP_BOX], H.box[RIGHT_BOX], H.box[LEFT_BOX]);

      if (H.box[RIGHT_BOX] != -1) {
        cudaMemcpy(sendbufru, sendbufruDEV, size * sizeof(double), cudaMemcpyDeviceToHost);
        CheckErr("pack_arrayh 1");
        MPI_Isend(sendbufru, size, MPI_DOUBLE, H.box[RIGHT_BOX], 123, MPI_COMM_WORLD, &requests[reqcnt]);
        reqcnt++;
      }
      if (H.box[LEFT_BOX] != -1) {
        cudaMemcpy(sendbufld, sendbufldDEV, size * sizeof(double), cudaMemcpyDeviceToHost);
        CheckErr("pack_arrayh 1");
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
          cudaMemcpy(recvbufruDEV, recvbufru, size * sizeof(double), cudaMemcpyHostToDevice);
          CheckErr("pack_arrayh 1");
          unpack_arrayv <<< grid, block >>> (i, H.nxt, H.nyt, H.nvar, recvbufruDEV, uoldDEV);
          CheckErr("unpack_arrayv 1");
        }
      }

      if (H.box[LEFT_BOX] != -1) {
        {
          i = 0;
          cudaMemcpy(recvbufldDEV, recvbufld, size * sizeof(double), cudaMemcpyHostToDevice);
          CheckErr("pack_arrayh 1");
          unpack_arrayv <<< grid, block >>> (i, H.nxt, H.nyt, H.nvar, recvbufldDEV, uoldDEV);
          CheckErr("unpack_arrayv 2");
        }
      }
      cudaThreadSynchronize();
      CheckErr("cudaThreadSynchronize unpack_arrayv");
#endif
    }
    // Left boundary
    n = ((H.jmax - ExtraLayer) - (H.jmin + ExtraLayer));
    SetBlockDims(n * H.nvar, THREADSSZ, block, grid);
    if (H.boundary_left > 0) {
      for (i = 0; i < ExtraLayer; i++) {
        sign = 1.0;
        if (H.boundary_left == 1) {
          i0 = ExtraLayerTot - i - 1;
          //                 if (ivar == IU) {
          //                     sign = -1.0;
          //                 }
        } else if (H.boundary_left == 2) {
          i0 = 2;
        } else {
          i0 = H.nx + i;
        }
        // on traite les deux boucles d'un coup
        Loop1KcuMakeBoundary <<< grid, block >>> (i, i0, sign, H.jmin, n, H.nxt, H.nyt, H.nvar, uoldDEV);
        CheckErr("Loop1KcuMakeBoundary");
        cudaThreadSynchronize();
        CheckErr("After synchronize Loop1KcuMakeBoundary");
      }
      nops = H.nvar * ExtraLayer * ((H.jmax - ExtraLayer) - (H.jmin + ExtraLayer));
      FLOPS(1 * nops, 0 * nops, 0 * nops, 0 * nops);
    }
    // Right boundary
    if (H.boundary_right > 0) {
      for (i = H.nx + ExtraLayer; i < H.nx + ExtraLayerTot; i++) {
        sign = 1.0;
        if (H.boundary_right == 1) {
          i0 = 2 * H.nx + ExtraLayerTot - i - 1;
          //                 if (ivar == IU) {
          //                     sign = -1.0;
          //                 }
        } else if (H.boundary_right == 2) {
          i0 = H.nx + ExtraLayer;
        } else {
          i0 = i - H.nx;
        }
        Loop1KcuMakeBoundary <<< grid, block >>> (i, i0, sign, H.jmin, n, H.nxt, H.nyt, H.nvar, uoldDEV);
        CheckErr("Loop1KcuMakeBoundary 2");
        cudaThreadSynchronize();
        CheckErr("After synchronize Loop1KcuMakeBoundary 2");
      }
      nops = H.nvar * ((H.jmax - ExtraLayer) - (H.jmin + ExtraLayer)) * ((H.nx + ExtraLayerTot) - (H.nx + ExtraLayer));
      FLOPS(1 * nops, 0 * nops, 0 * nops, 0 * nops);    }
  } else {
    if (H.nproc > 1) {
#ifdef WITHMPI
      SetBlockDims(H.nxt, THREADSSZ, block, grid);
      j = ExtraLayer;
      pack_arrayh <<< grid, block >>> (j, H.nxt, H.nyt, H.nvar, sendbufldDEV, uoldDEV);
      CheckErr("pack_arrayh 1");
      cudaThreadSynchronize();
      CheckErr("cudaThreadSynchronize pack_arrayh 1");
      j = H.ny;
      pack_arrayh <<< grid, block >>> (j, H.nxt, H.nyt, H.nvar, sendbufruDEV, uoldDEV);
      CheckErr("pack_arrayh 2");
      cudaThreadSynchronize();
      CheckErr("cudaThreadSynchronize pack_arrayh 2");

      size = ExtraLayer * H.nxt * H.nvar;
      // printf("[%d] size pack_arrayh1 %d [%d %d %d %d]\n", H.mype, size, H.box[DOWN_BOX], H.box[UP_BOX], H.box[RIGHT_BOX], H.box[LEFT_BOX]);

      if (H.box[DOWN_BOX] != -1) {
        cudaMemcpy(sendbufld, sendbufldDEV, size * sizeof(double), cudaMemcpyDeviceToHost);
        CheckErr("pack_arrayh 1");
        // print_bufferh(stderr, ExtraLayer, H, Hv, sendbufld);
        MPI_Isend(sendbufld, size, MPI_DOUBLE, H.box[DOWN_BOX], 123, MPI_COMM_WORLD, &requests[reqcnt]);
        reqcnt++;
      }
      if (H.box[UP_BOX] != -1) {
        cudaMemcpy(sendbufru, sendbufruDEV, size * sizeof(double), cudaMemcpyDeviceToHost);
        CheckErr("pack_arrayh 1");
        // print_bufferh(stderr, j, H, Hv, sendbufru);
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
          cudaMemcpy(recvbufldDEV, recvbufld, size * sizeof(double), cudaMemcpyHostToDevice);
          CheckErr("cudaMemcpy recvbufldDEV 1");
          // print_bufferh(stdout, j, H, Hv, recvbufld);
          unpack_arrayh <<< grid, block >>> (j, H.nxt, H.nyt, H.nvar, recvbufldDEV, uoldDEV);
          CheckErr("unpack_arrayh 1");
        }
      }
      if (H.box[UP_BOX] != -1) {
        {
          j = H.ny + ExtraLayer;
          cudaMemcpy(recvbufruDEV, recvbufru, size * sizeof(double), cudaMemcpyHostToDevice);
          CheckErr("cudaMemcpy recvbufruDEV 2");
          // print_bufferh(stdout, j, H, Hv, recvbufru);
          unpack_arrayh <<< grid, block >>> (j, H.nxt, H.nyt, H.nvar, recvbufruDEV, uoldDEV);
          CheckErr("unpack_arrayh 2");
        }
      }
      cudaThreadSynchronize();
      CheckErr("cudaThreadSynchronize unpack_arrayv");
 #endif
   }

    n = ((H.imax - ExtraLayer) - (H.imin + ExtraLayer));
    SetBlockDims(n * H.nvar, THREADSSZ, block, grid);
    // Lower boundary
    if (H.boundary_down > 0) {
      j0 = 0;
      for (j = 0; j < ExtraLayer; j++) {
        sign = 1.0;
        if (H.boundary_down == 1) {
          j0 = ExtraLayerTot - j - 1;
          //                 if (ivar == IV) {
          //                     sign = -1.0;
          //                 }
        } else if (H.boundary_down == 2) {
          j0 = ExtraLayerTot;
        } else {
          j0 = H.ny + j;
        }
        Loop2KcuMakeBoundary <<< grid, block >>> (j, j0, sign, H.imin, n, H.nxt, H.nyt, H.nvar, uoldDEV);
        CheckErr("Loop2KcuMakeBoundary ");
        cudaThreadSynchronize();
        CheckErr("After synchronize Loop2KcuMakeBoundary ");
      }
      nops = H.nvar * ((ExtraLayer) - (0)) * ((H.imax - ExtraLayer) - (H.imin + ExtraLayer));
      FLOPS(1 * nops, 0 * nops, 0 * nops, 0 * nops);    }
    // Upper boundary
    if (H.boundary_up > 0) {
      for (j = H.ny + ExtraLayer; j < H.ny + ExtraLayerTot; j++) {
        sign = 1.0;
        if (H.boundary_up == 1) {
          j0 = 2 * H.ny + ExtraLayerTot - j - 1;
          //                 if (ivar == IV) {
          //                     sign = -1.0;
          //                 }
        } else if (H.boundary_up == 2) {
          j0 = H.ny + 1;
        } else {
          j0 = j - H.ny;
        }
        Loop2KcuMakeBoundary <<< grid, block >>> (j, j0, sign, H.imin, n, H.nxt, H.nyt, H.nvar, uoldDEV);
        CheckErr("Loop2KcuMakeBoundary 2");
        cudaThreadSynchronize();
        CheckErr("After synchronize Loop2KcuMakeBoundary 2");
      }
      nops = H.nvar * ((H.ny + ExtraLayerTot) - (H.ny + ExtraLayer)) * ((H.imax - ExtraLayer) - (H.imin + ExtraLayer));
      FLOPS(1 * nops, 0 * nops, 0 * nops, 0 * nops);    
    }
  }
  if (H.nproc > 1) {
#ifdef WITHMPI
    cudaFree(sendbufruDEV);
    cudaFree(sendbufldDEV);
    cudaFree(recvbufldDEV);
    cudaFree(recvbufruDEV);
#endif
  }
}                               // make_boundary


//EOF
