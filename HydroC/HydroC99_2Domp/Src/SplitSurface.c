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
#if FTI>0
#include <fti.h>
#endif
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

//
#include "SplitSurface.h"

/*
  This module splits a surface according to a k-d tree
*/

static char
SplitSurface(int level,
             int xmin, int xmax,
             int ymin, int ymax,
             int *xmin1, int *xmax1, int *ymin1, int *ymax1, int *xmin2, int *xmax2, int *ymin2, int *ymax2) {
  char split = 0;

  *xmin1 = *xmin2 = xmin;
  *ymin1 = *ymin2 = ymin;
  *xmax1 = *xmax2 = xmax;
  *ymax1 = *ymax2 = ymax;

  switch (level % 2) {
    case 0:
      if (xmin != (xmax - 1)) {
        *xmin2 = (xmin + xmax + (level % 2)) / 2;
        *xmax1 = *xmin2;
        split = 'X';
        break;
      }
    case 1:
      if (ymin != (ymax - 1)) {
        *ymin2 = (ymin + ymax + (level % 2)) / 2;
        *ymax1 = *ymin2;
        split = 'Y';
        break;
      }
  }                             // switch

  if (!split && ((xmin != (xmax - 1)) || (ymin != (ymax - 1)))) {
    split = SplitSurface(level - 1, xmin, xmax, ymin, ymax, xmin1, xmax1, ymin1, ymax1, xmin2, xmax2, ymin2, ymax2);
  }
  return split;
}

#ifdef WITHKDTREE
void
CalcSubSurface(int xmin, int xmax,
               int ymin, int ymax, int pmin, int pmax, int level, int box[MAXBOX_D], int mype, int pass) {
  long pmid;
  int split;
  int xmin1, xmax1, ymin1, ymax1;
  int xmin2, xmax2, ymin2, ymax2;

  // if (mype == 0) printf("Evaluating a KDTree\n");

  if (pmin == pmax) {
    // We're down to one PE it's our sub vol limits.
    // printf("[%4d] %4d %4d %4d %4d (%d)\n", pmin, xmin, xmax, ymin, ymax, level);
    if (pmin == mype) {
      box[0] = xmin;
      box[1] = xmax;
      box[2] = ymin;
      box[3] = ymax;
    } else {
      // look for a common edge
      if ((box[XMIN_D] == xmax) && (box[YMIN_D] == ymin) && (box[YMAX_D] == ymax))
        box[LEFT_D] = pmin;
      if ((box[XMAX_D] == xmin) && (box[YMIN_D] == ymin) && (box[YMAX_D] == ymax))
        box[RIGHT_D] = pmin;
      if ((box[YMIN_D] == ymax) && (box[XMIN_D] == xmin) && (box[XMAX_D] == xmax))
        box[DOWN_D] = pmin;
      if ((box[YMAX_D] == ymin) && (box[XMIN_D] == xmin) && (box[XMAX_D] == xmax))
        box[UP_D] = pmin;
    }
    return;
  }

  split = SplitSurface(level, xmin, xmax, ymin, ymax, &xmin1, &xmax1, &ymin1, &ymax1, &xmin2, &xmax2, &ymin2, &ymax2);

  if (split) {
    // recurse on sub problems
    pmid = (pmax + pmin) / 2;
    CalcSubSurface(xmin1, xmax1, ymin1, ymax1, pmin, pmid, level + 1, box, mype, pass);
    CalcSubSurface(xmin2, xmax2, ymin2, ymax2, pmid + 1, pmax, level + 1, box, mype, pass);
  } else {
    // cerr << "Too many PE for this problem size " << endl;
    fprintf(stderr, "Too many PE for this problem size => box[0] = -1\n");
    box[0] = -1;
    box[1] = -1;
    box[2] = -1;
    box[3] = -1;
  }
}

#else
void
CalcSubSurface(int xmin, int xmax,
               int ymin, int ymax, int pmin, int pmax, int level, int box[MAXBOX_D], int mype, int pass) {
  int nbpe = (pmax - pmin + 1);
  int ny = (int) sqrt(nbpe);
  int res = (int) (nbpe - ny * ny) / ny;
  int nx = ny + res;
  int pex = mype % nx;
  int pey = mype / nx;
  int lgx = (xmax - xmin + 1);
  int incx = lgx / nx;
  int lgy = (ymax - ymin + 1);
  int incy = lgy / ny;
  static int done = 0;

  if (nx * ny != nbpe) {
    // the closest shape to a square can't be maintain.
    // Let's find another alternative
    int divider = 2;
    int lastdevider = 1;
    while (divider < (int) sqrt(nbpe)) {
      if ((nbpe % divider) == 0) {
	lastdevider = divider;
      }
      divider++;
    }

    // if (mype == 0) printf("Last divider %d\n", lastdevider);

    if(lastdevider == 1) {
      if (mype == 0) {
	fprintf(stderr, "\tERROR: %d can't be devided evenly in x and y\n", nbpe);
	fprintf(stderr, "\tERROR: closest value is %d\n", nx * ny);
	fprintf(stderr, "\tERROR: please adapt the number of process\n");
      }
#ifdef MPI
#if FTI==0
      MPI_Finalize();
#endif
#if FTI>0
      FTI_Finalize();
      MPI_Finalize();
#endif
#endif
      exit(1);
    }
    ny = lastdevider;
    res = (int) (nbpe - ny * ny) / ny;
    nx = ny + res;
    pex = mype % nx;
    pey = mype / nx;
    incx = lgx / nx;
    incy = lgy / ny;
  }

  if ((incx * nx + xmin) < xmax) incx++;
  if ((incy * ny + ymin) < ymax) incy++;

  if (mype == 0 && !done) {
    printf("HydroC: Simple decomposition\n");
    printf("HydroC: nx=%d ny=%d\n", nx, ny);
    done=1;
  }

  box[XMIN_D] = pex * incx + xmin;
  if (box[XMIN_D] < 0) box[XMIN_D] = 0;

  box[XMAX_D] = (pex + 1) * incx + xmin;
  if (box[XMAX_D] > xmax) box[XMAX_D] = xmax;
  

  box[YMIN_D] = pey * incy + ymin;
  if (box[YMIN_D] < 0) box[YMIN_D] = 0;

  box[YMAX_D] = (pey + 1) * incy + ymin;
  if (box[YMAX_D] > ymax) box[YMAX_D] = ymax;

  box[UP_D] = mype + nx;
  if (box[UP_D] >= nbpe)
    box[UP_D] = -1;
  box[DOWN_D] = mype - nx;
  if (box[DOWN_D] < 0)
    box[DOWN_D] = -1;
  box[LEFT_D] = mype - 1;
  if (pex == 0)
    box[LEFT_D] = -1;
  box[RIGHT_D] = mype + 1;
  if (pex + 1 >= nx)
    box[RIGHT_D] = -1;
}
#endif

#ifdef TESTPGM
int
main(int argc, char **argv) {
  int nx = 16;
  int ny = 16;
  int nbproc = 16;
  int mype = 0;
  int bord = 0;

  for (mype = 0; mype < nbproc; mype++) {
    int box[MAXBOX_D] = { -1, -1, -1, -1, -1, -1, -1, -1 };
    // first pass determin our box
    CalcSubSurface(0, nx - 1, 0, ny - 1, 0, nbproc - 1, 0, box, mype, 0);
    // second pass determin our neighbours
    CalcSubSurface(0, nx - 1, 0, ny - 1, 0, nbproc - 1, 0, box, mype, 1);
    // printf("[%4d] %4d %4d %4d %4d / %4d %4d %4d %4d \n", mype, box[XMIN_D], box[XMAX_D], box[YMIN_D], box[YMAX_D], box[UP_D], box[DOWN_D], box[LEFT_D], box[RIGHT_D]);
    if ((box[UP_D] == -1) || (box[DOWN_D] == -1) || (box[LEFT_D] == -1) || (box[RIGHT_D] == -1)
      )
      bord++;
  }
  printf("B=%d / %d\n", bord, nbproc);
  return 0;
}
#endif
//EOF
