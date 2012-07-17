//
// (C) Guillaume.Colin-de-Verdiere at CEA.Fr
//
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
SplitSurface (int level,
	      int xmin, int xmax,
	      int ymin, int ymax,
	      int *xmin1, int *xmax1, int *ymin1, int *ymax1, int *xmin2,
	      int *xmax2, int *ymin2, int *ymax2)
{
  char split = 0;

  *xmin1 = *xmin2 = xmin;
  *ymin1 = *ymin2 = ymin;
  *xmax1 = *xmax2 = xmax;
  *ymax1 = *ymax2 = ymax;

  switch (level % 2)
    {
    case 0:
      if (xmin != (xmax - 1))
	{
	  *xmin2 = (xmin + xmax + (level % 2)) / 2;
	  *xmax1 = *xmin2;
	  split = 'X';
	  break;
	}
    case 1:
      if (ymin != (ymax - 1))
	{
	  *ymin2 = (ymin + ymax + (level % 2)) / 2;
	  *ymax1 = *ymin2;
	  split = 'Y';
	  break;
	}
    }				// switch

  if (!split && ((xmin != (xmax - 1)) || (ymin != (ymax - 1))))
    {
      split =
	SplitSurface (level - 1, xmin, xmax, ymin, ymax, xmin1, xmax1, ymin1,
		      ymax1, xmin2, xmax2, ymin2, ymax2);
    }
  return split;
}

void
CalcSubSurface (int xmin, int xmax,
		int ymin, int ymax, int pmin, int pmax, int level,
		int box[MAXBOX_D], int mype, int pass)
{
  long pmid;
  int split;
  int xmin1, xmax1, ymin1, ymax1;
  int xmin2, xmax2, ymin2, ymax2;

  if (pmin == pmax)
    {
      // We're down to one PE it's our sub vol limits.
      // printf("[%4d] %4d %4d %4d %4d (%d)\n", pmin, xmin, xmax, ymin, ymax, level);
      if (pmin == mype)
	{
	  box[0] = xmin;
	  box[1] = xmax;
	  box[2] = ymin;
	  box[3] = ymax;
	}
      else
	{
	  // look for a common edge
	  if ((box[XMIN_D] == xmax) && (box[YMIN_D] == ymin)
	      && (box[YMAX_D] == ymax))
	    box[LEFT_D] = pmin;
	  if ((box[XMAX_D] == xmin) && (box[YMIN_D] == ymin)
	      && (box[YMAX_D] == ymax))
	    box[RIGHT_D] = pmin;
	  if ((box[YMIN_D] == ymax) && (box[XMIN_D] == xmin)
	      && (box[XMAX_D] == xmax))
	    box[DOWN_D] = pmin;
	  if ((box[YMAX_D] == ymin) && (box[XMIN_D] == xmin)
	      && (box[XMAX_D] == xmax))
	    box[UP_D] = pmin;
	}
      return;
    }

  split =
    SplitSurface (level, xmin, xmax, ymin, ymax, &xmin1, &xmax1, &ymin1,
		  &ymax1, &xmin2, &xmax2, &ymin2, &ymax2);

  if (split)
    {
      // recurse on sub problems
      pmid = (pmax + pmin) / 2;
      CalcSubSurface (xmin1, xmax1, ymin1, ymax1, pmin, pmid, level + 1, box,
		      mype, pass);
      CalcSubSurface (xmin2, xmax2, ymin2, ymax2, pmid + 1, pmax, level + 1,
		      box, mype, pass);
    }
  else
    {
      // cerr << "Too many PE for this problem size " << endl;
      fprintf (stderr, "Too many PE for this problem size => box[0] = -1\n");
      box[0] = -1;
      box[1] = -1;
      box[2] = -1;
      box[3] = -1;
    }
}

#ifdef TESTPGM
int
main (int argc, char **argv)
{
  int nx = 128;
  int ny = 128;
  int nbproc = 32;
  int mype = 0;
  int bord = 0;

  for (mype = 0; mype < nbproc; mype++)
    {
      int box[MAXBOX_D] = { -1, -1, -1, -1, -1, -1, -1, -1 };
      // first pass determin our box
      CalcSubSurface (0, nx - 1, 0, ny - 1, 0, nbproc - 1, 0, box, mype, 0);
      // second pass determin our neighbours
      CalcSubSurface (0, nx - 1, 0, ny - 1, 0, nbproc - 1, 0, box, mype, 1);
      printf ("[%4d] %4d %4d %4d %4d / %4d %4d %4d %4d \n", mype,
	      box[XMIN_D], box[XMAX_D], box[YMIN_D], box[YMAX_D], box[UP_D],
	      box[DOWN_D], box[LEFT_D], box[RIGHT_D]);
      if ((box[UP_D] == -1) || (box[DOWN_D] == -1) || (box[LEFT_D] == -1)
	  || (box[RIGHT_D] == -1))
	bord++;
    }
  printf ("B=%d / %d\n", bord, nbproc);
  return 0;
}
#endif
//EOF
