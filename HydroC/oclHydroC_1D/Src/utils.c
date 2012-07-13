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

#include <unistd.h>
#include <stdlib.h>
#include <assert.h>
#include <stdio.h>
#include <time.h>
#include <sys/types.h>
#include <sys/time.h>

#include "utils.h"
// #include "parametres.h"
double **
allocate(long imin, long imax, long nvar)
{
  long i;

#ifdef FAST
  double **r = (double **) malloc(nvar * sizeof(double *));

#else /*  */
  double **r = (double **) calloc(nvar, sizeof(double *));

#endif /*  */
  assert(r != NULL);
  for (i = 0; i < nvar; i++) {
    r[i] = DMalloc(imax - imin + 1 + MallocGuard);
  }
  return r;
}

double *
DMalloc(long n)
{

#ifdef FAST
  double *r = (double *) malloc((n + MallocGuard) * sizeof(double));

#else /*  */
  double *r = (double *) calloc((n + MallocGuard), sizeof(double));

#endif /*  */
  assert(r != NULL);
  return r;
}

long *
IMalloc(long n)
{

#ifdef FAST
  long *r = (long *) malloc((n + MallocGuard) * sizeof(long));

#else /*  */
  long *r = (long *) calloc((n + MallocGuard), sizeof(long));

#endif /*  */
  assert(r != NULL);
  return r;
}


#include "parametres.h"
#define VALPERLINE 11
void
printuold(const hydroparam_t H, hydrovar_t * Hv)
{
  long i, j, nvar;
  for (nvar = 0; nvar < H.nvar; nvar++) {
    fprintf(stdout, "=uold %ld >\n", nvar);
    for (j = 0; j < H.nyt; j++) {
      long nbr = 1;
      for (i = 0; i < H.nxt; i++) {
        fprintf(stdout, "%13.6e ", Hv->uold[IHv(i, j, nvar)]);
        nbr++;
        if (nbr == VALPERLINE) {
          fprintf(stdout, "\n");
          nbr = 1;
        }
      }
      if (nbr != 1)
        fprintf(stdout, "\n");
      fprintf(stdout, "%%\n");
    }
  }
}
void
printarray(double *a, long n, const char *nom)
{
  long i, nbr = 1;
  fprintf(stdout, "=%s >\n", nom);
  for (i = 0; i < n; i++) {
    fprintf(stdout, "%13.6e ", a[i]);
    nbr++;
    if (nbr == VALPERLINE) {
      fprintf(stdout, "\n");
      nbr = 1;
    }
  }
  if (nbr != 1)
    fprintf(stdout, "\n");
}

void
printarrayi(long *a, long n, const char *nom)
{
  long i, nbr = 1;
  fprintf(stdout, "=%s >\n", nom);
  for (i = 0; i < n; i++) {
    fprintf(stdout, "%4ld ", a[i]);
    nbr++;
    if (nbr == VALPERLINE) {
      fprintf(stdout, "\n");
      nbr = 1;
    }
  }
  if (nbr != 1)
    fprintf(stdout, "\n");
}

void
printarrayv(double *a, long n, const char *nom, const hydroparam_t H)
{
  long i, nbr = 1;
  long nvar;
  fprintf(stdout, "=%s >\n", nom);
  for (nvar = 0; nvar < H.nvar; nvar++) {
    nbr = 1;
    for (i = 0; i < n; i++) {
      fprintf(stdout, "%13.6e ", a[IHvw(i, nvar)]);
      nbr++;
      if (nbr == VALPERLINE) {
        fprintf(stdout, "\n");
        nbr = 1;
      }
    }
    if (nbr != 1)
      fprintf(stdout, "\n");
    fprintf(stdout, "---\n");
  }
}
void
timeToString(char *buf, const double timeInS)
{
  char ctenth[10];
  long hour = timeInS / 3600;
  long minute = (timeInS - hour * 3600) / 60;
  long second = timeInS - hour * 3600 - minute * 60;
  float tenth = timeInS - hour * 3600 - minute * 60 - second;
  sprintf(ctenth, "%.3f", tenth);
  sprintf(buf, "%02ld:%02ld:%02ld%s", hour, minute, second, &ctenth[1]);
} double
cclock(void)
{
  const double micro = 1.0e-06; /* Conversion constant */
  static long start = 0L, startu;
  struct timeval tp;            /* Structure used by gettimeofday */
  double wall_time;             /* To hold the result */
  if (gettimeofday(&tp, NULL) == -1)
    wall_time = -1.0e0;

  else if (!start) {
    start = tp.tv_sec;
    startu = tp.tv_usec;
    wall_time = 0.0e0;
  } else
    wall_time = (double) (tp.tv_sec - start) + micro * (tp.tv_usec - startu);
  return wall_time;
}


//EOF
