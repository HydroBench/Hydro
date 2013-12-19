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
#include <string.h>
#include <assert.h>
#include <stdio.h>
#include <time.h>
#include <sys/types.h>
#include <sys/time.h>

#include "utils.h"
// #include "parametres.h"
real_t **
allocate(int imin, int imax, int nvar) {
  int i;

  real_t **r = (real_t **) calloc(nvar, sizeof(real_t *));
  assert(r != NULL);
  for (i = 0; i < nvar; i++) {
    r[i] = DMalloc(imax - imin + 1 + MallocGuard);
  }
  return r;
}

#ifndef __MIC__
#define NUMA_ALLOC 0
#endif

#ifdef __MIC__
#define MEMSET 1
#else
#define MEMSET 0
#endif
#if NUMA_ALLOC==1
#include <numa.h>
#endif

void DFree(real_t ** adr, size_t n)
{
#if NUMA_ALLOC == 1
  numa_free(*adr, sizeof(real_t) * (n + MallocGuard));
#else
  free(*adr); 
#endif
*adr = NULL;
}
void IFree(int ** adr, size_t n)
{
#if NUMA_ALLOC == 1
  numa_free(*adr, sizeof(int) * (n + MallocGuard));
#else
  free(*adr); 
#endif
*adr = NULL;
}

real_t *
DMalloc(size_t n) {
  size_t i;
#if NUMA_ALLOC == 1
  real_t *r = (real_t *) numa_alloc_interleaved((n + MallocGuard) * sizeof(real_t));
#else
  real_t *r = (real_t *) calloc((n + MallocGuard), sizeof(real_t));
#endif
  assert(r != NULL);
  
#if MEMSET == 1
  memset(r, 1, n * sizeof(real_t));
#else
#ifndef NOTOUCHPAGE
#pragma omp parallel for private(i) shared(r) 
  for (i = 0; i < n; i++)
    r[i] = 0.0L;
#endif
#endif
  return r;
}

int *
IMalloc(size_t n) {
  size_t i;
#if NUMA_ALLOC == 1
  int *r = (int *) numa_alloc((n + MallocGuard) * sizeof(int));
#else
  int *r = (int *) calloc((n + MallocGuard), sizeof(int));
#endif
  assert(r != NULL);

#if MEMSET == 1
  memset(r, 1, n * sizeof(int));
#else
#pragma omp parallel for private(i) shared(r) 
  for (i = 0; i < n; i++)
    r[i] = 0;
#endif
  return r;
}


#include "parametres.h"
#define VALPERLINE 16
void
printuoldf(FILE * fic, const hydroparam_t H, hydrovar_t * Hv) {
  int i, j, nvar;
  for (nvar = 0; nvar < H.nvar; nvar++) {
    fprintf(fic, "=uold %d >\n", nvar);
    for (j = 0; j < H.nyt; j++) {
      int nbr = 1;
      for (i = 0; i < H.nxt; i++) {
        fprintf(fic, "%12.4e ", Hv->uold[IHv(i, j, nvar)]);
        nbr++;
        if (nbr == VALPERLINE) {
          fprintf(fic, "\n");
          fflush(fic);
          nbr = 1;
        }
      }
      if (nbr != 1)
        fprintf(fic, "\n");
      // fprintf(fic, "%%\n");
      fflush(fic);
    }
  }
}

void
printarray(FILE * fic, real_t *a, int n, const char *nom, const hydroparam_t H) {
  real_t (*ptr)[H.nxyt] = (real_t (*)[H.nxyt]) a;
  long i, j, nbr = 1;
  fprintf(fic, "=%s >\n", nom);
  for (j = 0; j < H.nxystep; j++) {
    nbr = 1;
    for (i = 0; i < n; i++) {
      fprintf(fic, "%12.4e ", ptr[j][i]);
      nbr++;
      if (nbr == VALPERLINE) {
        fprintf(fic, "\n");
        nbr = 1;
      }
    }
    if (nbr != 1)
      fprintf(fic, "\n");
  }
  fprintf(fic, "\n");
}

void
printarrayi(FILE * fic, int *a, int n, const char *nom) {
  int i, nbr = 1;
  fprintf(fic, "=%s >\n", nom);
  for (i = 0; i < n; i++) {
    fprintf(fic, "%4d ", a[i]);
    nbr++;
    if (nbr == VALPERLINE) {
      fprintf(fic, "\n");
      nbr = 1;
    }
  }
  if (nbr != 1)
    fprintf(fic, "\n");
}

void
printarrayv(FILE * fic, real_t *a, int n, const char *nom, const hydroparam_t H) {
  int i, nbr = 1;
  int nvar;
  fprintf(fic, "=%s >\n", nom);
  real_t (*ptr)[H.nxyt] = (real_t (*)[H.nxyt]) a;
  for (nvar = 0; nvar < H.nvar; nvar++) {
    nbr = 1;
    for (i = 0; i < n; i++) {
      fprintf(fic, "%12.4e ", ptr[nvar][i]);
      nbr++;
      if (nbr == VALPERLINE) {
        fprintf(fic, "\n");
        nbr = 1;
      }
    }
    if (nbr != 1)
      fprintf(fic, "\n");
    fprintf(fic, "---\n");
  }
}

void
printarrayv2(FILE * fic, real_t *a, int n, const char *nom, const hydroparam_t H) {
  int i, j, nbr = 1;
  int nvar;
  fprintf(fic, "=%s >\n#", nom);
  real_t (*ptr)[H.nxystep][H.nxyt] = (real_t (*)[H.nxystep][H.nxyt]) a;
  for (nvar = 0; nvar < H.nvar; nvar++) {
    for (j = 0; j < H.nxystep; j++) {
      nbr = 1;
      for (i = 0; i < n; i++) {
        fprintf(fic, "%12.4le ", ptr[nvar][j][i]);
        nbr++;
        if (nbr == VALPERLINE) {
          fprintf(fic, "\n#");
          nbr = 1;
        }
      }
      if (nbr != 1)
        fprintf(fic, "@\n#");
    }
    fprintf(fic, "-J-\n#");
  }
  fprintf(fic, "---\n");
}

void
timeToString(char *buf, const double timeInS) {
  char ctenth[10];
  int hour = (int) (timeInS / 3600.0);
  int minute = (int) ((timeInS - hour * 3600) / 60.0);
  int second = (int) (timeInS - hour * 3600 - minute * 60);
  float tenth = (float) (timeInS - hour * 3600 - minute * 60 - second);
  sprintf(ctenth, "%.3f", tenth);
  sprintf(buf, "%02d:%02d:%02d%s", hour, minute, second, &ctenth[1]);
}


// double
// cclock(void) {
//   const double micro = 1.0e-06; /* Conversion constant */
//   static long start = 0L, startu;
//   struct timeval tp;            /* Structure used by gettimeofday */
//   double wall_time;             /* To hold the result */
//   if (gettimeofday(&tp, NULL) == -1)
//     wall_time = -1.0e0;

//   else if (!start) {
//     start = tp.tv_sec;
//     startu = tp.tv_usec;
//     wall_time = 0.0e0;
//   } else
//     wall_time = (double) (tp.tv_sec - start) + micro * (tp.tv_usec - startu);
//   return wall_time;
// }


//EOF
