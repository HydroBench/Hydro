#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#include "utils.h"
// #include "parametres.h"
real_t **allocate(int imin, int imax, int nvar) {
    int i;

    real_t **r = (real_t **)calloc(nvar, sizeof(real_t *));
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
#if NUMA_ALLOC == 1
#include <numa.h>
#endif

void DFree(real_t **adr, size_t n) {
#if NUMA_ALLOC == 1
    numa_free(*adr, sizeof(real_t) * (n + MallocGuard));
#else
    free(*adr);
#endif
    *adr = NULL;
}

void IFree(int **adr, size_t n) {
#if NUMA_ALLOC == 1
    numa_free(*adr, sizeof(int) * (n + MallocGuard));
#else
    free(*adr);
#endif
    *adr = NULL;
}

real_t *DMalloc(size_t n) {
    size_t i;
#if NUMA_ALLOC == 1
    real_t *r = (real_t *)numa_alloc_interleaved((n + MallocGuard) * sizeof(real_t));
#else
    real_t *r = (real_t *)calloc((n + MallocGuard), sizeof(real_t));
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

int *IMalloc(size_t n) {
    size_t i;
#if NUMA_ALLOC == 1
    int *r = (int *)numa_alloc((n + MallocGuard) * sizeof(int));
#else
    int *r = (int *)calloc((n + MallocGuard), sizeof(int));
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
void printuoldf(FILE *fic, const hydroparam_t H, hydrovar_t *Hv) {
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

void printarray(FILE *fic, real_t *a, int n, const char *nom, const hydroparam_t H) {
    real_t(*ptr)[H.nxyt] = (real_t(*)[H.nxyt])a;
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

void printarrayi(FILE *fic, int *a, int n, const char *nom) {
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

void printarrayv(FILE *fic, real_t *a, int n, const char *nom, const hydroparam_t H) {
    int i, nbr = 1;
    int nvar;
    fprintf(fic, "=%s >\n", nom);
    real_t(*ptr)[H.nxyt] = (real_t(*)[H.nxyt])a;
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

void printarrayv2(FILE *fic, real_t *a, int n, const char *nom, const hydroparam_t H) {
    int i, j, nbr = 1;
    int nvar;
    fprintf(fic, "=%s >\n#", nom);
    real_t(*ptr)[H.nxystep][H.nxyt] = (real_t(*)[H.nxystep][H.nxyt])a;
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

void timeToString(char *buf, const double timeInS) {
    char ctenth[10];
    int hour = (int)(timeInS / 3600.0);
    int minute = (int)((timeInS - hour * 3600) / 60.0);
    int second = (int)(timeInS - hour * 3600 - minute * 60);
    float tenth = (float)(timeInS - hour * 3600 - minute * 60 - second);
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

// EOF
