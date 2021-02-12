#ifndef UTILS_H_INCLUDED
#define UTILS_H_INCLUDED

#include "parametres.h"

#ifndef Square
#define Square(x) ((x) * (x))
#endif /*  */

#ifndef MAX
#define MAX(x, y) ((x) > (y) ? (x) : (y))
#endif /*  */
#ifndef MIN
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#endif /*  */

real_t **allocate(int imin, int imax, int nvar);
real_t *DMalloc(size_t n);
int *IMalloc(size_t n);
void DFree(real_t **adr, size_t n);
void IFree(int **adr, size_t n);

// 0 means perfect memory management from the code ;-)
// static const int MallocGuard = 0;
#define MallocGuard 0
void printuoldf(FILE *fic, const hydroparam_t H, hydrovar_t *Hv);
void printarray(FILE *fic, real_t *a, int n, const char *nom, const hydroparam_t H);
void printarrayi(FILE *fic, int *a, int n, const char *nom);
void printarrayv(FILE *fic, real_t *a, int n, const char *nom, const hydroparam_t H);
void printarrayv2(FILE *fic, real_t *a, int n, const char *nom, const hydroparam_t H);
void timeToString(char *buf, const double timeInS);

#ifndef PRINTUOLD
#define PRINTUOLD(f, x, y)                                                                         \
    if ((x).prt) {                                                                                 \
        printuoldf((f), (x), (y));                                                                 \
    }
#define PRINTARRAY(f, x, y, z, t)                                                                  \
    if ((t).prt) {                                                                                 \
        printarray((f), (real_t *)(x), (y), (z), (t));                                             \
    }
#define PRINTARRAYI(f, x, y, z, t)                                                                 \
    if ((t).prt) {                                                                                 \
        printarrayi((f), (x), (y), (z));                                                           \
    }
#define PRINTARRAYV(f, x, y, z, t)                                                                 \
    if ((t).prt) {                                                                                 \
        printarrayv((f), (real_t *)(x), (y), (z), (t));                                            \
    }
#define PRINTARRAYV2(f, x, y, z, t)                                                                \
    if ((t).prt) {                                                                                 \
        printarrayv2((f), (real_t *)(x), (y), (z), (t));                                           \
    }
#endif /*  */

#ifndef WHERE
// #define WHERE(n) do { if (H.prt) {fprintf(stdout, "@@%s in %s\n", (n), __FILE__); }} while (0)
#define WHERE(n)                                                                                   \
    do {                                                                                           \
        if (0) {                                                                                   \
            fprintf(stdout, "@@%s\n", (n));                                                        \
        }                                                                                          \
    } while (0)
#endif /*  */

#define RESTRICT __restrict

// #ifndef __MIC__
// #pragma message "collapse activated (2)"
// #define COLLAPSE collapse(2)
// #else
// #pragma message "collapse deactivated on MIC"
#define COLLAPSE
// #endif

#endif // UTILS_H_INCLUDED
