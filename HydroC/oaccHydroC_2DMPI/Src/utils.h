#ifndef UTILS_H_INCLUDED
#define UTILS_H_INCLUDED

#include "parametres.h"

#ifndef Square
#define Square(x) ((x) * (x))
#endif /*  */

#ifndef MAX
#define MAX(x, y) ((x) > (y)? (x): (y))
#endif /*  */
#ifndef MIN
#define MIN(x, y) ((x) < (y)? (x): (y))
#endif /*  */

#ifndef Free
// Make sure that the pointer is unusable afterwards.
#define Free(x) do { if ((x)) { free((x)); }; (x) = NULL; } while (0)
#endif /*  */
double **allocate (int imin, int imax, int nvar);
double *DMalloc (long n);
int *IMalloc (long n);

// 0 means perfect memory management from the code ;-)
// static const int MallocGuard = 0;
#define MallocGuard 0
void printuoldf (FILE * fic, const hydroparam_t H, hydrovar_t * Hv);
void printarray (FILE * fic, double *a, int n, const char *nom,
		 const hydroparam_t H);
void printarrayi (FILE * fic, int *a, int n, const char *nom);
void printarrayv (FILE * fic, double *a, int n, const char *nom,
		  const hydroparam_t H);
void printarrayv2 (FILE * fic, double *a, int n, const char *nom,
		   const hydroparam_t H);
void timeToString (char *buf, const double timeInS);
double cclock (void);

#ifndef PRINTUOLD
#ifndef HMPP
#define PRINTUOLD(f,x,y) if ((x).prt) { printuoldf((f), (x), (y)); }
#define PRINTARRAY(f,x,y,z,t) if ((t).prt) { printarray((f), (double *) (x), (y), (z), (t)); }
#define PRINTARRAYI(f,x,y,z,t) if ((t).prt) { printarrayi((f), (x), (y), (z)); }
#define PRINTARRAYV(f,x,y,z,t) if ((t).prt) { printarrayv((f), (double *) (x), (y), (z), (t)); }
#define PRINTARRAYV2(f,x,y,z,t) if ((t).prt) { printarrayv2((f), (double *) (x), (y), (z), (t)); }
#else /*  */
// HMPP doesn't support prints : kill them
#define PRINTUOLD(x, y)
#define PRINTARRAY(x, y, z, t)
#define PRINTARRAYI(x, y, z, t)
#define PRINTARRAYV(x, y, z, t)
#endif /*  */
#endif /*  */

#ifndef WHERE
#ifndef HMPP
// #define WHERE(n) do { if (H.prt) {fprintf(stdout, "@@%s in %s\n", (n), __FILE__); }} while (0)
#define WHERE(n) do { if (0) {fprintf(stdout, "@@%s\n", (n)); }} while (0)
#else /*  */
#define WHERE(n)
#endif /*  */
#endif /*  */

#ifndef HMPP
#define RESTRICT __restrict
#else /*  */
#define RESTRICT __restrict
#endif /*  */

#ifdef HMPP
#undef FLOPS
#endif /*  */

#endif // UTILS_H_INCLUDED
