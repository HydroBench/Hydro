#ifndef UTILS_H_INCLUDED
#define UTILS_H_INCLUDED

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
double **allocate(long imin, long imax, long nvar);
double *DMalloc(long n);
long *IMalloc(long n);

// 0 means perfect memory management from the code ;-)
static const long MallocGuard = 0;
#ifdef __cplusplus
extern "C" {
#endif
void printuold(FILE * fic, const hydroparam_t H, hydrovar_t * Hv);
void printarray(FILE * fic, double *a, long n, const char *nom, const hydroparam_t H);
void printarrayi(FILE * fic, long *a, long n, const char *nom);
void printarrayv(FILE * fic, double *a, long n, const char *nom, const hydroparam_t H);
void printarrayv2(FILE * fic, double *a, long n, const char *nom, const hydroparam_t H);
void timeToString(char *buf, const double timeInS);
double cclock(void);
#ifdef __cplusplus
};
#endif

#ifndef PRINTUOLD
#ifndef HMPP
#define PRINTUOLD(f, x, y) if ((x).prt) { printuold((f), (x), (y)); }
#define PRINTARRAY(f, x, y, z, t) if ((t).prt) { printarray((f), (x), (y), (z), (t)); }
#define PRINTARRAYI(f, x, y, z, t) if ((t).prt) { printarrayi((f), (x), (y), (z)); }
#define PRINTARRAYV(f, x, y, z, t) if ((t).prt) { printarrayv((f), (x), (y), (z), (t)); }
#define PRINTARRAYV2(f, x, y, z, t) if ((t).prt) { printarrayv2((f), (x), (y), (z), (t)); }
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
#define RESTRICT
#endif /*  */

#ifdef HMPP
#undef FLOPS
#endif /*  */

#endif // UTILS_H_INCLUDED
