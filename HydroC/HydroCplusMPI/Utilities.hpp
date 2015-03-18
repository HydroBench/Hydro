#ifndef UTILITIES_HPP
#define UTILITIES_HPP

#include "precision.hpp"
#include "EnumDefs.hpp"

template < typename T > static inline void Swap(T & x, T & y)
{
	T t = x;
	x = y;
	y = t;
}

// static
// inline real_t
// Square(real_t x) {
//   return x * x;
// }
#define Square(x) ((x) * (x))

// template < typename T > static
// inline T
// Max(T x, T y) {
//   T r = (x > y) ? x : y;
//   return r;
// }
#define Max(x, y) (((x) > (y)) ? (x) : (y))

// template < typename T > static
// inline T
// Min(T x, T y) {
//   T r = (x < y) ? x : y;
//   return r;
// }
#define Min(x, y) (((x) < (y)) ? (x) : (y))

// template < typename T > static
// inline T
// Fabs(T x) {
//   T r = (x > 0) ? x : -x;
//   return r;
// }
#define Fabs(x) (((x) > 0) ? (x) : -(x))

void CalcSubSurface(int xmin, int xmax, int ymin, int ymax, int pmin, int pmax, int box[MAXBOX_D], int mype);

long getMemUsed(void);
void getCPUName(char cpuName[1024]);

real_t *AlignedAllocReal(size_t lg);
int *AlignedAllocInt(size_t lg);
long *AlignedAllocLong(size_t lg);

template < typename T > int AlignedAlloc(T ** r, size_t lg);

#endif
