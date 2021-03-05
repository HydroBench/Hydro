#ifndef UTILITIES_HPP
#define UTILITIES_HPP

#include "EnumDefs.hpp"
#include "precision.hpp"

#include <cstddef> // for size_t


template <typename T> inline T Square(T x) { return x * x; }

void CalcSubSurface(int xmin, int xmax, int ymin, int ymax, int pmin, int pmax, int box[MAXBOX_D],
                    int mype);

long getMemUsed(void);
void getCPUName(char cpuName[1024]);

real_t *AlignedAllocReal(size_t lg);
int *AlignedAllocInt(size_t lg);
long *AlignedAllocLong(size_t lg);

template <typename T> int AlignedAlloc(T **r, size_t lg);

#endif
