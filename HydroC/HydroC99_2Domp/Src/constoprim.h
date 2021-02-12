#ifndef CONSTOPRIM_H_INCLUDED
#define CONSTOPRIM_H_INCLUDED

#include "utils.h"

void constoprim(const int n, const int Hnxyt, const int Hnvar, const real_t Hsmallr,
                const int slices, const int Hstep, real_t u[Hnvar][Hstep][Hnxyt],
                real_t q[Hnvar][Hstep][Hnxyt], real_t e[Hstep][Hnxyt]);

#endif // CONSTOPRIM_H_INCLUDED
