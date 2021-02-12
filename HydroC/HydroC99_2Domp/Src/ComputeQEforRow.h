#ifndef COMPUTEQEFORROW_H_INCLUDED
#define COMPUTEQEFORROW_H_INCLUDED

#include "parametres.h"

void ComputeQEforRow(const int j, const real_t Hsmallr, const int Hnx, const int Hnxt,
                     const int Hnyt, const int Hnxyt, const int Hnvar, const int slices,
                     const int Hstep, real_t *uold, real_t q[Hnvar][Hstep][Hnxyt],
                     real_t e[Hstep][Hnxyt]);
#endif // COMPUTEQEFORROW_H_INCLUDED
