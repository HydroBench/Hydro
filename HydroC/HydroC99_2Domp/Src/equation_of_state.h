#ifndef EQUATION_OF_STATE_H_INCLUDED
#define EQUATION_OF_STATE_H_INCLUDED

#include "License.h"
#include "utils.h"
#include "parametres.h"

void equation_of_state(int imin,
		       int imax,
		       const int Hnxyt,
		       const int Hnvar,
		       const real_t Hsmallc,
		       const real_t Hgamma,
		       const int slices, const int Hstep,
		       real_t eint[Hstep][Hnxyt], real_t q[Hnvar][Hstep][Hnxyt],
		       real_t c[Hstep][Hnxyt]);

#endif				// EQUATION_OF_STATE_H_INCLUDED
