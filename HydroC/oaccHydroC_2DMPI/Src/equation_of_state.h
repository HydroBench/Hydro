#ifndef EQUATION_OF_STATE_H_INCLUDED
#define EQUATION_OF_STATE_H_INCLUDED

#include "utils.h"
#include "parametres.h"

#ifdef HMPP

#endif

void equation_of_state (int imin,
			int imax,
			const int Hnxyt,
			const int Hnvar,
			const hydro_real_t Hsmallc,
			const hydro_real_t Hgamma,
			const int slices, const int Hstep,
			hydro_real_t *eint, hydro_real_t *q, hydro_real_t *c);

#endif // EQUATION_OF_STATE_H_INCLUDED
