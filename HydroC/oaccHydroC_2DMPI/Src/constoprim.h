#ifndef CONSTOPRIM_H_INCLUDED
#define CONSTOPRIM_H_INCLUDED

#include "utils.h"



void constoprim (const int n,
		 const int Hnxyt,
		 const int Hnvar,
		 const hydro_real_t  Hsmallr,
		 const int slices, const int Hstep,
		 hydro_real_t  *u, hydro_real_t *q, hydro_real_t *e);

#endif // CONSTOPRIM_H_INCLUDED
