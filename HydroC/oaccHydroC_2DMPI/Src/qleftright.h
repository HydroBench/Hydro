#ifndef QLEFTRIGHT_H_INCLUDED
#define QLEFTRIGHT_H_INCLUDED

#include "hmpp.h"


void
qleftright (const int idim,
	    const int Hnx,
	    const int Hny,
	    const int Hnxyt,
	    const int Hnvar,
	    const int slices, const int Hstep,
	    hydro_real_t *qxm, hydro_real_t *qxp, hydro_real_t *qleft, hydro_real_t *qright);

#endif
