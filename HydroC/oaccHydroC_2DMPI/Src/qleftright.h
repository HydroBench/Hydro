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
	    double *qxm, double *qxp, double *qleft, double *qright);

#endif
