#ifndef TRACE_H_INCLUDED
#define TRACE_H_INCLUDED

#include "hmpp.h"



void trace (const hydro_real_t dtdx,
	    const int n,
	    const int Hscheme,
	    const int Hnvar,
	    const int Hnxyt,
	    const int slices, const int Hstep,
	    hydro_real_t *q, hydro_real_t *dq, hydro_real_t *c, hydro_real_t *qxm, hydro_real_t *qxp);

#endif // TRACE_H_INCLUDED
