#ifndef TRACE_H_INCLUDED
#define TRACE_H_INCLUDED

#include "hmpp.h"



void trace (const real dtdx,
	    const int n,
	    const int Hscheme,
	    const int Hnvar,
	    const int Hnxyt,
	    const int slices, const int Hstep,
	    real *q, real *dq, real *c, real *qxm, real *qxp);

#endif // TRACE_H_INCLUDED
