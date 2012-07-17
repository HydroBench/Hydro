#ifndef TRACE_H_INCLUDED
#define TRACE_H_INCLUDED

#include "hmpp.h"



void trace (const double dtdx,
	    const int n,
	    const int Hscheme,
	    const int Hnvar,
	    const int Hnxyt,
	    const int slices, const int Hstep,
	    double *q, double *dq, double *c, double *qxm, double *qxp);

#endif // TRACE_H_INCLUDED
