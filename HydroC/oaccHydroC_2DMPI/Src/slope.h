#ifndef SLOPE_H_INCLUDED
#define SLOPE_H_INCLUDED

#include "hmpp.h"

/*
void slope (const int n,
	    const int Hnvar,
	    const int Hnxyt,
	    const double Hslope_type,
	    const int slices, const int Hstep, double *q, double *dq);
*/

void
slope (const int n,
       const int Hnvar,
       const int Hnxyt,
       const double Hslope_type,
       const int slices, const int Hstep, double *q, double *dq);
       //const int slices, const int Hstep, double q[Hnvar][Hstep][Hnxyt], double dq[Hnvar][Hstep][Hnxyt]);


#endif // SLOPE_H_INCLUDED
