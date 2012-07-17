#ifndef CMPFLX_H_INCLUDED
#define CMPFLX_H_INCLUDED

#include "utils.h"

void cmpflx (const int narray,
	     const int Hnxyt,
	     const int Hnvar,
	     const double Hgamma,
	     const int slices, const int Hstep, double *qgdnv, double *flux);

#endif // CMPFLX_H_INCLUDED
