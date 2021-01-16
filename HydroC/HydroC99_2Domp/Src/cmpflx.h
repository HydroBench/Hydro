#ifndef CMPFLX_H_INCLUDED
#define CMPFLX_H_INCLUDED

#include "utils.h"

void cmpflx(const int narray,
	    const int Hnxyt,
	    const int Hnvar,
	    const real_t Hgamma,
	    const int slices,
	    const int Hstep,
	    real_t qgdnv[Hnvar][Hstep][Hnxyt],
	    real_t flux[Hnvar][Hstep][Hnxyt]);

#endif				// CMPFLX_H_INCLUDED
