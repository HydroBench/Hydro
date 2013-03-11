#ifndef CMPFLX_H_INCLUDED
#define CMPFLX_H_INCLUDED

#include "utils.h"

void cmpflx (const int narray,
	     const int Hnxyt,
	     const int Hnvar,
	     const hydro_real_t Hgamma,
	     const int slices, const int Hstep, hydro_real_t *qgdnv, hydro_real_t *flux);

#endif // CMPFLX_H_INCLUDED
