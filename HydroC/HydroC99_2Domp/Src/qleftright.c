#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <stdio.h>

#include "parametres.h"
#include "utils.h"
#include "qleftright.h"

void
// qleftright(const int idim, const hydroparam_t H, hydrovarwork_t * Hvw)
qleftright(const int idim,
	   const int Hnx,
	   const int Hny,
	   const int Hnxyt,
	   const int Hnvar,
	   const int slices, const int Hstep,
	   real_t qxm[Hnvar][Hstep][Hnxyt],
	   real_t qxp[Hnvar][Hstep][Hnxyt],
	   real_t qleft[Hnvar][Hstep][Hnxyt],
	   real_t qright[Hnvar][Hstep][Hnxyt])
{
    // #define IHVW(i,v) ((i) + (v) * Hnxyt)
    int nvar, i, s;
    int bmax;
    WHERE("qleftright");
    if (idim == 1) {
	bmax = Hnx + 1;
    } else {
	bmax = Hny + 1;
    }

#ifdef TARGETON
#pragma message "TARGET on QLEFTRIGHT"
#pragma omp target				\
	map(to:qxm[0:Hnvar][0:Hstep][0:bmax])	\
	map(to:qxp[0:Hnvar][0:Hstep][0:bmax])	\
	map(from:qleft[0:Hnvar][0:Hstep][0:bmax])	\
	map(from:qright[0:Hnvar][0:Hstep][0:bmax])
#pragma omp teams distribute parallel for default(none) private(s, i, nvar), shared(qleft, qright, qxm, qxp)  collapse(3)
#else
#pragma omp parallel for private(nvar, i, s), shared(qleft, qright)
#endif
    for (s = 0; s < slices; s++) {
	for (nvar = 0; nvar < Hnvar; nvar++) {
// #pragma omp simd
	    for (i = 0; i < bmax; i++) {
		qleft[nvar][s][i] = qxm[nvar][s][i + 1];
		qright[nvar][s][i] = qxp[nvar][s][i + 2];
	    }
	}
    }
}

#undef IHVW

// EOF
