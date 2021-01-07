#include <math.h>
#include <malloc.h>
// #include <unistd.h>
// #include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "parametres.h"
#include "utils.h"
#include "cmpflx.h"
#include "perfcnt.h"

void
cmpflx(const int narray,
       const int Hnxyt,
       const int Hnvar,
       const real_t Hgamma,
       const int slices,
       const int Hstep,
       real_t qgdnv[Hnvar][Hstep][Hnxyt], real_t flux[Hnvar][Hstep][Hnxyt])
{
    int nface, i, IN;
    real_t entho, ekin, etot;
    WHERE("cmpflx");
    int s;

    nface = narray;
    entho = one / (Hgamma - one);
    FLOPS(1, 1, 0, 0);

    // Compute fluxes

#ifdef TARGETON
#pragma message "TARGET on CMPFLX"
#pragma omp target				\
	map(qgdnv[0:Hnvar][0:Hstep][0:Hnxyt])	\
	map(flux[0:Hnvar][0:Hstep][0:Hnxyt])
#pragma omp teams distribute parallel for \
	default(none) private(s, i, ekin, etot), \
	shared(flux, qgdnv) \
	collapse(2)
#else
#pragma omp parallel for private(s, i, ekin, etot), shared(flux)
#endif
    for (s = 0; s < slices; s++) {
	for (i = 0; i < nface; i++) {
	    real_t qgdnvID = qgdnv[ID][s][i];
	    real_t qgdnvIU = qgdnv[IU][s][i];
	    real_t qgdnvIP = qgdnv[IP][s][i];
	    real_t qgdnvIV = qgdnv[IV][s][i];

	    // Mass density
	    real_t massDensity = qgdnvID * qgdnvIU;
	    flux[ID][s][i] = massDensity;

	    // Normal momentum
	    flux[IU][s][i] = massDensity * qgdnvIU + qgdnvIP;
	    // Transverse momentum 1
	    flux[IV][s][i] = massDensity * qgdnvIV;

	    // Total energy
	    ekin = half * qgdnvID * (Square(qgdnvIU) + Square(qgdnvIV));
	    etot = qgdnvIP * entho + ekin;

	    flux[IP][s][i] = qgdnvIU * (etot + qgdnvIP);
	}
    }
    {
	int nops = slices * nface;
	FLOPS(13 * nops, 0 * nops, 0 * nops, 0 * nops);
    }

    // Other advected quantities
    if (Hnvar > IP) {
	for (s = 0; s < slices; s++) {
	    for (IN = IP + 1; IN < Hnvar; IN++) {
		for (i = 0; i < nface; i++) {
		    flux[IN][s][i] = flux[IN][s][i] * qgdnv[IN][s][i];
		}
	    }
	}
    }
}				// cmpflx

//EOF
