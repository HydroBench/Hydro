#ifndef RIEMANN_H_INCLUDED
#define RIEMANN_H_INCLUDED

#include "hmpp.h"


void riemann (int narray,
	      const hydro_real_t Hsmallr,
	      const hydro_real_t Hsmallc,
	      const hydro_real_t Hgamma,
	      const int Hniter_riemann,
	      const int Hnvar,
	      const int Hnxyt,
	      const int slices, const int Hstep,
	      hydro_real_t *qleft, hydro_real_t *qright, hydro_real_t *qgdnv, int *sgnm);

void Dmemset (size_t nbr, hydro_real_t t[nbr], hydro_real_t motif);

#endif // RIEMANN_H_INCLUDED
