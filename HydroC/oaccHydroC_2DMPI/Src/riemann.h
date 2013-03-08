#ifndef RIEMANN_H_INCLUDED
#define RIEMANN_H_INCLUDED

#include "hmpp.h"


void riemann (int narray,
	      const real Hsmallr,
	      const real Hsmallc,
	      const real Hgamma,
	      const int Hniter_riemann,
	      const int Hnvar,
	      const int Hnxyt,
	      const int slices, const int Hstep,
	      real *qleft, real *qright, real *qgdnv, int *sgnm);

void Dmemset (size_t nbr, real t[nbr], real motif);

#endif // RIEMANN_H_INCLUDED
