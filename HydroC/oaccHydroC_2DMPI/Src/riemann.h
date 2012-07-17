#ifndef RIEMANN_H_INCLUDED
#define RIEMANN_H_INCLUDED

#include "hmpp.h"


void riemann (int narray,
	      const double Hsmallr,
	      const double Hsmallc,
	      const double Hgamma,
	      const int Hniter_riemann,
	      const int Hnvar,
	      const int Hnxyt,
	      const int slices, const int Hstep,
	      double *qleft, double *qright, double *qgdnv, int *sgnm);

void Dmemset (size_t nbr, double t[nbr], double motif);

#endif // RIEMANN_H_INCLUDED
