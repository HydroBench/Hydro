//
// (C) Guillaume.Colin-de-Verdiere at CEA.Fr
//
#ifndef MPIBOUNDS_H
#define MPIBOUNDS_H
//
#include "parametres.h"

void mpileftright(int idim, const hydroparam_t H, hydrovar_t * Hv,
		  real_t * sendbufru, real_t * sendbufld, real_t * recvbufru,
		  real_t * recvbufld);
void mpiupdown(int idim, const hydroparam_t H, hydrovar_t * Hv,
	       real_t * sendbufru, real_t * sendbufld, real_t * recvbufru,
	       real_t * recvbufld);
//
#endif
//EOF
