#ifndef COURANTONXY_H_INCLUDED
#define COURANTONXY_H_INCLUDED

void
courantOnXY(real_t * cournox,
	    real_t * cournoy,
	    const int Hnx,
	    const int Hnxyt,
	    const int Hnvar, const int slices, const int Hstep,
	    real_t c[Hstep][Hnxyt], real_t q[Hnvar][Hstep][Hnxyt],
	    real_t * tmpm1, real_t * tmpm2);

#endif				// COURANTONXY_H_INCLUDED
