#ifndef TRACE_H_INCLUDED
#define TRACE_H_INCLUDED

void trace(const real_t dtdx,
	   const int n,
	   const int Hscheme,
	   const int Hnvar,
	   const int Hnxyt,
	   const int slices, const int Hstep,
	   real_t q[Hnvar][Hstep][Hnxyt],
	   real_t dq[Hnvar][Hstep][Hnxyt],
	   real_t c[Hstep][Hnxyt], real_t qxm[Hnvar][Hstep][Hnxyt],
	   real_t qxp[Hnvar][Hstep][Hnxyt]
    );

#endif				// TRACE_H_INCLUDED
