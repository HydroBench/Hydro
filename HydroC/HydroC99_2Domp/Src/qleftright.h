#ifndef QLEFTRIGHT_H_INCLUDED
#define QLEFTRIGHT_H_INCLUDED

void
qleftright(const int idim,
	   const int Hnx,
	   const int Hny,
	   const int Hnxyt,
	   const int Hnvar,
	   const int slices, const int Hstep,
	   real_t qxm[Hnvar][Hstep][Hnxyt],
	   real_t qxp[Hnvar][Hstep][Hnxyt], real_t qleft[Hnvar][Hstep][Hnxyt],
	   real_t qright[Hnvar][Hstep][Hnxyt]);

#endif
