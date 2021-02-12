#ifndef RIEMANN_H_INCLUDED
#define RIEMANN_H_INCLUDED

void riemann(int narray, const real_t Hsmallr, const real_t Hsmallc, const real_t Hgamma,
             const int Hniter_riemann, const int Hnvar, const int Hnxyt, const int slices,
             const int Hstep, real_t qleft[Hnvar][Hstep][Hnxyt],
             real_t qright[Hnvar][Hstep][Hnxyt], //
             real_t qgdnv[Hnvar][Hstep][Hnxyt],  //
             int sgnm[Hstep][Hnxyt], hydrowork_t *Hw);

void Dmemset(size_t nbr, real_t t[nbr], real_t motif);

#endif // RIEMANN_H_INCLUDED
