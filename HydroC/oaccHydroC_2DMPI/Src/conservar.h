#ifndef CONSERVAR_H_INCLUDED
#define CONSERVAR_H_INCLUDED



void gatherConservativeVars (const int idim,
			     const int rowcol,
			     const int Himin,
			     const int Himax,
			     const int Hjmin,
			     const int Hjmax,
			     const int Hnvar,
			     const int Hnxt,
			     const int Hnyt,
			     const int Hnxyt,
			     const int slices, const int Hstep,
			     hydro_real_t *uold, hydro_real_t *u);

void updateConservativeVars (const int idim,
			     const int rowcol,
			     const hydro_real_t dtdx,
			     const int Himin,
			     const int Himax,
			     const int Hjmin,
			     const int Hjmax,
			     const int Hnvar,
			     const int Hnxt,
			     const int Hnyt,
			     const int Hnxyt,
			     const int slices, const int Hstep,
			     hydro_real_t uold[Hnvar * Hnxt * Hnyt],
			     hydro_real_t *u, hydro_real_t *flux);

#endif // CONSERVAR_H_INCLUDED
