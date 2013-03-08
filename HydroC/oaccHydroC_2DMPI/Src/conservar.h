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
			     real *uold, real *u);

void updateConservativeVars (const int idim,
			     const int rowcol,
			     const real dtdx,
			     const int Himin,
			     const int Himax,
			     const int Hjmin,
			     const int Hjmax,
			     const int Hnvar,
			     const int Hnxt,
			     const int Hnyt,
			     const int Hnxyt,
			     const int slices, const int Hstep,
			     real uold[Hnvar * Hnxt * Hnyt],
			     real *u, real *flux);

#endif // CONSERVAR_H_INCLUDED
