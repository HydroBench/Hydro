#ifndef CONSERVAR_H_INCLUDED
#define CONSERVAR_H_INCLUDED



void gatherConservativeVars(const int idim,
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
                            real_t uold[Hnvar * Hnxt * Hnyt], real_t u[Hnvar][Hstep][Hnxyt]);

void updateConservativeVars(const int idim,
                            const int rowcol,
                            const real_t dtdx,
                            const int Himin,
                            const int Himax,
                            const int Hjmin,
                            const int Hjmax,
                            const int Hnvar,
                            const int Hnxt,
                            const int Hnyt,
                            const int Hnxyt,
                            const int slices, const int Hstep,
                            real_t uold[Hnvar * Hnxt * Hnyt],
                            real_t u[Hnvar][Hstep][Hnxyt], real_t flux[Hnvar][Hstep][Hnxyt]
  );

#endif // CONSERVAR_H_INCLUDED
