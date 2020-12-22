/*
  A simple 2D hydro code
  (C) Romain Teyssier : CEA/IRFU           -- original F90 code
  (C) Pierre-Francois Lavallee : IDRIS      -- original F90 code
  (C) Guillaume Colin de Verdiere : CEA/DAM -- for the C version
*/

/*

This software is governed by the CeCILL license under French law and
abiding by the rules of distribution of free software.  You can  use, 
modify and/ or redistribute the software under the terms of the CeCILL
license as circulated by CEA, CNRS and INRIA at the following URL
"http://www.cecill.info". 

As a counterpart to the access to the source code and  rights to copy,
modify and redistribute granted by the license, users are provided only
with a limited warranty  and the software's author,  the holder of the
economic rights,  and the successive licensors  have only  limited
liability. 

In this respect, the user's attention is drawn to the risks associated
with loading,  using,  modifying and/or developing or reproducing the
software by the user in light of its specific status of free software,
that may mean  that it is complicated to manipulate,  and  that  also
therefore means  that it is reserved for developers  and  experienced
professionals having in-depth computer knowledge. Users are therefore
encouraged to load and test the software's suitability as regards their
requirements in conditions enabling the security of their systems and/or 
data to be ensured and,  more generally, to use and operate it in the 
same conditions as regards security. 

The fact that you are presently reading this means that you have had
knowledge of the CeCILL license and that you accept its terms.

*/
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
