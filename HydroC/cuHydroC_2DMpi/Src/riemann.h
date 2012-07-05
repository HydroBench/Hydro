#ifndef RIEMANN_H_INCLUDED
#define RIEMANN_H_INCLUDED

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

void riemann(const long narray, const double Hsmallr, const double Hsmallc, const double Hgamma,        //
             const long Hniter_riemann, const long Hnvar, const long Hnxyt,     //
             double *RESTRICT qleft, double *RESTRICT qright, double *RESTRICT qgdnv, long *RESTRICT sgnm,      //
             double *RESTRICT rl, double *RESTRICT ul, double *RESTRICT pl, double *RESTRICT cl, double *RESTRICT wl, double *RESTRICT rr, double *RESTRICT ur, double *RESTRICT pr, double *RESTRICT cr, double *RESTRICT wr, double *RESTRICT ro, double *RESTRICT uo, double *RESTRICT po, double *RESTRICT co, double *RESTRICT wo, double *RESTRICT rstar, double *RESTRICT ustar, double *RESTRICT pstar, double *RESTRICT cstar, double *RESTRICT spin, double *RESTRICT spout, double *RESTRICT ushock, double *RESTRICT frac, double *RESTRICT scr, double *RESTRICT delp, double *RESTRICT pold, long *RESTRICT ind, long *RESTRICT ind2      //
  );

#endif // RIEMANN_H_INCLUDED
