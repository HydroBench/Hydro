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

#ifndef CURIEMANN_H_INCLUDED
#define CURIEMANN_H_INCLUDED


void cuRiemann(const long narray, const double Hsmallr, const double Hsmallc, const double Hgamma, //
	       const long Hniter_riemann, const long Hnvar, const long Hnxyt, const int slices, const int Hnxystep,      //
               double *RESTRICT qleftDEV,       // [Hnvar][Hnxystep][Hnxyt]
               double *RESTRICT qrightDEV,      // [Hnvar][Hnxystep][Hnxyt]
               double *RESTRICT qgdnvDEV,       // [Hnvar][Hnxystep][Hnxyt]
               long *RESTRICT sgnmDEV  // [Hnxystep][narray]
               // temporaries
//                , double *RESTRICT rlDEV,  // [Hnxystep][narray]
//                double *RESTRICT ulDEV,  // [Hnxystep][narray] 
//                double *RESTRICT plDEV,  // [Hnxystep][narray]
//                double *RESTRICT clDEV,  // [Hnxystep][narray] 
//                double *RESTRICT wlDEV,  // [Hnxystep][narray]
//                double *RESTRICT rrDEV,  // [Hnxystep][narray] 
//                double *RESTRICT urDEV,  // [Hnxystep][narray]
//                double *RESTRICT prDEV,  // [Hnxystep][narray] 
//                double *RESTRICT crDEV,  // [Hnxystep][narray]
//                double *RESTRICT wrDEV,  // [Hnxystep][narray] 
//                double *RESTRICT roDEV,  // [Hnxystep][narray]
//                double *RESTRICT uoDEV,  // [Hnxystep][narray] 
//                double *RESTRICT poDEV,  // [Hnxystep][narray]
//                double *RESTRICT coDEV,  // [Hnxystep][narray] 
//                double *RESTRICT woDEV,  // [Hnxystep][narray]
//                double *RESTRICT rstarDEV,       // [Hnxystep][narray] 
//                double *RESTRICT ustarDEV,       // [Hnxystep][narray]
//                double *RESTRICT pstarDEV,       // [Hnxystep][narray] 
//                double *RESTRICT cstarDEV,       // [Hnxystep][narray]
//                double *RESTRICT spinDEV,        // [Hnxystep][narray]
//                double *RESTRICT spoutDEV,       // [Hnxystep][narray] 
//                double *RESTRICT ushockDEV,      // [Hnxystep][narray]
//                double *RESTRICT fracDEV,        // [Hnxystep][narray] 
//                double *RESTRICT scrDEV, // [Hnxystep][narray]
//                double *RESTRICT delpDEV,        // [Hnxystep][narray] 
//                double *RESTRICT poldDEV,        // [Hnxystep][narray]
//                long *RESTRICT indDEV,   // [Hnxystep][narray] 
//                long *RESTRICT ind2DEV   // [Hnxystep][narray]
  );

#endif // CURIEMANN_H_INCLUDED
