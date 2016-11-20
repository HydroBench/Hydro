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
/*
  This is the bridge between c options and opencl ones.
*/
// inline size_t
// IHS(const int i, const int s, const int Hnxyt) {
//   return (i) + Hnxyt * (s);
// }

// inline size_t
// IHVWS(const int i, const int s, const int v, const int Hnxyt, const int Hnxystep) {
//   return i + Hnxyt * s + Hnxyt * Hnxystep * v;
// }

#ifndef IHS
#define IHS(i,s)     ((i) + Hnxyt * (s))
#define IHS_(i,s,Hnxyt)     ((i) + (Hnxyt) * (s))
#endif

#ifndef IHVWS
#define IHVWS(i,j,v) ( (i) + Hnxyt * (j) + Hnxyt * Hnxystep * (v) )
#define IHVWS_(i,j,v,Hnxyt,Hnxystep) ( (i) + (Hnxyt) * (j) + (Hnxyt) * (Hnxystep) * (v) )
#endif

// Warning the typedef AND the define must be active to allow for
// checks in the hydro_hernels.cl file

#define SELECT_DOUBLE 0   // 0 = SP, 1 = DP
#if SELECT_DOUBLE == 0
typedef float real_t;
#define SIMPLE_PRECISION_VERSION 1
#else
typedef double real_t;
#define DOUBLE_PRECISION_VERSION 1
#endif
#undef SELECT_DOUBLE
// end of warning
