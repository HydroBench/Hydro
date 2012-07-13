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

#include <CL/cl.h>

void oclRiemann(cl_mem qleftDEV, cl_mem qrightDEV,
                cl_mem qgdnvDEV, cl_mem rlDEV,
                cl_mem ulDEV, cl_mem plDEV,
                cl_mem clDEV, cl_mem wlDEV,
                cl_mem rrDEV, cl_mem urDEV,
                cl_mem prDEV, cl_mem crDEV,
                cl_mem wrDEV, cl_mem roDEV,
                cl_mem uoDEV, cl_mem poDEV,
                cl_mem coDEV, cl_mem woDEV,
                cl_mem rstarDEV, cl_mem ustarDEV,
                cl_mem pstarDEV, cl_mem cstarDEV,
                cl_mem sgnmDEV, cl_mem spinDEV,
                cl_mem spoutDEV, cl_mem ushockDEV,
                cl_mem fracDEV, cl_mem scrDEV,
                cl_mem delpDEV, cl_mem poldDEV,
                cl_mem indDEV, cl_mem ind2DEV,
                const long narray,
                const double Hsmallr,
                const double Hsmallc, const double Hgamma, const long Hniter_riemann, const long Hnvar, const long Hnxyt);

#endif // CURIEMANN_H_INCLUDED
