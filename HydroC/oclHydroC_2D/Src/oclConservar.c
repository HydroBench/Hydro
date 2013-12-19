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

#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <stdio.h>

#include "parametres.h"
#include "utils.h"
#include "oclConservar.h"
#include "oclInit.h"
#include "ocltools.h"

void
oclGatherConservativeVars(const long idim,
			  const long rowcol,
			  const long Himin,
			  const long Himax,
			  const long Hjmin,
			  const long Hjmax, 
			  const long Hnvar, 
			  const long Hnxt, 
			  const long Hnyt, 
			  const long Hnxyt,
			  const int slices, const int Hnxystep,
			  cl_mem uold,
			  cl_mem u)
{


  WHERE("gatherConservativeVars");
  if (idim == 1) {
    // Gather conservative variables
    OCLSETARG10(ker[Loop1KcuGather], uold, u, rowcol, Hnxt, Himin, Himax, Hnyt, Hnxyt, slices, Hnxystep);
    oclLaunchKernel2D(ker[Loop1KcuGather], cqueue, (Himax - Himin), slices, THREADSSZ, __FILE__, __LINE__);
    if (Hnvar > IP + 1) {
      OCLSETARG11(ker[Loop3KcuGather], uold, u, rowcol, Hnxt, Himin, Himax, Hnyt, Hnxyt, Hnvar, slices, Hnxystep);
      oclLaunchKernel(ker[Loop3KcuGather], cqueue, (Himax - Himin), THREADSSZ, __FILE__, __LINE__);
    }
  } else {
    // Gather conservative variables
    OCLSETARG10(ker[Loop2KcuGather], uold, u, rowcol, Hnxt, Hjmin, Hjmax, Hnyt, Hnxyt, slices, Hnxystep);
    oclLaunchKernel2D(ker[Loop2KcuGather], cqueue, (Hjmax - Hjmin), slices, THREADSSZ, __FILE__, __LINE__);
    if (Hnvar > IP + 1) {
      OCLSETARG11(ker[Loop4KcuGather], uold, u, rowcol, Hnxt, Hjmin, Hjmax, Hnyt, Hnxyt, Hnvar, slices, Hnxystep);
      oclLaunchKernel(ker[Loop4KcuGather], cqueue, (Hjmax - Hjmin), THREADSSZ, __FILE__, __LINE__);
    }
  }
}


void
oclUpdateConservativeVars(const long idim,
			  const long rowcol,
			  const real_t dtdx,
			  const long Himin,
			  const long Himax,
			  const long Hjmin,
			  const long Hjmax, 
			  const long Hnvar, 
			  const long Hnxt, 
			  const long Hnyt, 
			  const long Hnxyt,
			  const int slices,
			  const int Hnxystep,
			  cl_mem uold,
			  cl_mem u,
			  cl_mem flux)
{
  WHERE("updateConservativeVars");

  if (idim == 1) {
    OCLSETARG12(ker[Loop1KcuUpdate], rowcol, dtdx, uold, u, flux, Himin, Himax, Hnxt, Hnyt, Hnxyt, slices, Hnxystep);
    oclLaunchKernel2D(ker[Loop1KcuUpdate], cqueue, Hnxyt, slices, THREADSSZ, __FILE__, __LINE__);
    if (Hnvar > IP + 1) {
      OCLSETARG13(ker[Loop2KcuUpdate], rowcol, dtdx, uold, u, flux, Himin, Himax, Hnvar, Hnxt, Hnyt, Hnxyt, slices, Hnxystep);
      oclLaunchKernel(ker[Loop2KcuUpdate], cqueue, Hnxyt * slices, THREADSSZ, __FILE__, __LINE__);
    }
  } else {
    OCLSETARG12(ker[Loop3KcuUpdate], rowcol, dtdx, uold, u, flux, Hjmin, Hjmax, Hnxt, Hnyt, Hnxyt, slices, Hnxystep);
    oclLaunchKernel2D(ker[Loop3KcuUpdate], cqueue, Hnxyt, slices, THREADSSZ, __FILE__, __LINE__);
    if (Hnvar > IP + 1) {
      OCLSETARG13(ker[Loop4KcuUpdate], rowcol, dtdx, uold, u, flux, Hjmin, Hjmax, Hnvar, Hnxt, Hnyt, Hnxyt, slices, Hnxystep);
      oclLaunchKernel(ker[Loop4KcuUpdate], cqueue, Hnxyt * slices, THREADSSZ, __FILE__, __LINE__);
    }
  }
}
//EOF
