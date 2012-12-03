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

#define IHU(i, j, v)  ((i) + Hnxt * ((j) + Hnyt * (v)))
#define IHVW(i, v) ((i) + (v) * Hnxyt)

void
oclGatherConservativeVars(const long idim, const long rowcol,
                          cl_mem uold,
                          cl_mem u,
                          const long Himin,
                          const long Himax,
                          const long Hjmin,
                          const long Hjmax, const long Hnvar, const long Hnxt, const long Hnyt, const long Hnxyt)
{


  WHERE("gatherConservativeVars");
  if (idim == 1) {
    // Gather conservative variables
//         SetBlockDims((Himax - Himin), THREADSSZ, block, grid);
//         Loop1KcuGather <<< grid, block >>> (uold, u, rowcol, Hnxt, Himin, Himax, Hnyt,
//                                             Hnxyt);
    OCLINITARG;
    OCLSETARG(ker[Loop1KcuGather], uold);
    OCLSETARG(ker[Loop1KcuGather], u);
    OCLSETARG(ker[Loop1KcuGather], rowcol);
    OCLSETARG(ker[Loop1KcuGather], Hnxt);
    OCLSETARG(ker[Loop1KcuGather], Himin);
    OCLSETARG(ker[Loop1KcuGather], Himax);
    OCLSETARG(ker[Loop1KcuGather], Hnyt);
    OCLSETARG(ker[Loop1KcuGather], Hnxyt);
    oclLaunchKernel(ker[Loop1KcuGather], cqueue, (Himax - Himin), THREADSSZ, __FILE__, __LINE__);
    if (Hnvar > IP + 1) {
//             Loop3KcuGather <<< grid, block >>> (uold, u, rowcol, Hnxt, Himin, Himax, Hnyt,
//                                                 Hnxyt, Hnvar);
      OCLINITARG;
      OCLSETARG(ker[Loop3KcuGather], uold);
      OCLSETARG(ker[Loop3KcuGather], u);
      OCLSETARG(ker[Loop3KcuGather], rowcol);
      OCLSETARG(ker[Loop3KcuGather], Hnxt);
      OCLSETARG(ker[Loop3KcuGather], Himin);
      OCLSETARG(ker[Loop3KcuGather], Himax);
      OCLSETARG(ker[Loop3KcuGather], Hnyt);
      OCLSETARG(ker[Loop3KcuGather], Hnxyt);
      OCLSETARG(ker[Loop3KcuGather], Hnvar);
      oclLaunchKernel(ker[Loop3KcuGather], cqueue, (Himax - Himin), THREADSSZ, __FILE__, __LINE__);
    }
  } else {
    // Gather conservative variables
    //         SetBlockDims((Hjmax - Hjmin), THREADSSZ, block, grid);
    //         Loop2KcuGather <<< grid, block >>> (uold, u, rowcol, Hnxt, Hjmin, Hjmax, Hnyt,
    //                                             Hnxyt);
    OCLINITARG;
    OCLSETARG(ker[Loop2KcuGather], uold);
    OCLSETARG(ker[Loop2KcuGather], u);
    OCLSETARG(ker[Loop2KcuGather], rowcol);
    OCLSETARG(ker[Loop2KcuGather], Hnxt);
    OCLSETARG(ker[Loop2KcuGather], Hjmin);
    OCLSETARG(ker[Loop2KcuGather], Hjmax);
    OCLSETARG(ker[Loop2KcuGather], Hnyt);
    OCLSETARG(ker[Loop2KcuGather], Hnxyt);
    oclLaunchKernel(ker[Loop2KcuGather], cqueue, (Hjmax - Hjmin), THREADSSZ, __FILE__, __LINE__);
    if (Hnvar > IP + 1) {
      //             Loop4KcuGather <<< grid, block >>> (uold, u, rowcol, Hnxt, Hjmin, Hjmax, Hnyt,
      //                                                 Hnxyt, Hnvar);
      OCLINITARG;
      OCLSETARG(ker[Loop4KcuGather], uold);
      OCLSETARG(ker[Loop4KcuGather], u);
      OCLSETARG(ker[Loop4KcuGather], rowcol);
      OCLSETARG(ker[Loop4KcuGather], Hnxt);
      OCLSETARG(ker[Loop4KcuGather], Hjmin);
      OCLSETARG(ker[Loop4KcuGather], Hjmax);
      OCLSETARG(ker[Loop4KcuGather], Hnyt);
      OCLSETARG(ker[Loop4KcuGather], Hnxyt);
      OCLSETARG(ker[Loop4KcuGather], Hnvar);
      oclLaunchKernel(ker[Loop4KcuGather], cqueue, (Hjmax - Hjmin), THREADSSZ, __FILE__, __LINE__);
    }
  }
}


void
oclUpdateConservativeVars(const long idim, const long rowcol, const double dtdx,
                          cl_mem uold,
                          cl_mem u,
                          cl_mem flux,
                          const long Himin,
                          const long Himax,
                          const long Hjmin,
                          const long Hjmax, const long Hnvar, const long Hnxt, const long Hnyt, const long Hnxyt)
{
  WHERE("updateConservativeVars");

  if (idim == 1) {
//         SetBlockDims((Himax - ExtraLayer) - (Himin + ExtraLayer), THREADSSZ, block, grid);
//         // Update conservative variables
//         Loop1KcuUpdate <<< grid, block >>> (rowcol, dtdx, uold, u, flux, Himin, Himax,
//                                             Hnxt, Hnyt, Hnxyt);
//         CheckErr("Loop1KcuUpdate");
//         if (Hnvar > IP + 1) {
//             Loop2KcuUpdate <<< grid, block >>> (rowcol, dtdx, uold, u, flux, Himin, Himax,
//                                                 Hnvar, Hnxt, Hnyt, Hnxyt);
//             CheckErr("Loop2KcuUpdate");
//         }
    OCLINITARG;
    OCLSETARG(ker[Loop1KcuUpdate], rowcol);
    OCLSETARG(ker[Loop1KcuUpdate], dtdx);
    OCLSETARG(ker[Loop1KcuUpdate], uold);
    OCLSETARG(ker[Loop1KcuUpdate], u);
    OCLSETARG(ker[Loop1KcuUpdate], flux);
    OCLSETARG(ker[Loop1KcuUpdate], Himin);
    OCLSETARG(ker[Loop1KcuUpdate], Himax);
    OCLSETARG(ker[Loop1KcuUpdate], Hnxt);
    OCLSETARG(ker[Loop1KcuUpdate], Hnyt);
    OCLSETARG(ker[Loop1KcuUpdate], Hnxyt);
    oclLaunchKernel(ker[Loop1KcuUpdate], cqueue, (Himax - ExtraLayer) - (Himin + ExtraLayer), THREADSSZ, __FILE__, __LINE__);
    if (Hnvar > IP + 1) {
      OCLINITARG;
      OCLSETARG(ker[Loop2KcuUpdate], rowcol);
      OCLSETARG(ker[Loop2KcuUpdate], dtdx);
      OCLSETARG(ker[Loop2KcuUpdate], uold);
      OCLSETARG(ker[Loop2KcuUpdate], u);
      OCLSETARG(ker[Loop2KcuUpdate], flux);
      OCLSETARG(ker[Loop2KcuUpdate], Himin);
      OCLSETARG(ker[Loop2KcuUpdate], Himax);
      OCLSETARG(ker[Loop2KcuUpdate], Hnvar);
      OCLSETARG(ker[Loop2KcuUpdate], Hnxt);
      OCLSETARG(ker[Loop2KcuUpdate], Hnyt);
      OCLSETARG(ker[Loop2KcuUpdate], Hnxyt);
      oclLaunchKernel(ker[Loop2KcuUpdate], cqueue, (Himax - ExtraLayer) - (Himin + ExtraLayer), THREADSSZ, __FILE__, __LINE__);
    }
  } else {
//         // Update conservative variables
//         SetBlockDims((Hjmax - ExtraLayer) - (Hjmin + ExtraLayer), THREADSSZ, block, grid);
//         Loop3KcuUpdate <<< grid, block >>> (rowcol, dtdx, uold, u, flux, Hjmin, Hjmax,
//                                             Hnxt, Hnyt, Hnxyt);
//         CheckErr("Loop3KcuUpdate");
//         if (Hnvar > IP + 1) {
//             Loop4KcuUpdate <<< grid, block >>> (rowcol, dtdx, uold, u, flux, Hjmin, Hjmax,
//                                                 Hnvar, Hnxt, Hnyt, Hnxyt);
//             CheckErr("Loop4KcuUpdate");
//         }
    OCLINITARG;
    OCLSETARG(ker[Loop3KcuUpdate], rowcol);
    OCLSETARG(ker[Loop3KcuUpdate], dtdx);
    OCLSETARG(ker[Loop3KcuUpdate], uold);
    OCLSETARG(ker[Loop3KcuUpdate], u);
    OCLSETARG(ker[Loop3KcuUpdate], flux);
    OCLSETARG(ker[Loop3KcuUpdate], Hjmin);
    OCLSETARG(ker[Loop3KcuUpdate], Hjmax);
    OCLSETARG(ker[Loop3KcuUpdate], Hnxt);
    OCLSETARG(ker[Loop3KcuUpdate], Hnyt);
    OCLSETARG(ker[Loop3KcuUpdate], Hnxyt);
    oclLaunchKernel(ker[Loop3KcuUpdate], cqueue, (Hjmax - ExtraLayer) - (Hjmin + ExtraLayer), THREADSSZ, __FILE__, __LINE__);
    if (Hnvar > IP + 1) {
      OCLINITARG;
      OCLSETARG(ker[Loop4KcuUpdate], rowcol);
      OCLSETARG(ker[Loop4KcuUpdate], dtdx);
      OCLSETARG(ker[Loop4KcuUpdate], uold);
      OCLSETARG(ker[Loop4KcuUpdate], u);
      OCLSETARG(ker[Loop4KcuUpdate], flux);
      OCLSETARG(ker[Loop4KcuUpdate], Hjmin);
      OCLSETARG(ker[Loop4KcuUpdate], Hjmax);
      OCLSETARG(ker[Loop4KcuUpdate], Hnvar);
      OCLSETARG(ker[Loop4KcuUpdate], Hnxt);
      OCLSETARG(ker[Loop4KcuUpdate], Hnyt);
      OCLSETARG(ker[Loop4KcuUpdate], Hnxyt);
      oclLaunchKernel(ker[Loop4KcuUpdate], cqueue, (Hjmax - ExtraLayer) - (Hjmin + ExtraLayer), THREADSSZ, __FILE__, __LINE__);
    }
  }
}

#undef IHVW
#undef IHU

//EOF
