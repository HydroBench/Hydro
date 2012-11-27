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
#include "oclConstoprim.h"
#include "oclInit.h"
#include "ocltools.h"

#define IHVW(i,v) ((i) + (v) * Hnxyt)

void
oclConstoprim(cl_mem u, cl_mem q, cl_mem e, const long n, const long Hnxyt, const long Hnvar, const double Hsmallr)
{

  WHERE("constoprim");
//   SetBlockDims(n, THREADSSZ, block, grid);
//   Loop1KcuConstoprim <<< grid, block >>> (n, u, q, e, Hnxyt, Hsmallr);
//   CheckErr("Loop1KcuConstoprim");
//   if (Hnvar > IP + 1) {
//     Loop2KcuConstoprim <<< grid, block >>> (n, u, q, Hnxyt, Hnvar);
//     CheckErr("Loop2KcuConstoprim");
//   }
//   cudaThreadSynchronize();
//   CheckErr("After synchronize cuConstoprim");

  OCLINITARG;
  OCLSETARG(ker[Loop1KcuConstoprim], n);
  OCLSETARG(ker[Loop1KcuConstoprim], u);
  OCLSETARG(ker[Loop1KcuConstoprim], q);
  OCLSETARG(ker[Loop1KcuConstoprim], e);
  OCLSETARG(ker[Loop1KcuConstoprim], Hnxyt);
  OCLSETARG(ker[Loop1KcuConstoprim], Hsmallr);
  oclLaunchKernel(ker[Loop1KcuConstoprim], cqueue, n, THREADSSZ, __FILE__, __LINE__);
  if (Hnvar > IP + 1) {
    OCLINITARG;
    OCLSETARG(ker[Loop2KcuConstoprim], n);
    OCLSETARG(ker[Loop2KcuConstoprim], u);
    OCLSETARG(ker[Loop2KcuConstoprim], q);
    OCLSETARG(ker[Loop2KcuConstoprim], Hnxyt);
    OCLSETARG(ker[Loop2KcuConstoprim], Hnvar);
    oclLaunchKernel(ker[Loop2KcuConstoprim], cqueue, n, THREADSSZ, __FILE__, __LINE__);
  }
}                               // constoprim


#undef IHVW
//EOF
