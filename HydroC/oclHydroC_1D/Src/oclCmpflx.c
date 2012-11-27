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

#include <math.h>
#include <malloc.h>
// #include <unistd.h>
// #include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "parametres.h"
#include "utils.h"
#include "oclCmpflx.h"

#include "oclInit.h"
#include "ocltools.h"

#define IHVW(i,v) ((i) + (v) * Hnxyt)

void
oclCmpflx(cl_mem qgdnv, cl_mem flux, const long narray, const long Hnxyt, const long Hnvar, const double Hgamma)
{
  cl_int err = 0;
  dim3 gws, lws;
  cl_event event;
  double elapsk;

  WHERE("cmpflx");

  // SetBlockDims(narray, THREADSSZ, block, grid);
  oclMkNDrange(narray, THREADSSZ, NDR_1D, gws, lws);

  // Compute fluxes
  // Loop1KcuCmpflx <<< grid, block >>> (qgdnv, flux, narray, Hnxyt, Hgamma);
  oclSetArg(ker[Loop1KcuCmpflx], 0, sizeof(cl_mem), &qgdnv, __FILE__, __LINE__);
  oclSetArg(ker[Loop1KcuCmpflx], 1, sizeof(cl_mem), &flux, __FILE__, __LINE__);
  oclSetArg(ker[Loop1KcuCmpflx], 2, sizeof(narray), &narray, __FILE__, __LINE__);
  oclSetArg(ker[Loop1KcuCmpflx], 3, sizeof(Hnxyt), &Hnxyt, __FILE__, __LINE__);
  oclSetArg(ker[Loop1KcuCmpflx], 4, sizeof(Hgamma), &Hgamma, __FILE__, __LINE__);

  err = clEnqueueNDRangeKernel(cqueue, ker[Loop1KcuCmpflx], 1, NULL, gws, lws, 0, NULL, &event);
  oclCheckErr(err, "clEnqueueNDRangeKernel Loop1KcuCmpflx");
  err = clWaitForEvents(1, &event);
  oclCheckErr(err, "clWaitForEvents");
  elapsk = oclChronoElaps(event);
  err = clReleaseEvent(event);
  oclCheckErr(err, "clReleaseEvent");

  // Other advected quantities
  if (Hnvar > IP + 1) {
    // Loop2KcuCmpflx <<< grid, block >>> (qgdnv, flux, narray, Hnxyt, Hnvar);
    oclSetArg(ker[Loop2KcuCmpflx], 0, sizeof(cl_mem), &qgdnv, __FILE__, __LINE__);
    oclSetArg(ker[Loop2KcuCmpflx], 1, sizeof(cl_mem), &flux, __FILE__, __LINE__);
    oclSetArg(ker[Loop2KcuCmpflx], 2, sizeof(narray), &narray, __FILE__, __LINE__);
    oclSetArg(ker[Loop2KcuCmpflx], 3, sizeof(Hnxyt), &Hnxyt, __FILE__, __LINE__);
    oclSetArg(ker[Loop2KcuCmpflx], 4, sizeof(Hnvar), &Hnvar, __FILE__, __LINE__);
    err = clEnqueueNDRangeKernel(cqueue, ker[Loop2KcuCmpflx], 1, NULL, gws, lws, 0, NULL, &event);
    oclCheckErr(err, "clEnqueueNDRangeKernel Loop1KcuCmpflx");
    err = clWaitForEvents(1, &event);
    oclCheckErr(err, "clWaitForEvents");
    elapsk = oclChronoElaps(event);
    err = clReleaseEvent(event);
    oclCheckErr(err, "clReleaseEvent");
  }
}                               // cmpflx


#undef IHVW

//EOF
