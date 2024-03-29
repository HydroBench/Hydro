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
#include <stdio.h>
#include <string.h>
#include <strings.h>
#include <unistd.h>
#include <stdlib.h>
#include <malloc.h>
#include <assert.h>
#include <limits.h>
# include <sys/time.h>
# include <float.h>
# include <math.h>
# include <hip/hip_runtime.h>

//
#include "perfcnt.h"
#include "gridfuncs.h"

// - - - Performance counting
long flopsAri = 0;
long flopsSqr = 0;
long flopsMin = 0;
long flopsTra = 0;

double MflopsSUM = 0;
long nbFLOPS = 0;

int perfInitDone = 0;
int * flops_dev = 0;

#define VERIF(x, ou) if ((x) != hipSuccess)  { CheckErr((ou)); }
int *flop_dev = NULL;
void cuPerfInit(void)
{
  hipError_t status;
  if (perfInitDone == 0) {
    status = hipMalloc((void **) &flops_dev, 4 * sizeof(int)); VERIF(status, "cuPerfInit");
    perfInitDone = 1;
  }
  status = hipMemset(flops_dev, 0, 4 * sizeof(int)); VERIF(status, "cuPerfInit");
}

void cuPerfGet(void)
{
  int flops_tmp[4];
  hipMemcpy(flops_tmp, flops_dev, 4 * sizeof(int), hipMemcpyDeviceToHost);
  FLOPS(flops_tmp[0], flops_tmp[1], flops_tmp[2], flops_tmp[3]);
  // printf("flopsAri_t=%d, flopsSqr_t=%d, flopsMin_t=%d, flopsTra_t=%d\n", flops_tmp[0], flops_tmp[1], flops_tmp[2], flops_tmp[3]);
}

//EOF
