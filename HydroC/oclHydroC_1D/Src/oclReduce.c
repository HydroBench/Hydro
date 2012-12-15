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

#include <CL/cl.h>
#include <stdio.h>
#include <assert.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <CL/cl.h>

#include "oclReduce.h"
#include "oclInit.h"
#include "ocltools.h"

double
oclReduceMax(cl_mem array, long nb) {
  cl_event event;
  cl_int err;
  double resultat = 0, elapsk;
  cl_mem temp1DEV;
  size_t gws[3], lws[3];
  int wrksiz, wrkgrp;
  size_t lgrlocal;

  // fprintf(stdout, "Reduc: %d %d %ld\n", wrksiz, wrkgrp, nb);
  wrksiz = wrkgrp = oclGetMaxWorkSize(ker[LoopKredMaxDble], oclGetDeviceOfCQueue(cqueue));
  wrksiz = wrkgrp = 16;

  lgrlocal = wrkgrp * sizeof(double);

  temp1DEV = clCreateBuffer(ctx, CL_MEM_READ_WRITE, lgrlocal, NULL, &err);
  oclCheckErr(err, "");

  lws[0] = wrkgrp;
  gws[0] = wrkgrp;
  OCLSETARG03(ker[LoopKredMaxDble], array, nb, temp1DEV);
  oclSetArg(ker[LoopKredMaxDble], 3, lgrlocal, NULL, __FILE__, __LINE__);
  err = clEnqueueNDRangeKernel(cqueue, ker[LoopKredMaxDble], 1, NULL, gws, lws, 0, NULL, &event);
  oclCheckErr(err, "clEnqueueNDRangeKernel");
  err = clWaitForEvents(1, &event);
  oclCheckErr(err, "clWaitForEvents");
  err = clReleaseEvent(event);
  oclCheckErr(err, "clReleaseEvent");

  err = clEnqueueReadBuffer(cqueue, temp1DEV, CL_TRUE, 0, sizeof(double), &resultat, 0, NULL, NULL);
  oclCheckErr(err, "clEnqueueReadBuffer");

  OCLFREE(temp1DEV);
  return resultat;
}

//EOF
