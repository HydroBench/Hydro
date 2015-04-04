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

#ifndef OCLINIT_H
#define OCLINIT_H
#include <CL/cl.h>

#define THREADSSZ 2048

// This enum list all the possible kernels that are created in one shot.

typedef enum {
  Loop1KcuCmpflx = 1,
  Loop2KcuCmpflx,
  LoopKQEforRow,
  LoopKcourant,
  Loop1KcuGather,
  Loop2KcuGather,
  Loop3KcuGather,
  Loop4KcuGather,
  Loop1KcuUpdate,
  Loop2KcuUpdate,
  Loop3KcuUpdate,
  Loop4KcuUpdate,
  Loop1KcuConstoprim,
  Loop2KcuConstoprim,
  LoopEOS,
  Loop1KcuMakeBoundary,
  Loop2KcuMakeBoundary,
  Loop1KcuQleftright,
  Loop1KcuRiemann,
  Loop10KcuRiemann,
  LoopKcuSlope,
  Loop1KcuTrace,
  Loop2KcuTrace,
  LoopKredMaxReal,
  KernelMemset,
  KernelMemsetV4,
  kpack_arrayv, kunpack_arrayv, kpack_arrayh, kunpack_arrayh, 
  LoopKComputeDeltat,
  LastEntryKernel
} myKernel_t;

typedef enum {
  RUN_NOTDEF,
  RUN_CPU,
  RUN_GPU,
  RUN_ACC
} OclUnit_t;

// Those global variables are ugly but that's the fastest way to do it.
extern cl_command_queue cqueue;
extern cl_context ctx;
extern cl_program pgm;
extern int devselected;
extern int platformselected;
extern OclUnit_t runUnit;

extern cl_kernel *ker;
void oclMemset(cl_mem a, cl_int v, size_t lbyte);
void oclMakeHydroKernels();

void oclInitCode(const int nproc, const int mype);
void oclCloseupCode();

#endif // OCLINIT_H
//EOF
