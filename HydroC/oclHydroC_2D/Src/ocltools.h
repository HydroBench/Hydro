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
 * (C) CEA/DAM Guillaume.Colin-de-Verdiere at Cea.Fr
 */

#ifndef OCLTOOLS_H
#define OCLTOOLS_H

#include <CL/cl.h>

#include "oclerror.h"

#ifndef MIN
#define MIN(a, b) (((a) <= (b))? (a): (b))
#endif

typedef enum { NDR_1D = 1, NDR_2D = 2, NDR_3D = 3 } MkNDrange_t;
typedef size_t dim3[3];

// Here we define a set of macros to pass arguments to kernels.  The
// only limitation is that each argument MUST be a variable. No
// expression or constant is allowed by the OpenCL mechanism.

#define OCLINITARG cl_uint narg = 0
#define OCLRESETARG narg = 0
#define OCLSETARG(k, a) do { oclSetArg((k), narg, sizeof((a)), &(a), __FILE__, __LINE__); narg++; } while(0);

#define OCLSETARG01(k, a) { OCLINITARG ; OCLSETARG((k), (a)) ; }
#define OCLSETARG02(k, a, b) { OCLINITARG ; OCLSETARG((k), (a)) ; OCLSETARG((k), (b)) ; }
#define OCLSETARG03(k, a, b, c) { OCLINITARG ; OCLSETARG((k), (a)) ; OCLSETARG((k), (b)) ;  \
    OCLSETARG((k), (c)) ; }
#define OCLSETARG04(k, a, b, c, d) { OCLINITARG ; OCLSETARG((k), (a)) ; OCLSETARG((k), (b)) ;  \
    OCLSETARG((k), (c)) ; OCLSETARG((k), (d)) ; }
#define OCLSETARG05(k, a, b, c, d, e) { OCLINITARG ;			\
    OCLSETARG((k), (a)) ; OCLSETARG((k), (b)) ; OCLSETARG((k), (c)) ; OCLSETARG((k), (d)) ; \
    OCLSETARG((k), (e)) ; }
#define OCLSETARG06(k, a, b, c, d, e, f) { OCLINITARG ;			\
    OCLSETARG((k), (a)) ; OCLSETARG((k), (b)) ; OCLSETARG((k), (c)) ; OCLSETARG((k), (d)) ; \
    OCLSETARG((k), (e)) ; OCLSETARG((k), (f)) ; }
#define OCLSETARG07(k, a, b, c, d, e, f, g) { OCLINITARG ;		\
    OCLSETARG((k), (a)) ; OCLSETARG((k), (b)) ; OCLSETARG((k), (c)) ; OCLSETARG((k), (d)) ; \
    OCLSETARG((k), (e)) ; OCLSETARG((k), (f)) ; OCLSETARG((k), (g)) ; }
#define OCLSETARG08(k, a, b, c, d, e, f, g, h) { OCLINITARG ;		\
    OCLSETARG((k), (a)) ; OCLSETARG((k), (b)) ; OCLSETARG((k), (c)) ; OCLSETARG((k), (d)) ; \
    OCLSETARG((k), (e)) ; OCLSETARG((k), (f)) ; OCLSETARG((k), (g)) ; OCLSETARG((k), (h)) ; }
#define OCLSETARG09(k, a, b, c, d, e, f, g, h, i) { OCLINITARG ;	\
    OCLSETARG((k), (a)) ; OCLSETARG((k), (b)) ; OCLSETARG((k), (c)) ; OCLSETARG((k), (d)) ; \
    OCLSETARG((k), (e)) ; OCLSETARG((k), (f)) ; OCLSETARG((k), (g)) ; OCLSETARG((k), (h)) ; \
    OCLSETARG((k), (i)) ; }
#define OCLSETARG10(k, a, b, c, d, e, f, g, h, i, j) { OCLINITARG ;	\
    OCLSETARG((k), (a)) ; OCLSETARG((k), (b)) ; OCLSETARG((k), (c)) ; OCLSETARG((k), (d)) ; \
    OCLSETARG((k), (e)) ; OCLSETARG((k), (f)) ; OCLSETARG((k), (g)) ; OCLSETARG((k), (h)) ; \
    OCLSETARG((k), (i)) ; OCLSETARG((k), (j)) ; }
#define OCLSETARG11(k, a, b, c, d, e, f, g, h, i, j, l) { OCLINITARG ;	\
    OCLSETARG((k), (a)) ; OCLSETARG((k), (b)) ; OCLSETARG((k), (c)) ; OCLSETARG((k), (d)) ; \
    OCLSETARG((k), (e)) ; OCLSETARG((k), (f)) ; OCLSETARG((k), (g)) ; OCLSETARG((k), (h)) ; \
    OCLSETARG((k), (i)) ; OCLSETARG((k), (j)) ; OCLSETARG((k), (l)) ; }
#define OCLSETARG12(k, a, b, c, d, e, f, g, h, i, j, l, m) { OCLINITARG ; \
    OCLSETARG((k), (a)) ; OCLSETARG((k), (b)) ; OCLSETARG((k), (c)) ; OCLSETARG((k), (d)) ; \
    OCLSETARG((k), (e)) ; OCLSETARG((k), (f)) ; OCLSETARG((k), (g)) ; OCLSETARG((k), (h)) ; \
    OCLSETARG((k), (i)) ; OCLSETARG((k), (j)) ; OCLSETARG((k), (l)) ; OCLSETARG((k), (m)) ; }

#define OCLSETARG13(k, a, b, c, d, e, f, g, h, i, j, l, m, n) { OCLINITARG ; \
    OCLSETARG((k), (a)) ; OCLSETARG((k), (b)) ; OCLSETARG((k), (c)) ; OCLSETARG((k), (d)) ; \
    OCLSETARG((k), (e)) ; OCLSETARG((k), (f)) ; OCLSETARG((k), (g)) ; OCLSETARG((k), (h)) ; \
    OCLSETARG((k), (i)) ; OCLSETARG((k), (j)) ; OCLSETARG((k), (l)) ; OCLSETARG((k), (m)) ; \
    OCLSETARG((k), (n)) ; }
#define OCLSETARG14(k, a, b, c, d, e, f, g, h, i, j, l, m, n, o) { OCLINITARG ; \
    OCLSETARG((k), (a)) ; OCLSETARG((k), (b)) ; OCLSETARG((k), (c)) ; OCLSETARG((k), (d)) ; \
    OCLSETARG((k), (e)) ; OCLSETARG((k), (f)) ; OCLSETARG((k), (g)) ; OCLSETARG((k), (h)) ; \
    OCLSETARG((k), (i)) ; OCLSETARG((k), (j)) ; OCLSETARG((k), (l)) ; OCLSETARG((k), (m)) ; \
    OCLSETARG((k), (n)) ; OCLSETARG((k), (o)) ; }

#define OCLSETARG17(k, a, b, c, d, e, f, g, h, i, j, l, m, n, o, p, q, r) { OCLINITARG ; \
    OCLSETARG((k), (a)) ; OCLSETARG((k), (b)) ; OCLSETARG((k), (c)) ; OCLSETARG((k), (d)) ; \
    OCLSETARG((k), (e)) ; OCLSETARG((k), (f)) ; OCLSETARG((k), (g)) ; OCLSETARG((k), (h)) ; \
    OCLSETARG((k), (i)) ; OCLSETARG((k), (j)) ; OCLSETARG((k), (l)) ; OCLSETARG((k), (m)) ; \
    OCLSETARG((k), (n)) ; OCLSETARG((k), (o)) ; OCLSETARG((k), (p)) ; OCLSETARG((k), (q)) ; OCLSETARG((k), (r)) ;}
	
#define CREATEKER(pgm, k, a) do {cl_int err = 0; (k) = clCreateKernel((pgm), #a, &err); oclCheckErrF(err, #a, __FILE__, __LINE__); } while (0)
#define FREEKER(k, a) do {cl_int err = clReleaseKernel((k)); oclCheckErr(err, #a); } while (0)
#define OCLFREE(tab) do {cl_int status = 0; status = clReleaseMemObject((tab)); oclCheckErrF(status, "",  __FILE__, __LINE__); } while (0)

#ifdef __cplusplus
extern "C" {
#endif
  int oclMultiple(int N, int n);
  void oclMkNDrange(const size_t nb, const size_t nbthreads, const MkNDrange_t form, size_t gws[3], size_t lws[3]);
  double oclChronoElaps(const cl_event event);
  cl_uint oclCreateProgramString(const char *fname, char ***pgmt, cl_uint * pgml);
  cl_program oclCreatePgmFromCtx(const char *srcFile, const char *srcDir,
                                 const cl_context ctx, const int theplatform, const int thedev, const int verbose);

  int oclGetNbPlatforms(const int verbose);
  cl_context oclCreateCtxForPlatform(const int theplatform, const int verbose);
  cl_command_queue oclCreateCommandQueueForDev(const int theplatform,
                                               const int devselected, const cl_context ctx, const int profiling);
  int oclGetNumberOfDev(const int theplatform);
  int oclGetNbOfGpu(const int theplatform);
  int oclGetNbOfAcc(const int theplatform);
  int oclGetNbOfCpu(const int theplatform);
  int oclGetAccDev(const int theplatform, const int accnum);
  int oclGetGpuDev(const int theplatform, const int gpunum);
  int oclGetCpuDev(const int theplatform, const int cpunum);
  char *oclGetDevNature(const int theplatform, const int thedev);

  cl_platform_id oclGetPlatformOfCQueue(cl_command_queue q);
  cl_device_id oclGetDeviceOfCQueue(cl_command_queue q);
  cl_context oclGetContextOfCQueue(cl_command_queue q);

  int oclGetDeviceNum(int theplatform, cl_device_id dev);
  int oclGetPlatformNum(cl_platform_id plat);

  size_t oclGetMaxWorkSize(cl_kernel k, cl_device_id d);
  size_t oclGetMaxMemAllocSize(int theplatform, int thedev);
  int oclFp64Avail(int theplatform, int thedev);
  void oclSetArg(cl_kernel k, cl_uint narg, size_t l, const void *arg, const char * file, const int line);
  void oclSetArgLocal(cl_kernel k, cl_uint narg, size_t l, const char * file, const int line);
  double oclLaunchKernel  (cl_kernel k, cl_command_queue q, int nbobj, int nbthread, const char *fname, const int line);
  double oclLaunchKernel2D(cl_kernel k, cl_command_queue q, int nbobjx, int nbobjy, int nbthread, const char *fname, const int line);
  double oclLaunchKernel3D(cl_kernel k, cl_command_queue q, int nbobjx, int nbobjy, int nbobjz, int nbthread, const char *fname, const int line);
  void oclNbBlocks(cl_kernel k, cl_command_queue q, size_t nbobj, int nbthread, long *maxth, long *nbblocks);
#ifdef __cplusplus
};
#endif

#endif
