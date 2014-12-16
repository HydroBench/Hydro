/*
 * (C) CEA/DAM Guillaume.Colin-de-Verdiere at Cea.Fr
 *     Alain Cady
 */
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
#ifndef _WIN32
#include <unistd.h>
#endif
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <math.h>

#include "ocltools.h"
#include "oclerror.h"

// fix : no cbrt in VC
#ifdef _MSC_VER
#include <float.h>

double
cbrt(double x)
{

  if (fabs(x) < DBL_EPSILON)
    return 0.0;

  if (x > 0.0)
    return pow(x, 1.0 / 3.0);

  return -pow(-x, 1.0 / 3.0);

}
#endif

typedef struct _DeviceDesc {
  int    maxcu;                 // CL_DEVICE_MAX_COMPUTE_UNITS
  size_t mwid;                  // CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS
  size_t mwis[3];               // CL_DEVICE_MAX_WORK_ITEM_SIZES
  size_t mwgs;                  // CL_DEVICE_MAX_WORK_GROUP_SIZE
  size_t mmas;                  // CL_DEVICE_MAX_MEM_ALLOC_SIZE
  size_t dgms;                  // CL_DEVICE_GLOBAL_MEM_SIZE
  int vecchar;                  // CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR
  int vecshort;                 // CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT
  int vecint;                   // CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT
  int veclong;                  // CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG
  int vecfloat;                 // CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT
  int vecdouble;                // CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE
  int fpdp;                     // Has Double Float support 
  char *nature;                 // CPU GPU ACC
} DeviceDesc_t;

typedef struct _PlatformDesc {
  cl_device_id *devices;
  DeviceDesc_t *devdesc;
  cl_uint nbdevices;
} PlatformDesc_t;

static cl_uint nbplatforms = 0;
static PlatformDesc_t *pdesc = NULL;
static cl_platform_id *platform = NULL;
static int _profiling = 0;

int oclMultiple(int N, int n)
{
  int count = (N + (n - 1)) / n;
  return n * count;
}

int oclGetDeviceNum(int theplatform, cl_device_id dev){
  int i;
  for (i = 0; i < pdesc[theplatform].nbdevices; i++) {
    if (dev == pdesc[theplatform].devices[i]) return i;
  }
  return -1;
}

int oclGetPlatformNum(cl_platform_id plat){
  int i;
  for (i = 0; i < nbplatforms; i++) {
    if (plat == platform[i]) return i;
  }
  return -1;
}


void
oclMkNDrange(const size_t nb, const size_t nbthreads, const MkNDrange_t form, size_t gws[3], size_t lws[3])
{
  size_t sizec;
  long leftover;
  gws[0] = 1;
  gws[1] = 1;
  gws[2] = 1;
  lws[0] = 1;
  lws[1] = 1;
  lws[2] = 1;

  if (form == NDR_1D) {
    sizec = nb;
    lws[0] = ((nbthreads + 1) % 32) * 32;
    gws[0] = (sizec + lws[0] - 1) / lws[0];
    gws[0] *= lws[0];
    // fprintf(stderr, "gws[0] %ld lws[0] %ld\n", gws[0], lws[0]);
  }

  if (form == NDR_2D) {
    sizec = (size_t) sqrt((double) nb);
    if ((sizec * sizec) < nb)
      sizec++;
    while ((lws[0] * lws[1]) < nbthreads) {
      if ((lws[0] * lws[1]) < nbthreads)
        lws[0] *= 2;
      if ((lws[0] * lws[1]) < nbthreads)
        lws[1] *= 2;
    }
    // 
    if ((lws[0] * lws[1]) > nbthreads)
      lws[1]--;
    // normalisation of dimensions to please OpenCL
    gws[0] = (sizec + lws[0] - 1) / lws[0];
    gws[0] *= lws[0];
    gws[1] = (sizec + lws[1] - 1) / lws[1];
    gws[1] *= lws[1];
  }

  if (form == NDR_3D) {
    sizec = (size_t) cbrt((double) nb);
    if ((sizec * sizec * sizec) < nb)
      sizec++;
    while ((lws[0] * lws[1] * lws[2]) < nbthreads) {
      if ((lws[0] * lws[1] * lws[2]) < nbthreads)
        lws[0] *= 2;
      if ((lws[0] * lws[1] * lws[2]) < nbthreads)
        lws[1] *= 2;
      if ((lws[0] * lws[1] * lws[2]) < nbthreads)
        lws[0] *= 2;
      if ((lws[0] * lws[1] * lws[2]) < nbthreads)
        lws[1] *= 2;
      if ((lws[0] * lws[1] * lws[2]) < nbthreads)
        lws[2] *= 2;
    }
    // 
    if ((lws[0] * lws[1] * lws[2]) > nbthreads && (lws[2] > 1))
      lws[2]--;
    if ((lws[0] * lws[1] * lws[2]) > nbthreads && (lws[1] > 1))
      lws[1]--;
    if ((lws[0] * lws[1] * lws[2]) > nbthreads && (lws[0] > 1))
      lws[0]--;
    // normalisation of dimensions to please OpenCL
    gws[0] = (sizec + lws[0] - 1) / lws[0];
    gws[0] *= lws[0];
    gws[1] = (sizec + lws[1] - 1) / lws[1];
    gws[1] *= lws[1];
    gws[2] = (sizec + lws[2] - 1) / lws[2];
    gws[2] *= lws[2];
  }

  if ((gws[0] * gws[1] * gws[2]) < nb)
    gws[0] += lws[0];

  leftover = nb - (gws[0] * gws[1] * gws[2]);
  if (leftover > 0) {
    fprintf(stderr,
            "nb %ld nbt %ld, gws %ld %ld %ld lws %ld %ld %ld (%ld)\n", nb,
            nbthreads, gws[0], gws[1], gws[2], lws[0], lws[1], lws[2], leftover);
    exit(1);
  }
  return;
}

double
oclChronoElaps(const cl_event event)
{
  cl_ulong Wstart, Wend;
  double start = 0, end = 0;
  cl_int err = 0;

  if (_profiling) {
    err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(Wstart), &Wstart, NULL);
    oclCheckErr(err, "clGetEventProfilingInfo Wstart");
    err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(Wend), &Wend, NULL);
    oclCheckErr(err, "clGetEventProfilingInfo Wend");
    start = Wstart / 1.e9;
    end = Wend / 1.e9;
  }
  return (end - start);
}

cl_uint
oclCreateProgramString(const char *fname, char ***pgmt, cl_uint * pgml)
{
  FILE *fd;
  int i = 0;
  int nbr = 500;
  char buffer[1024], *ptr;

  *pgmt = (char **) calloc(500, sizeof(char *));
  fd = fopen(fname, "r");

  if (fd) {
    while ((ptr = fgets(buffer, 1024, fd)) != NULL) {
      if (i >= (nbr - 2)) {
        nbr += 500;
        *pgmt = (char **) realloc(*pgmt, nbr * sizeof(char *));
      }
      (*pgmt)[i] = strdup(buffer);
      (*pgmt)[i + 1] = NULL;
      i++;
    }

    fclose(fd);
  } else {
    fprintf(stderr, "File %s not found; Aborting.\n", fname);
    abort();
  }
  *pgml = i;
  return CL_SUCCESS;
}

cl_program
oclCreatePgmFromCtx(const char *srcFile, const char *srcDir,
                    const cl_context ctx, const int theplatform, const int thedev, const int verbose)
{
  int i;
  cl_int err = 0;
  char **pgmt;
  cl_uint pgml;
  cl_program pgm;
  char *message = NULL;
  size_t msgl = 0;
  char options[1000];

  // create a programme
  // printf("CreateProgramString\n");
  // -- we put the whole file in a string
  err = oclCreateProgramString(srcFile, &pgmt, &pgml);
  // printf("CreateProgramString. (%d)\n", pgml);

  // Output the program
  if (verbose == 2) {
    for (i = 0; i < pgml; i++) {
      printf("%s", pgmt[i]);
    }
  }
  // create the OpenCL program from the string
  // printf("clCreateProgramWithSource\n");
  pgm = clCreateProgramWithSource(ctx, pgml, (const char **) pgmt, NULL, &err);
  // printf("clCreateProgramWithSource.\n");
  oclCheckErr(err, "Creation du program");

  // compilation
  //err = clBuildProgram(pgm, 0, NULL, "", NULL, NULL);
  // ,-cl-nv-opt-level 3
  strcpy(options, "");
  // strcat(options, "-O3 ");
  strcat(options, "-cl-mad-enable ");
#if AMDATI==1
  strcat(options, "-DAMDATI ");
  strcat(options, "-cl-std=CL1.1 ");
#endif
#if NVIDIA==1
  strcat(options, "-DNVIDIA ");
  // strcat(options, "-cl-nv-opt-level 3 ");
  // strcat(options, "-O3 ");
#endif
#if INTEL==1
  strcat(options, "-DINTEL ");
  // strcat(options, "-g "); // -g has a huge perf impact on a SNB quad core.
#endif
  if (pdesc[theplatform].devdesc[thedev].fpdp) {
    strcat(options, "-DHASFP64 ");
  }

  if (srcDir != NULL) {
    strcat(options, "-I");
    strcat(options, srcDir);
    strcat(options, " ");
  }

  err = clBuildProgram(pgm, 1, &pdesc[theplatform].devices[thedev], options, NULL, NULL);
  // printf("clBuildProgram.\n");

  // The only way to retrieve compilation information is to ask for them
  // CheckErr(err, "clGetProgramBuildInfo");
  if (err != CL_SUCCESS) {
    fprintf(stderr, "Build OpenCL (opts=\"%s\") has error(s).\n", options);
    oclPrintErr(err, "clBuildProgram", __FILE__, __LINE__);
    assert(pdesc != NULL);
    assert(pdesc[theplatform].devices != NULL);

    // get the message length for the buffer allocation
    err = clGetProgramBuildInfo(pgm, pdesc[theplatform].devices[thedev], CL_PROGRAM_BUILD_LOG, 0, NULL, &msgl);
    // fprintf(stderr, "longueur du message d'erreur %d\n", msgl);
    message = (char *) calloc(msgl + 16, 1);
    assert(message != NULL);
    err = clGetProgramBuildInfo(pgm, pdesc[theplatform].devices[thedev], CL_PROGRAM_BUILD_LOG, msgl, message, NULL);
    fprintf(stderr, "\n\n");
    fprintf(stderr, "------------------------------------\n");
    fprintf(stderr, "%s\n", message);
    fprintf(stderr, "------------------------------------\n");
    fprintf(stderr, "\n\n");
    oclCheckErr(err, "clGetProgramBuildInfo");
    abort();
  } else {
    if (verbose) fprintf(stderr, "Build OpenCL (opts=\"%s\") OK.\n", options);
  }
  // cleanup
  for (i = 0; i < pgml; i++) {
    if (pgmt[i])
      free(pgmt[i]);
  }
  free(pgmt);

  return pgm;
}

int
oclGetNbPlatforms(const int verbose)
{
  cl_device_type devtype;
  char *message = NULL;
  char *msg;
  size_t msgl = 0;
  int maxwid = 0;
  size_t maxwis[4];
  size_t maxwiss = 0;
  size_t maxwgs;
  size_t maxwgss = 0;
  size_t maxclkmhz = 0;
  cl_uint maxcu = 0;
  cl_ulong maxmemallocsz = 0;
  int i, j, theplatform;
  cl_int err = 0;
  cl_uint nbdevices = 0;
  cl_uint devmaxcu = 0;

  // informations on the platform
  err = 0;
  err = clGetPlatformIDs(0, NULL, &nbplatforms);
  oclCheckErr(err, "GetPlatformIDs -- 1");
  platform = (cl_platform_id *) calloc(nbplatforms, sizeof(cl_platform_id));
  err = clGetPlatformIDs(nbplatforms, platform, &nbplatforms);
  oclCheckErr(err, "GetPlatformIDs -- 2");

  pdesc = (PlatformDesc_t *) calloc(nbplatforms + 1, sizeof(PlatformDesc_t));
  assert(pdesc != NULL);

  if (verbose)
    printf("Nb platform : %d\n", nbplatforms);
  for (i = 0; i < nbplatforms; i++) {
    err = clGetPlatformInfo(platform[i], CL_PLATFORM_PROFILE, 0, (void *) NULL, &msgl);
    oclCheckErr(err, "GetPFInof PROFILE");
    message = (char *) calloc(msgl, 1);
    err = clGetPlatformInfo(platform[i], CL_PLATFORM_PROFILE, msgl, (void *) message, &msgl);
    oclCheckErr(err, "GetPFInof PROFILE");
    if (verbose)
      printf("[%d] Profile : %s\n", i, message);
    free(message);

    err = clGetPlatformInfo(platform[i], CL_PLATFORM_VERSION, 0, (void *) NULL, &msgl);
    oclCheckErr(err, "GetPFInof VERSION");
    message = (char *) calloc(msgl, 1);
    err = clGetPlatformInfo(platform[i], CL_PLATFORM_VERSION, msgl, (void *) message, &msgl);
    oclCheckErr(err, "GetPFInof VERSION");
    if (verbose)
      printf("[%d] VERSION : %s\n", i, message);
    free(message);

    err = clGetPlatformInfo(platform[i], CL_PLATFORM_NAME, 0, (void *) NULL, &msgl);
    oclCheckErr(err, "GetPFInof NAME");
    message = (char *) calloc(msgl, 1);
    err = clGetPlatformInfo(platform[i], CL_PLATFORM_NAME, msgl, (void *) message, &msgl);
    oclCheckErr(err, "GetPFInof NAME");
    if (verbose)
      printf("[%d] NAME : %s\n", i, message);
    free(message);

    err = clGetPlatformInfo(platform[i], CL_PLATFORM_VENDOR, 0, (void *) NULL, &msgl);
    oclCheckErr(err, "GetPFInof VENDOR");
    message = (char *) calloc(msgl, 1);
    err = clGetPlatformInfo(platform[i], CL_PLATFORM_VENDOR, msgl, (void *) message, &msgl);
    oclCheckErr(err, "GetPFInof VENDOR");
    if (verbose)
      printf("[%d] VENDOR : %s\n", i, message);
    free(message);

    err = clGetPlatformInfo(platform[i], CL_PLATFORM_EXTENSIONS, 0, (void *) NULL, &msgl);
    oclCheckErr(err, "GetPFInof EXTENSIONS");
    message = (char *) calloc(msgl, 1);
    err = clGetPlatformInfo(platform[i], CL_PLATFORM_EXTENSIONS, msgl, (void *) message, &msgl);
    oclCheckErr(err, "GetPFInof EXTENSIONS");
    if (verbose)
      printf("[%d] EXTENSIONS : %s\n", i, message);
    free(message);

    // prepare memory for future use
    err = clGetDeviceIDs(platform[i], CL_DEVICE_TYPE_ALL, 0, NULL, &nbdevices);
    oclCheckErr(err, "GetDeviceInfo -- 1");
    pdesc[i].nbdevices = nbdevices;
    pdesc[i].devices = (cl_device_id *) calloc(nbdevices, sizeof(cl_device_id));
    assert(pdesc[i].devices != NULL);
    pdesc[i].devdesc = (DeviceDesc_t *) calloc(nbdevices, sizeof(DeviceDesc_t));
    assert(pdesc[i].devdesc != NULL);
    err = clGetDeviceIDs(platform[i], CL_DEVICE_TYPE_ALL, nbdevices, pdesc[i].devices, &nbdevices);
    oclCheckErr(err, "GetDeviceInfo -- 2");

    theplatform=i;
    for (j = 0; j < nbdevices; j++) {
      err = clGetDeviceInfo(pdesc[theplatform].devices[j], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(maxcu), &maxcu, NULL);
      oclCheckErr(err, "deviceInfo maxcu");
      pdesc[theplatform].devdesc[j].maxcu = maxcu;
      if (verbose)
	printf("(%d) :: device maxcu %d", j, maxcu);
      err = clGetDeviceInfo(pdesc[theplatform].devices[j], CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(maxwid), &maxwid, NULL);
      oclCheckErr(err, "deviceInfo maxwid");
      if (verbose)
	printf(" mxwkitdim %d", maxwid);
      pdesc[theplatform].devdesc[j].mwid = maxwid;

      err = clGetDeviceInfo(pdesc[theplatform].devices[j], CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(maxwis[0]) * 3, &maxwis, &maxwiss);
      oclCheckErr(err, "deviceInfo maxwis");
      if (verbose)
	printf(" mxwkitsz %ld %ld %ld", maxwis[0], maxwis[1], maxwis[2]);
      memcpy(pdesc[theplatform].devdesc[j].mwis, maxwis, 3 * sizeof(size_t));

      err = clGetDeviceInfo(pdesc[theplatform].devices[j], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(maxwgs), &maxwgs, &maxwgss);
      oclCheckErr(err, "deviceInfo maxwgs");
      if (verbose)
	printf(" mxwkgsz %ld ", maxwgs);
      pdesc[theplatform].devdesc[j].mwgs = maxwgs;

      err = clGetDeviceInfo(pdesc[theplatform].devices[j], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(maxclkmhz), &maxclkmhz, NULL);
      oclCheckErr(err, "deviceInfo maxclkmhz");
      if (verbose)
	printf(" mxclockMhz %ld", maxclkmhz);

      err =
	clGetDeviceInfo(pdesc[theplatform].devices[j], CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(maxmemallocsz), &maxmemallocsz, NULL);
      oclCheckErr(err, "deviceInfo maxmemallocsz");
      if (verbose)
	printf(" mxmemallocsz %ld (Mo)", maxmemallocsz / (1024 * 1024));
      pdesc[theplatform].devdesc[j].mmas = maxmemallocsz;
      err =
	clGetDeviceInfo(pdesc[theplatform].devices[j], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(maxmemallocsz), &maxmemallocsz, NULL);
      oclCheckErr(err, "deviceInfo maxmemallocsz");
      if (verbose)
	printf(" globmemsz %ld (Mo)",  maxmemallocsz / (1024 * 1024));
      pdesc[theplatform].devdesc[j].dgms = maxmemallocsz;

      message = calloc(1024, 1);
      err = clGetDeviceInfo(pdesc[theplatform].devices[j], CL_DEVICE_TYPE, sizeof(devtype), &devtype, NULL);
      oclCheckErr(err, "deviceInfo");
      if (verbose)
	printf(" type %ld", devtype);
      switch (devtype) {
      case CL_DEVICE_TYPE_CPU:
	strcpy(message, "CPU");
	pdesc[theplatform].devdesc[j].nature = strdup("CPU");
	break;
      case CL_DEVICE_TYPE_GPU:
	strcpy(message, "GPU");
	pdesc[theplatform].devdesc[j].nature = strdup("GPU");
	break;
      case CL_DEVICE_TYPE_ACCELERATOR:
	strcpy(message, "ACCELERATOR");
	pdesc[theplatform].devdesc[j].nature = strdup("ACCELERATOR");
	break;
      }
      if (verbose)
	printf(" [%s]\n", message);
      free(message);

      err = clGetDeviceInfo(pdesc[theplatform].devices[j], CL_DEVICE_EXTENSIONS, 0, NULL, &msgl);
      oclCheckErr(err, "deviceInfo");
      msg = calloc(msgl, 1);
      err = clGetDeviceInfo(pdesc[theplatform].devices[j], CL_DEVICE_EXTENSIONS, msgl, msg, &msgl);
      oclCheckErr(err, "deviceInfo");
      if (verbose) {
	printf("   extensions: %s\n", msg);
      }
      pdesc[theplatform].devdesc[j].fpdp = 0;
      if (strstr(msg, "cl_khr_fp64") != NULL) {
	if (verbose) {
	  printf("Device %d supports double precision floating point\n", j);
	}
	pdesc[theplatform].devdesc[j].fpdp = 1;
      }
      free(msg);
      err =
	clGetDeviceInfo(pdesc[theplatform].devices[j],
			CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR, sizeof(int), &pdesc[theplatform].devdesc[j].vecchar, NULL);
      err =
	clGetDeviceInfo(pdesc[theplatform].devices[j],
			CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT, sizeof(int), &pdesc[theplatform].devdesc[j].vecshort, NULL);
      err =
	clGetDeviceInfo(pdesc[theplatform].devices[j],
			CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT, sizeof(int), &pdesc[theplatform].devdesc[j].vecint, NULL);
      err =
	clGetDeviceInfo(pdesc[theplatform].devices[j],
			CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG, sizeof(int), &pdesc[theplatform].devdesc[j].veclong, NULL);
      err =
	clGetDeviceInfo(pdesc[theplatform].devices[j],
			CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT, sizeof(int), &pdesc[theplatform].devdesc[j].vecfloat, NULL);
      err =
	clGetDeviceInfo(pdesc[theplatform].devices[j],
			CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE, sizeof(int), &pdesc[theplatform].devdesc[j].vecdouble, NULL);
      if (verbose) {
	printf("   prefered vector size: c=%d s=%d i=%d l=%d f=%d d=%d\n",
	       pdesc[theplatform].devdesc[j].vecchar,
	       pdesc[theplatform].devdesc[j].vecshort,
	       pdesc[theplatform].devdesc[j].vecint,
	       pdesc[theplatform].devdesc[j].veclong,
	       pdesc[theplatform].devdesc[j].vecfloat, pdesc[theplatform].devdesc[j].vecdouble);
      }
    }
  }
  return nbplatforms;
}

cl_context
oclCreateCtxForPlatform(const int theplatform, const int verbose)
{
  cl_int err = 0;
  cl_uint nbdevices = 0;
  cl_context ctx;
  cl_context_properties proplist[1000];

  // memory has been allocated while probing the platforms
  assert(pdesc[theplatform].devices != NULL);
  assert(pdesc[theplatform].devdesc != NULL);
  nbdevices = pdesc[theplatform].nbdevices;

  if (verbose)
    printf("[%d] : nbdevices = %d\n", theplatform, nbdevices);
  if (verbose)
    fflush(stdout);
  // Create a contexte for the platform

  proplist[0] = CL_CONTEXT_PLATFORM;
  proplist[1] = (cl_context_properties) platform[theplatform];
  proplist[2] = (cl_context_properties) NULL;
  ctx = clCreateContext(proplist, pdesc[theplatform].nbdevices, pdesc[theplatform].devices, NULL, NULL, &err);
  oclCheckErr(err, "Creation CTX");

  return ctx;
}

cl_command_queue
oclCreateCommandQueueForDev(const int theplatform, const int devselected, const cl_context ctx, const int profiling)
{
  cl_command_queue cqueue;
  cl_int err = 0;

  assert(pdesc != NULL);
  assert(pdesc[theplatform].devices != NULL);

  // create a command queue
  if (profiling) {
    cqueue = clCreateCommandQueue(ctx, pdesc[theplatform].devices[devselected], CL_QUEUE_PROFILING_ENABLE, &err);
    _profiling = 1;
  } else {
    cqueue = clCreateCommandQueue(ctx, pdesc[theplatform].devices[devselected], 0, &err);
  }
  // could be CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE
  oclCheckErr(err, "Creation queue");
  return cqueue;
}

int
oclGetNumberOfDev(const int theplatform)
{
  assert(pdesc != NULL);

  return pdesc[theplatform].nbdevices;
}

int
oclGetNbOfAcc(const int theplatform)
{
  int j, nbAcc = 0;
  cl_int err = 0;
  cl_device_type devtype;

  assert(pdesc != NULL);
  assert(pdesc[theplatform].devices != NULL);

  for (j = 0; j < pdesc[theplatform].nbdevices; j++) {
    err = clGetDeviceInfo(pdesc[theplatform].devices[j], CL_DEVICE_TYPE, sizeof(devtype), &devtype, NULL);
    oclCheckErr(err, "deviceInfo");
    switch (devtype) {
    case CL_DEVICE_TYPE_ACCELERATOR:
      nbAcc++;
      break;
    }
  }
  return nbAcc;
}

int
oclGetNbOfGpu(const int theplatform)
{
  int j, nbgpu = 0;
  cl_int err = 0;
  cl_device_type devtype;

  assert(pdesc != NULL);
  assert(pdesc[theplatform].devices != NULL);

  for (j = 0; j < pdesc[theplatform].nbdevices; j++) {
    err = clGetDeviceInfo(pdesc[theplatform].devices[j], CL_DEVICE_TYPE, sizeof(devtype), &devtype, NULL);
    oclCheckErr(err, "deviceInfo");
    switch (devtype) {
    case CL_DEVICE_TYPE_GPU:
      nbgpu++;
      break;
    }
  }
  return nbgpu;
}


int
oclGetNbOfCpu(const int theplatform)
{
  int j, nbcpu = 0;
  cl_int err = 0;
  cl_device_type devtype;

  assert(pdesc != NULL);
  assert(pdesc[theplatform].devices != NULL);

  for (j = 0; j < pdesc[theplatform].nbdevices; j++) {
    err = clGetDeviceInfo(pdesc[theplatform].devices[j], CL_DEVICE_TYPE, sizeof(devtype), &devtype, NULL);
    oclCheckErr(err, "deviceInfo");
    switch (devtype) {
    case CL_DEVICE_TYPE_CPU:
      nbcpu++;
      break;
    }
  }
  return nbcpu;
}

int
oclGetAccDev(const int theplatform, const int accnum)
{
  int j, nbacc = 0, numdev = -1;
  cl_int err = 0;
  cl_device_type devtype;

  assert(pdesc != NULL);
  assert(pdesc[theplatform].devices != NULL);

  for (j = 0; j < pdesc[theplatform].nbdevices; j++) {
    err = clGetDeviceInfo(pdesc[theplatform].devices[j], CL_DEVICE_TYPE, sizeof(devtype), &devtype, NULL);
    oclCheckErr(err, "deviceInfo");
    switch (devtype) {
    case CL_DEVICE_TYPE_ACCELERATOR:
      if (accnum == nbacc) {
	numdev = j;
      }
      nbacc++;
      break;
    }
  }
  return numdev;
}

int
oclGetGpuDev(const int theplatform, const int gpunum)
{
  int j, nbgpu = 0, numdev = -1;
  cl_int err = 0;
  cl_device_type devtype;

  assert(pdesc != NULL);
  assert(pdesc[theplatform].devices != NULL);

  for (j = 0; j < pdesc[theplatform].nbdevices; j++) {
    err = clGetDeviceInfo(pdesc[theplatform].devices[j], CL_DEVICE_TYPE, sizeof(devtype), &devtype, NULL);
    oclCheckErr(err, "deviceInfo");
    switch (devtype) {
    case CL_DEVICE_TYPE_GPU:
      if (gpunum == nbgpu) {
	numdev = j;
      }
      nbgpu++;
      break;
    }
  }
  return numdev;
}

int
oclGetCpuDev(const int theplatform, const int cpunum)
{
  int j, nbcpu = 0, numdev = -1;
  cl_int err = 0;
  cl_device_type devtype;

  assert(pdesc != NULL);
  assert(pdesc[theplatform].devices != NULL);

  for (j = 0; j < pdesc[theplatform].nbdevices; j++) {
    err = clGetDeviceInfo(pdesc[theplatform].devices[j], CL_DEVICE_TYPE, sizeof(devtype), &devtype, NULL);
    oclCheckErr(err, "deviceInfo");
    switch (devtype) {
    case CL_DEVICE_TYPE_CPU:
      if (cpunum == nbcpu) {
	numdev = j;
      }
      nbcpu++;
      break;
    }
  }
  return numdev;
}

cl_device_id
oclGetDeviceOfCQueue(cl_command_queue q)
{
  cl_device_id res;
  cl_int err = 0;
  size_t lres = 0;
  err = clGetCommandQueueInfo(q, CL_QUEUE_DEVICE, sizeof(cl_device_id), &res, &lres);
  oclCheckErr(err, "clGetCommandQueueInfo qDev");
  return res;
}

cl_platform_id
oclGetPlatformOfCQueue(cl_command_queue q)
{
  cl_platform_id res;
  cl_int err = 0;
  size_t lres = 0;
  err = clGetDeviceInfo(oclGetDeviceOfCQueue(q), CL_DEVICE_PLATFORM, sizeof(cl_platform_id), &res, &lres);
  oclCheckErr(err, "clGetCommandQueueInfo qDev");
  return res;
}

cl_context
oclGetContextOfCQueue(cl_command_queue q)
{
  cl_context res;
  cl_int err = 0;
  size_t lres = 0;
  err = clGetCommandQueueInfo(q, CL_QUEUE_CONTEXT, sizeof(cl_context), &res, &lres);
  oclCheckErr(err, "clGetCommandQueueInfo qCtx");
  return res;
}

size_t
oclGetMaxWorkSize(cl_kernel k, cl_device_id d)
{
  cl_int err = 0;
  size_t lres = 0;
  size_t res;
  size_t multp = 0;

#ifdef INTEL
  // For some reason I still have to figure out, this branch gives the best results on the KNC
   err = clGetDeviceInfo(d, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(res), &res, NULL);
   oclCheckErr(err, "oclGetMaxWorkSize maxcu");
#else
   // whereas this one work fine for the NVIDIA (and possibly for AMD)
   err = clGetKernelWorkGroupInfo(k, d, CL_KERNEL_WORK_GROUP_SIZE, sizeof(res), &res, NULL);
   oclCheckErr(err, "oclGetMaxWorkSize maxcu"); 
#endif
   err = clGetKernelWorkGroupInfo(k, d, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(multp), &multp, NULL);
   oclCheckErr(err, "clGetKernelWorkGroupInfo  ");
   // printf("res %ld multp %ld\n", res, multp);
   if ((res % multp) != 0) {
     res = oclMultiple(res, multp);
   }
   return res;
}


size_t
oclGetMaxMemAllocSize(int theplatform, int thedev)
{
  assert(pdesc != NULL);
  assert(pdesc[theplatform].devices != NULL);
  assert(pdesc[theplatform].devices[thedev] != NULL);
  return pdesc[theplatform].devdesc[thedev].mmas;
}

void
oclSetArg(cl_kernel k, cl_uint narg, size_t l, const void *arg, const char * file, const int line)
{
  cl_int err = 0;
  err = clSetKernelArg(k, narg, l, arg);
  if (err != CL_SUCCESS) {
    char msg[2048];
    sprintf(msg, "clSetKernelArg [%s, l=%d] arg=%d", file, line, narg);
    oclCheckErr(err, msg);
  }
}

void
oclSetArgLocal(cl_kernel k, cl_uint narg, size_t l, const char * file, const int line)
{
  cl_int err = 0;
  err = clSetKernelArg(k, narg, l, NULL);
  if (err != CL_SUCCESS) {
    char msg[2048];
    sprintf(msg, "clSetKernelArgLocal [%s, l=%d] arg=%d", file, line, narg);
    oclCheckErr(err, msg);
  }
}

void
oclNbBlocks(cl_kernel k, cl_command_queue q, size_t nbobj, int nbthread, long *maxth, long *nbblocks)
{
  dim3 gws, lws;
  int maxThreads = 0;
  maxThreads = oclGetMaxWorkSize(k, oclGetDeviceOfCQueue(q));
  maxThreads = MIN(maxThreads, nbthread);

  oclMkNDrange(nbobj, maxThreads, NDR_1D, gws, lws);
  *maxth = maxThreads;
  *nbblocks = (gws[0] * gws[1] * gws[2]) / maxThreads;
  return;
}

double
oclLaunchKernel(cl_kernel k, cl_command_queue q, int nbobj, int nbthread, const char *fname, const int line)
{
  cl_int err = 0;
  dim3 gws, lws;
  cl_event event;
  double elapsk;
  int maxThreads = 0;
  cl_uint one = 1;
  cl_device_id dId = oclGetDeviceOfCQueue(q);
  size_t prefsz = 32;

  maxThreads = oclGetMaxWorkSize(k, dId);
  maxThreads = MIN(maxThreads, nbthread);

  // Get the proper size for the hardware
  err = clGetKernelWorkGroupInfo(k, dId, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(prefsz), &prefsz, NULL);
  oclCheckErr(err, "clGetKernelWorkGroupInfo CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE");
 
  // make sure we have the proper multiple: AMD 7970 crashes is not met.
  maxThreads = oclMultiple(maxThreads, prefsz);
  // printf("1D %d \n", maxThreads);

  oclMkNDrange(nbobj, maxThreads, NDR_1D, gws, lws);
  // printf("Launch: %ld G:%ld %ld %ld L:%ld %ld %ld\n", nbobj, gws[0], gws[1], gws[2], lws[0], lws[1], lws[2]);

  err = clEnqueueNDRangeKernel(q, k, NDR_1D, NULL, gws, lws, 0, NULL, &event);
  oclCheckErrF(err, "clEnqueueNDRangeKernel", fname, line);

  err = clWaitForEvents(one, &event);
  oclCheckErrF(err, "clWaitForEvents", fname, line);

  elapsk = oclChronoElaps(event);

  err = clReleaseEvent(event);
  oclCheckErrF(err, "clReleaseEvent", fname, line);

  return elapsk;
}

double
oclLaunchKernel2D(cl_kernel k, cl_command_queue q, int nbobjx, int nbobjy, int nbthread, const char *fname, const int line)
{
  cl_int err = 0;
  dim3 gws, lws;
  cl_event event;
  double elapsk;
  int maxThreads = 0;
  cl_uint one = 1;
  cl_device_id dId = oclGetDeviceOfCQueue(q);
  size_t prefsz = 32;

  maxThreads = oclGetMaxWorkSize(k, dId);
  // printf("%d ", maxThreads);
  maxThreads = MIN(maxThreads, nbthread);
  // printf("%d ", nbthread);

  // Get the proper size for the hardware
  err = clGetKernelWorkGroupInfo(k, dId, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(prefsz), &prefsz, NULL);
  oclCheckErr(err, "clGetKernelWorkGroupInfo CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE");
 
  // make sure we have the proper multiple: AMD 7970 crashes is not met.
  maxThreads = oclMultiple(maxThreads, prefsz);
  // printf("%ld ", prefsz);
  // printf("2D %d \n", maxThreads);

  gws[2] = lws[2] = 0;
  gws[1] = lws[1] = 1;
  gws[0] = lws[0] = 1;
  //
  lws[0] = maxThreads;
  // lws[0] /= 2; lws[1] *= 2;
  gws[0] = oclMultiple(nbobjx, lws[0]);
  gws[1] = oclMultiple(nbobjy, lws[1]);

  // printf("Launch: %ld G:%ld %ld %ld L:%ld %ld %ld\n", nbobjx * nbobjy, gws[0], gws[1], gws[2], lws[0], lws[1], lws[2]);
  err = clEnqueueNDRangeKernel(q, k, NDR_2D, NULL, gws, lws, 0, NULL, &event);
  oclCheckErrF(err, "clEnqueueNDRangeKernel", fname, line);

  err = clWaitForEvents(one, &event);
  oclCheckErrF(err, "clWaitForEvents", fname, line);

  elapsk = oclChronoElaps(event);

  err = clReleaseEvent(event);
  oclCheckErrF(err, "clReleaseEvent", fname, line);

  return elapsk;
}

double
oclLaunchKernel3D(cl_kernel k, cl_command_queue q, int nbobjx, int nbobjy, int nbobjz, int nbthread, const char *fname, const int line)
{
  cl_int err = 0;
  dim3 gws, lws;
  cl_event event;
  double elapsk;
  int maxThreads = 0;
  cl_uint one = 1;
  cl_device_id dId = oclGetDeviceOfCQueue(q);
  size_t prefsz = 32;

  maxThreads = oclGetMaxWorkSize(k, dId);
  // printf("%d ", maxThreads);
  maxThreads = MIN(maxThreads, nbthread);
  // printf("%d ", nbthread);
  // printf("%d ", maxThreads);

  // Get the proper size for the hardware
  err = clGetKernelWorkGroupInfo(k, dId, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(prefsz), &prefsz, NULL);
  oclCheckErr(err, "clGetKernelWorkGroupInfo CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE");
 
  // make sure we have the proper multiple: AMD 7970 crashes is not met.
  maxThreads = oclMultiple(maxThreads, prefsz);
  // printf("%ld ", prefsz);
  // printf("3D %d\n", maxThreads);

  gws[2] = lws[2] = 1;
  gws[1] = lws[1] = 1;
  gws[0] = lws[0] = 1;
  //
  lws[0] = maxThreads;
  //lws[0] /= 2; lws[1] *= 2;
  //lws[0] /= 2; lws[2] *= 2;
  gws[0] = oclMultiple(nbobjx, lws[0]);
  gws[1] = oclMultiple(nbobjy, lws[1]);
  gws[2] = oclMultiple(nbobjz, lws[2]);

  // printf("Launch: %ld G:%ld %ld %ld L:%ld %ld %ld\n", nbobjx * nbobjy, gws[0], gws[1], gws[2], lws[0], lws[1], lws[2]);
  err = clEnqueueNDRangeKernel(q, k, NDR_3D, NULL, gws, lws, 0, NULL, &event);
  oclCheckErrF(err, "clEnqueueNDRangeKernel", fname, line);

  err = clWaitForEvents(one, &event);
  oclCheckErrF(err, "clWaitForEvents", fname, line);

  elapsk = oclChronoElaps(event);

  err = clReleaseEvent(event);
  oclCheckErrF(err, "clReleaseEvent", fname, line);

  return elapsk;
}

int
oclFp64Avail(int theplatform, int thedev)
{
  return pdesc[theplatform].devdesc[thedev].fpdp;
}

char *
oclGetDevNature(const int theplatform, const int thedev)
{
  return pdesc[theplatform].devdesc[thedev].nature;
}

//EOF
