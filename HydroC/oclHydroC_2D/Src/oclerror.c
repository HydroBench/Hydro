//
// (C) Guillaume.Colin-de-Verdiere at CEA.Fr
//
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

//
#include "oclerror.h"

void
oclPrintErr(cl_int rc, const char *msg, const char *file, const int line)
{
  char message[1000];
  if (rc != CL_SUCCESS) {
    strcpy(message, "");
    switch (rc) {
      case 0:
        strcat(message, "CL_SUCCESS");;
        break;
      case -1:
        strcat(message, "CL_DEVICE_NOT_FOUND");;
        break;
      case -2:
        strcat(message, "CL_DEVICE_NOT_AVAILABLE");;
        break;
      case -3:
        strcat(message, "CL_COMPILER_NOT_AVAILABLE");;
        break;
      case -4:
        strcat(message, "CL_MEM_OBJECT_ALLOCATION_FAILURE");;
        break;
      case -5:
        strcat(message, "CL_OUT_OF_RESOURCES");;
        break;
      case -6:
        strcat(message, "CL_OUT_OF_HOST_MEMORY");;
        break;
      case -7:
        strcat(message, "CL_PROFILING_INFO_NOT_AVAILABLE");;
        break;
      case -8:
        strcat(message, "CL_MEM_COPY_OVERLAP");;
        break;
      case -9:
        strcat(message, "CL_IMAGE_FORMAT_MISMATCH");;
        break;
      case -10:
        strcat(message, "CL_IMAGE_FORMAT_NOT_SUPPORTED");;
        break;
      case -11:
        strcat(message, "CL_BUILD_PROGRAM_FAILURE");;
        break;
      case -12:
        strcat(message, "CL_MAP_FAILURE");;
        break;
      case -13:
        strcat(message, "CL_MISALIGNED_SUB_BUFFER_OFFSET");;
        break;
      case -30:
        strcat(message, "CL_INVALID_VALUE");;
        break;
      case -31:
        strcat(message, "CL_INVALID_DEVICE_TYPE");;
        break;
      case -32:
        strcat(message, "CL_INVALID_PLATFORM");;
        break;
      case -33:
        strcat(message, "CL_INVALID_DEVICE");;
        break;
      case -34:
        strcat(message, "CL_INVALID_CONTEXT");;
        break;
      case -35:
        strcat(message, "CL_INVALID_QUEUE_PROPERTIES");;
        break;
      case -36:
        strcat(message, "CL_INVALID_COMMAND_QUEUE");;
        break;
      case -37:
        strcat(message, "CL_INVALID_HOST_PTR");;
        break;
      case -38:
        strcat(message, "CL_INVALID_MEM_OBJECT");;
        break;
      case -39:
        strcat(message, "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR");;
        break;
      case -40:
        strcat(message, "CL_INVALID_IMAGE_SIZE");;
        break;
      case -41:
        strcat(message, "CL_INVALID_SAMPLER");;
        break;
      case -42:
        strcat(message, "CL_INVALID_BINARY");;
        break;
      case -43:
        strcat(message, "CL_INVALID_BUILD_OPTIONS");;
        break;
      case -44:
        strcat(message, "CL_INVALID_PROGRAM");;
        break;
      case -45:
        strcat(message, "CL_INVALID_PROGRAM_EXECUTABLE");;
        break;
      case -46:
        strcat(message, "CL_INVALID_KERNEL_NAME");;
        break;
      case -47:
        strcat(message, "CL_INVALID_KERNEL_DEFINITION");;
        break;
      case -48:
        strcat(message, "CL_INVALID_KERNEL");;
        break;
      case -49:
        strcat(message, "CL_INVALID_ARG_INDEX");;
        break;
      case -50:
        strcat(message, "CL_INVALID_ARG_VALUE");;
        break;
      case -51:
        strcat(message, "CL_INVALID_ARG_SIZE");;
        break;
      case -52:
        strcat(message, "CL_INVALID_KERNEL_ARGS");;
        break;
      case -53:
        strcat(message, "CL_INVALID_WORK_DIMENSION");;
        break;
      case -54:
        strcat(message, "CL_INVALID_WORK_GROUP_SIZE");;
        break;
      case -55:
        strcat(message, "CL_INVALID_WORK_ITEM_SIZE");;
        break;
      case -56:
        strcat(message, "CL_INVALID_GLOBAL_OFFSET");;
        break;
      case -57:
        strcat(message, "CL_INVALID_EVENT_WAIT_LIST");;
        break;
      case -58:
        strcat(message, "CL_INVALID_EVENT");;
        break;
      case -59:
        strcat(message, "CL_INVALID_OPERATION");;
        break;
      case -60:
        strcat(message, "CL_INVALID_GL_OBJECT");;
        break;
      case -61:
        strcat(message, "CL_INVALID_BUFFER_SIZE");;
        break;
      case -62:
        strcat(message, "CL_INVALID_MIP_LEVEL");;
        break;
      case -63:
        strcat(message, "CL_INVALID_GLOBAL_WORK_SIZE");;
        break;
      case -1001:
        strcat(message, "CL_PLATFORM_NOT_FOUND_KHR");;
        break;
      default:
        strcat(message, "unknown code");
    }
    fprintf(stderr, "Error %d <%s> (%s) [f=%s l=%d]\n", rc, message, msg, file, line);
  }
}

void
oclCheckErrF(cl_int rc, const char *msg, const char *file, const int line)
{
  if (rc != CL_SUCCESS) {
    oclPrintErr(rc, msg, file, line);
    abort();
  }
}


//EOF
