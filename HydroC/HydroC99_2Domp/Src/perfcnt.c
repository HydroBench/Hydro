#include <assert.h>
#include <float.h>
#include <limits.h>
#include <malloc.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/time.h>
#include <unistd.h>

//
#include "perfcnt.h"

// - - - Performance counting
long flopsAri = 0;
long flopsSqr = 0;
long flopsMin = 0;
long flopsTra = 0;

double MflopsSUM = 0;
long nbFLOPS = 0;

// EOF
