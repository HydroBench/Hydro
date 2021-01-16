#include <stdio.h>
#include <string.h>
#include <strings.h>
#include <unistd.h>
#include <stdlib.h>
#include <malloc.h>
#include <assert.h>
#include <limits.h>
#include <sys/time.h>
#include <float.h>
#include <math.h>

//
#include "perfcnt.h"

// - - - Performance counting
long flopsAri = 0;
long flopsSqr = 0;
long flopsMin = 0;
long flopsTra = 0;

double MflopsSUM = 0;
long nbFLOPS = 0;

//EOF
