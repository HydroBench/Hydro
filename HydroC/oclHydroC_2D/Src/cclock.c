#include "time.h"
#include "math.h"
#include "stdio.h"
#include "cclock.h"

/*
  A small helper function to manipulate accurate times (on LINUX).

  Do not forget to add    -lrt   at link time
  (C) G. Colin de Verdiere, CEA.

 */


void
psecs(struct timespec start) {
  printf("(%ld %ld)\n", start.tv_sec, start.tv_nsec);
}

double
tseconde(struct timespec start) {
  return (double) start.tv_sec + (double) 1e-9 *(double) start.tv_nsec;
}

double
dcclock(void) {
  return tseconde(cclock());
}

double
ccelaps(struct timespec start, struct timespec end) {
  double ds = end.tv_sec - start.tv_sec;
  double dns = end.tv_nsec - start.tv_nsec;
  if (dns < 0) {
    // wrap around will occur in the nanosec part. Compensate this.
    dns = 1e9 + end.tv_nsec - start.tv_nsec;
    ds = ds - 1;
  }
  double telaps = ds + dns * 1e-9;
  return telaps;
}

struct timespec
cclock(void) {
  struct timespec tstart;
  clockid_t cid = CLOCK_REALTIME;
  int status = 0;

  status = clock_gettime(cid, &tstart);
  return tstart;
}

//     
