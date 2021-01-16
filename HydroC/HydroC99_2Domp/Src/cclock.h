#ifndef CCLOCK_H
#define CCLOCK_H
#include <time.h>

/*
  A small helper function to manipulate accurate times (on LINUX).

  Do not forget to add    -lrt   at link time
  (C) G. Colin de Verdiere, CEA.

 */

typedef struct timespec ts_t;

#ifdef __cplusplus
extern "C" {
#endif
    ts_t cclock(void);		// for high precision operations on time
    double dcclock(void);	// might loose some precision 

    double ccelaps(ts_t start, ts_t end);
#ifdef __cplusplus
};
#endif

#endif				// CCLOCK_H
