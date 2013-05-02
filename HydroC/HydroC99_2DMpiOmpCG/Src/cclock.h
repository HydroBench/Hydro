#ifndef CCLOCK_H
#define CCLOCK_H
#include "time.h"

/*
  A small helper function to manipulate accurate times (on LINUX).

  Do not forget to add    -lrt   at link time
  (C) G. Colin de Verdiere, CEA.

 */


#ifdef __cplusplus
extern "C" {
#endif
  struct timespec cclock(void); // for high precision operations on time
  double dcclock(void);         // might loose some precision 

  double ccelaps(struct timespec start, struct timespec end);
#ifdef __cplusplus
};
#endif

#endif // CCLOCK_H
