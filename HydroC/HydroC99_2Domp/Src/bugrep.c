#include <stdio.h>
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
ts_t cclock(void);
double dcclock(void);

double ccelaps(ts_t start, ts_t end);
#ifdef __cplusplus
};
#endif

#endif

int main(int argc, char **argv) {
    struct timespec start, end;
    start = cclock();
    printf("Hello World\n");
    end = cclock();

    printf("%lf %lf\n", start, end);
    return 0;
}
