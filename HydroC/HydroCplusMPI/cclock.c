#include "time.h"
#include "math.h"
#include "stdio.h"
#include "cclock.h"

/*
  A small helper function to manipulate accurate times (on LINUX).

  Do not forget to add    -lrt   at link time
  (C) G. Colin de Verdiere, CEA.

 */

void psecs(struct timespec start)
{
	printf("(%ld %ld)\n", start.tv_sec, start.tv_nsec);
}

double tseconde(struct timespec start)
{
	return (double)start.tv_sec + (double)1e-9 *(double)start.tv_nsec;
}

double dcclock(void)
{
	return tseconde(cclock());
}

double ccelaps(struct timespec start, struct timespec end)
{
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

struct timespec cclock(void)
{
	struct timespec tstart;
	clockid_t cid = CLOCK_REALTIME;
	int status = 0;

	status = clock_gettime(cid, &tstart);
	return tstart;
}

void convertToHuman(char *s, double t)
{
	int subsec;
	int days, hours, minutes, secondes;
	double curt = t;

	s[0] = 0;
	days = (int)(curt / (3600 * 24));
	curt -= (days * 3600 * 24);
	hours = (int)(curt / 3600);
	curt -= (hours * 3600);
	minutes = (int)(curt / 60);
	curt -= (minutes * 60);
	secondes = (int)(curt);
	subsec = (int)(((float)(curt) - (float)(secondes)) * 100);
	if (days)
		sprintf(s, "[%d:]", days);
	sprintf(s, "%s%02d:%02d:%02d.%d", s, hours, minutes, secondes, subsec);
}

//     
