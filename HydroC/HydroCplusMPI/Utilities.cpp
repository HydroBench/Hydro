#ifdef MPI_ON
#include <mpi.h>
#endif
#include <stdint.h>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <malloc.h>
#include <sys/types.h>

#include "Options.hpp"
#include "EnumDefs.hpp"
#include "Utilities.hpp"

void CalcSubSurface(int xmin, int xmax, int ymin, int ymax, int pmin, int pmax, int *box, int mype)
{
	int nbpe = (pmax - pmin + 1);
	int ny = (int)sqrt(nbpe);
	int res = (int)(nbpe - ny * ny) / ny;
	int nx = ny + res;
	int pex = mype % nx;
	int pey = mype / nx;
	int lgx = (xmax - xmin + 1);
	int incx = lgx / nx;
	int lgy = (ymax - ymin + 1);
	int incy = lgy / ny;
	static int done = 0;

	if (nx * ny != nbpe) {
		// the closest shape to a square can't be maintain.
		// Let's find another alternative
		int divider = 2;
		int lastdevider = 1;
		while (divider < (int)sqrt(nbpe)) {
			if ((nbpe % divider) == 0) {
				lastdevider = divider;
			}
			divider++;
		}

		// if (mype == 0) printf("Last divider %d\n", lastdevider);

		if (lastdevider == 1) {
			if (mype == 0) {
				fprintf(stderr, "\tERROR: %d can't be devided evenly in x and y\n", nbpe);
				fprintf(stderr, "\tERROR: closest value is %d\n", nx * ny);
				fprintf(stderr, "\tERROR: please adapt the number of process\n");
			}
#ifdef MPI_ON
			MPI_Finalize();
#endif
			abort();
		}
		ny = lastdevider;
		res = (int)(nbpe - ny * ny) / ny;
		nx = ny + res;
		pex = mype % nx;
		pey = mype / nx;
		incx = lgx / nx;
		incy = lgy / ny;
	}

	if ((incx * nx + xmin) < xmax)
		incx++;
	if ((incy * ny + ymin) < ymax)
		incy++;

	if (mype == 0 && !done) {
		printf("HydroC: Simple decomposition\n");
		printf("HydroC: nx=%d ny=%d\n", nx, ny);
		done = 1;
	}

	box[XMIN_D] = pex * incx + xmin;
	if (box[XMIN_D] < 0)
		box[XMIN_D] = 0;

	box[XMAX_D] = (pex + 1) * incx + xmin;
	if (box[XMAX_D] > xmax)
		box[XMAX_D] = xmax;

	box[YMIN_D] = pey * incy + ymin;
	if (box[YMIN_D] < 0)
		box[YMIN_D] = 0;

	box[YMAX_D] = (pey + 1) * incy + ymin;
	if (box[YMAX_D] > ymax)
		box[YMAX_D] = ymax;

	box[UP_D] = mype + nx;
	if (box[UP_D] >= nbpe)
		box[UP_D] = -1;
	box[DOWN_D] = mype - nx;
	if (box[DOWN_D] < 0)
		box[DOWN_D] = -1;
	box[LEFT_D] = mype - 1;
	if (pex == 0)
		box[LEFT_D] = -1;
	box[RIGHT_D] = mype + 1;
	if (pex + 1 >= nx)
		box[RIGHT_D] = -1;
}

void getCPUName(char cpuName[1024])
{
	char cmd[1024];
	memset(cmd, 0, 1024);
	sprintf(cmd, "/proc/cpuinfo");
	FILE *fic = fopen(cmd, "r");

	if (fic != 0) {
		while (!feof(fic)) {
			char *l = fgets(cmd, 1023, fic);
			if (strstr(cmd, "model name") != NULL) {
				char *p = strchr(cmd, ':');
				strcpy(cpuName, p);
				p = strchr(cpuName, '\n');
				if (p)
					*p = 0;
			}
		}
		fclose(fic);
	}
}

long getMemUsed(void)
{
	long maxMemUsed = 0;
	char cmd[1024];
	memset(cmd, 0, 1024);
	sprintf(cmd, "/proc/%d/statm", getpid());
	FILE *fic = fopen(cmd, "r");
	if (fic != 0) {
		while (!feof(fic)) {
			int l = fscanf(fic, "%ld ", &maxMemUsed);
			break;
		}
		fclose(fic);
	}
	return maxMemUsed * getpagesize();
}

template < typename T > int AlignedAlloc(T ** r, size_t lg)
{
#if ALIGNED == 0
	*r = (T *) malloc(lg * sizeof(T));
#else
	*r = (T *) memalign(64, lg * sizeof(T));
#endif
	return *r != 0;
}

real_t *AlignedAllocReal(size_t lg)
{
	real_t *r;
	AlignedAlloc(&r, lg);
	return r;
}

int *AlignedAllocInt(size_t lg)
{
	int *r;
	AlignedAlloc(&r, lg);
	return r;
}

long *AlignedAllocLong(size_t lg)
{
	long *r;
	AlignedAlloc(&r, lg);
	return r;
}
