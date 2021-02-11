#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <stdio.h>

#include "parametres.h"
#include "constoprim.h"
#include "perfcnt.h"
#include "utils.h"
#include "cclock.h"

void
constoprim(const int n,
	   const int Hnxyt,
	   const int Hnvar,
	   const real_t Hsmallr,
	   const int slices, const int Hstep,
	   real_t u[Hnvar][Hstep][Hnxyt], real_t q[Hnvar][Hstep][Hnxyt],
	   real_t e[Hstep][Hnxyt])
{
    int ijmin, ijmax, IN, i, s;
    struct timespec start, end;
    // const int nxyt = Hnxyt;
    WHERE("constoprim");
#ifdef TRACKDATA
    fprintf(stderr, "Moving constoprim IN\n");
#endif
    start = cclock();
    ijmin = 0;
    ijmax = n;

#ifdef TARGETON
#pragma omp target				\
	map(u[0:Hnvar][0:Hstep][0:Hnxyt])	\
	map(q[0:Hnvar][0:Hstep][0:Hnxyt])	\
	map(e[0:Hstep][0:Hnxyt])
#endif
#ifdef LOOPFORM
#pragma omp teams loop bind(teams) private(s, i), collapse(2)
#else
#pragma omp TEAMSDIS parallel for default(none) firstprivate(ijmin, ijmax, slices, Hsmallr) private(s, i), shared(u, q, e) collapse(2)
#endif
    for (s = 0; s < slices; s++) {
	for (i = ijmin; i < ijmax; i++) {
	    real_t eken;
	    real_t qid = MAX(u[ID][s][i], Hsmallr);
	    q[ID][s][i] = qid;

	    real_t qiu = u[IU][s][i] / qid;
	    real_t qiv = u[IV][s][i] / qid;
	    q[IU][s][i] = qiu;
	    q[IV][s][i] = qiv;

	    eken = half * (Square(qiu) + Square(qiv));

	    real_t qip = u[IP][s][i] / qid - eken;
	    q[IP][s][i] = qip;
	    e[s][i] = qip;
	}
    }
    {
	int nops = slices * ((ijmax) - (ijmin));
	FLOPS(5 * nops, 3 * nops, 1 * nops, 0 * nops);
    }

    if (Hnvar > IP) {
#ifdef TARGETON
#pragma omp target				\
	map(u[0:Hnvar][0:Hstep][0:Hnxyt])	\
	map(q[0:Hnvar][0:Hstep][0:Hnxyt])	\
	map(e[0:Hstep][0:Hnxyt])
#endif
#ifdef LOOPFORM
#pragma omp teams loop bind(teams) private(s, i, IN), collapse(3)
#else
#pragma omp TEAMSDIS parallel for default(none) firstprivate(Hnvar, slices, ijmin, ijmax), private(s, i, IN), shared(u, q, e) collapse(3)
#endif
	for (IN = IP + 1; IN < Hnvar; IN++) {
	    for (s = 0; s < slices; s++) {
		for (i = ijmin; i < ijmax; i++) {
		    q[IN][s][i] = u[IN][s][i] / q[IN][s][i];
		}
	    }
	}
    }
#ifdef TRACKDATA
    fprintf(stderr, "Moving constoprim OUT\n");
#endif
    end = cclock();
    functim[TIM_CONPRI] += ccelaps(start, end);
}				// constoprim

#undef IHVW
//EOF
