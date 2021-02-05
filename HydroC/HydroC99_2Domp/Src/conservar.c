#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <stdio.h>

#include "parametres.h"
#include "utils.h"
#include "conservar.h"
#include "perfcnt.h"
#include "cclock.h"

#define BLOCKING 0
#define SSST 32
#define JJST 32

void
gatherConservativeVars(const int idim,
		       const int rowcol,
		       const int Himin,
		       const int Himax,
		       const int Hjmin,
		       const int Hjmax,
		       const int Hnvar,
		       const int Hnxt,
		       const int Hnyt,
		       const int Hnxyt,
		       const int slices, const int Hstep,
		       real_t uold[Hnvar * Hnxt * Hnyt],
		       real_t u[Hnvar][Hstep][Hnxyt]
    )
{
    int i, j, ivar, s;
    struct timespec start, end;

#define IHU(i, j, v)  ((i) + Hnxt  * ((j) + Hnyt  * (v)))
#define IHST(v,s,i)   ((i) + Hstep * ((j) + Hnvar * (v)))

    WHERE("gatherConservativeVars");
    start = cclock();

    if (idim == 1) {
	// Gather conservative variables
#ifdef TRACKDATA
	fprintf(stderr, "Moving gatherConservativeVars IN\n");
#endif
	
#ifdef TARGETON
#pragma omp target				\
	map(u[0:Hnvar][0:Hstep][0:Hnxyt])	\
	map(uold[0:Hnvar *Hnxt * Hnyt])
#endif
#pragma omp TEAMSDIS parallel for default(none)		\
	private(s, i), \
	firstprivate(slices, Himin, Himax, rowcol, Hnxt, Hnyt)	\
	shared(u, uold) collapse(2)
	for (s = 0; s < slices; s++) {
	    for (i = Himin; i < Himax; i++) {
		int idxuoID = IHU(i, rowcol + s, ID);
		u[ID][s][i] = uold[idxuoID];

		int idxuoIU = IHU(i, rowcol + s, IU);
		u[IU][s][i] = uold[idxuoIU];

		int idxuoIV = IHU(i, rowcol + s, IV);
		u[IV][s][i] = uold[idxuoIV];

		int idxuoIP = IHU(i, rowcol + s, IP);
		u[IP][s][i] = uold[idxuoIP];
	    }
	}

	if (Hnvar > IP) {
#ifdef TRACKDATA
	    fprintf(stderr, "Moving gatherConservativeVars IN\n");
#endif
	    
#ifdef TARGETON
#pragma omp target				\
    map(u[0:Hnvar][0:Hstep][0:Hnxyt])		\
    map(uold[0:Hnvar *Hnxt * Hnyt])
#endif
#pragma omp TEAMSDIS parallel for default(none)			\
    firstprivate(Hnvar, slices, Himin, Himax, rowcol, Hnxt, Hnyt),	\
    private(s, i, ivar), shared(u, uold) collapse(3)
	    for (ivar = IP + 1; ivar < Hnvar; ivar++) {
		for (s = 0; s < slices; s++) {
		    for (i = Himin; i < Himax; i++) {
			u[ivar][s][i] = uold[IHU(i, rowcol + s, ivar)];
		    }
		}
	    }
	}
    } else {
	// Gather conservative variables
#ifdef TARGETON
#pragma omp target				\
	map(u[0:Hnvar][0:Hstep][Himin:Himax])	\
	map(uold[0:Hnvar * Hnxt * Hnyt])
#endif
#pragma omp TEAMSDIS parallel for default(none) \
 	firstprivate(Hnvar, slices, Hjmin, Hjmax, rowcol, Hnxt, Hnyt),	\
	private(s, j), shared(u, uold) collapse(2)
	for (s = 0; s < slices; s++) {
	    for (j = Hjmin; j < Hjmax; j++) {
		u[ID][s][j] = uold[IHU(rowcol + s, j, ID)];
		u[IU][s][j] = uold[IHU(rowcol + s, j, IV)];
		u[IV][s][j] = uold[IHU(rowcol + s, j, IU)];
		u[IP][s][j] = uold[IHU(rowcol + s, j, IP)];
	    }
	}

	if (Hnvar > IP) {
#ifdef TRACKDATA
	    fprintf(stderr, "Moving gatherConservativeVars IN\n");
#endif
	    
#ifdef TARGETON
#pragma omp target				\
	map(u[0:Hnvar][0:Hstep][0:Hnxyt])	\
	map(uold[0:Hnvar *Hnxt * Hnyt])
#endif
#pragma omp TEAMSDIS parallel for default(none)\
	firstprivate(Hnvar, slices, Hjmin, Hjmax, rowcol, Hnxt, Hnyt),	       \
        private(s, j, ivar), shared(u, uold) collapse(3)
	    for (ivar = IP + 1; ivar < Hnvar; ivar++) {
		for (s = 0; s < slices; s++) {
		    for (j = Hjmin; j < Hjmax; j++) {
			u[ivar][s][j] = uold[IHU(rowcol + s, j, ivar)];
		    }
		}
	    }
	}
    }
#ifdef TRACKDATA
    fprintf(stderr, "Moving gatherConservativeVars OUT\n");
#endif
    
    end = cclock();
    functim[TIM_GATCON] += ccelaps(start, end);
}

#undef IHU

void
updateConservativeVars(const int idim,
		       const int rowcol,
		       const real_t dtdx,
		       const int Himin,
		       const int Himax,
		       const int Hjmin,
		       const int Hjmax,
		       const int Hnvar,
		       const int Hnxt,
		       const int Hnyt,
		       const int Hnxyt,
		       const int slices, const int Hstep,
		       real_t uold[Hnvar * Hnxt * Hnyt],
		       real_t u[Hnvar][Hstep][Hnxyt],
		       real_t flux[Hnvar][Hstep][Hnxyt]
    )
{
    int i, j, ivar, s;
    struct timespec start, end;
    WHERE("updateConservativeVars");
#ifdef TRACKDATA
    fprintf(stderr, "Moving updateConservativeVars IN\n");
#endif
    start = cclock();

#define IHU(i, j, v)  ((i) + Hnxt * ((j) + Hnyt * (v)))

    if (idim == 1) {

	// Update conservative variables
#ifdef TARGETON
#pragma omp target				\
	map(u[0:Hnvar][0:Hstep][0:Hnxyt])	\
	map(flux[0:Hnvar][0:Hstep][0:Hnxyt])	\
	map(uold[0:Hnvar * Hnxt * Hnyt])
#endif
#pragma omp TEAMSDIS parallel for \
	firstprivate(slices, Himin, Himax, rowcol, dtdx, Hnxt, Hnyt)	\
	default(none) \
	private(s, i, ivar), shared(u, uold, flux) collapse(2)
	for (s = 0; s < slices; s++) {
	    for (ivar = 0; ivar <= IP; ivar++) {
		for (i = Himin + ExtraLayer; i < Himax - ExtraLayer; i++) {
		    uold[IHU(i, rowcol + s, ivar)] =
			u[ivar][s][i] + (flux[ivar][s][i - 2] -
					 flux[ivar][s][i - 1]) * dtdx;
		}
	    }
	}

	if (Hnvar > IP) {
#ifdef TARGETON
#pragma omp target				\
	map(u[0:Hnvar][0:Hstep][0:Hnxyt])	\
	map(flux[0:Hnvar][0:Hstep][0:Hnxyt])	\
	map(uold[0:Hnvar * Hnxt * Hnyt])
#endif
#pragma omp TEAMSDIS parallel for default(none)\
	firstprivate(rowcol, Hnvar, slices, Himin, Himax, dtdx, Hnxt, Hnyt),		\
	private(s, i, ivar), shared(u, uold, flux) collapse(3)
	    for (ivar = IP + 1; ivar < Hnvar; ivar++) {
		for (s = 0; s < slices; s++) {
		    for (i = Himin + ExtraLayer; i < Himax - ExtraLayer; i++) {
			uold[IHU(i, rowcol + s, ivar)] =
			    u[ivar][s][i] + (flux[ivar][s][i - 2] -
					     flux[ivar][s][i - 1]) * dtdx;
		    }
		}
	    }
	}
	{
	    int nops =
		(IP + 1) * slices * ((Himax - ExtraLayer) -
				     (Himin + ExtraLayer));
	    FLOPS(6 * nops, 0 * nops, 0 * nops, 0 * nops);
	}
    } else {
	// Update conservative variables
#ifdef TARGETON
#pragma omp target				\
	map(u[0:Hnvar][0:Hstep][0:Hnxyt])	\
	map(flux[0:Hnvar][0:Hstep][0:Hnxyt])	\
	map(uold[0:Hnvar * Hnxt * Hnyt])
#endif
#pragma omp TEAMSDIS parallel for			\
	default(none) \
	private(s, j, ivar), \
	firstprivate(slices, Hjmin, Hjmax, dtdx, Hnxt, Hnyt, rowcol)	\
	shared(u, uold, flux) \
	collapse(2)
	for (s = 0; s < slices; s++) {
	    for (j = (Hjmin + ExtraLayer); j < (Hjmax - ExtraLayer); j++) {
		uold[IHU(rowcol + s, j, ID)] =
		    u[ID][s][j] + (flux[ID][s][j - 2] -
				   flux[ID][s][j - 1]) * dtdx;
		uold[IHU(rowcol + s, j, IV)] =
		    u[IU][s][j] + (flux[IU][s][j - 2] -
				   flux[IU][s][j - 1]) * dtdx;
		uold[IHU(rowcol + s, j, IU)] =
		    u[IV][s][j] + (flux[IV][s][j - 2] -
				   flux[IV][s][j - 1]) * dtdx;
		uold[IHU(rowcol + s, j, IP)] =
		    u[IP][s][j] + (flux[IP][s][j - 2] -
				   flux[IP][s][j - 1]) * dtdx;
	    }
	}
	{
	    int nops = slices * ((Hjmax - ExtraLayer) - (Hjmin + ExtraLayer));
	    FLOPS(12 * nops, 0 * nops, 0 * nops, 0 * nops);
	}

	if (Hnvar > IP) {
#ifdef TARGETON
#pragma omp target				\
	map(u[0:Hnvar][0:Hstep][0:Hnxyt])	\
	map(flux[0:Hnvar][0:Hstep][0:Hnxyt])	\
	map(uold[0:Hnvar * Hnxt * Hnyt])
#endif
#pragma omp TEAMSDIS parallel for default(none)\
	firstprivate(Hnvar, slices, Hjmin, Hjmax, dtdx, Hnxt, Hnyt, rowcol),	\
	private(s, j, ivar), shared(u, uold, flux) collapse(3)
	    for (ivar = IP + 1; ivar < Hnvar; ivar++) {
		for (s = 0; s < slices; s++) {
		    for (j = Hjmin + ExtraLayer; j < Hjmax - ExtraLayer; j++) {
			uold[IHU(rowcol + s, j, ivar)] =
			    u[ivar][s][j] + (flux[ivar][s][j - 2] -
					     flux[ivar][s][j - 1]) * dtdx;
		    }
		}
	    }
	}
    }
    end = cclock();
    functim[TIM_UPDCON] += ccelaps(start, end);
#ifdef TRACKDATA
    fprintf(stderr, "Moving updateConservativeVars OUT\n");
#endif
}

#undef IHU
//EOF
