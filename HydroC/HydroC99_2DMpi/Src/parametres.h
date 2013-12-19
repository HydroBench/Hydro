/*
  A simple 2D hydro code
  (C) Romain Teyssier : CEA/IRFU           -- original F90 code
  (C) Pierre-Francois Lavallee : IDRIS      -- original F90 code
  (C) Guillaume Colin de Verdiere : CEA/DAM -- for the C version
*/

/*

This software is governed by the CeCILL license under French law and
abiding by the rules of distribution of free software.  You can  use, 
modify and/ or redistribute the software under the terms of the CeCILL
license as circulated by CEA, CNRS and INRIA at the following URL
"http://www.cecill.info". 

As a counterpart to the access to the source code and  rights to copy,
modify and redistribute granted by the license, users are provided only
with a limited warranty  and the software's author,  the holder of the
economic rights,  and the successive licensors  have only  limited
liability. 

In this respect, the user's attention is drawn to the risks associated
with loading,  using,  modifying and/or developing or reproducing the
software by the user in light of its specific status of free software,
that may mean  that it is complicated to manipulate,  and  that  also
therefore means  that it is reserved for developers  and  experienced
professionals having in-depth computer knowledge. Users are therefore
encouraged to load and test the software's suitability as regards their
requirements in conditions enabling the security of their systems and/or 
data to be ensured and,  more generally, to use and operate it in the 
same conditions as regards security. 

The fact that you are presently reading this means that you have had
knowledge of the CeCILL license and that you accept its terms.

*/
#ifndef PARAMETRES_H_INCLUDED
#define PARAMETRES_H_INCLUDED
extern unsigned long flops;

#ifndef PREC_SP
typedef double real_t;
#else
typedef float real_t;
#endif

typedef enum {
  XMIN_BOX, XMAX_BOX,
  YMIN_BOX, YMAX_BOX,
  UP_BOX, DOWN_BOX,
  LEFT_BOX, RIGHT_BOX,
  MAX_BOX
} Box_t;

typedef struct _hydroparam {
  int prt;

  // time control
  real_t t, tend;
  int nstep, nstepmax;
  int noutput;
  real_t dtoutput;

  // dimensions
  int imin, imax, jmin, jmax, nx, ny, nxt, nyt, nxyt, nxystep;

  /*
     nx, ny: real useful size of the domain
     i/j min, max: the total domain index range
     nxt, nyt: the total domain size (includes 2 extra layers around the domain)
     nxyt: maximum of nxt and nyt to minimize allocations
   */
  int nproc, mype;              // MPI version
  int globnx, globny;           // global size of the problem
  int box[MAX_BOX];             // our domain size and its position relative to the others

  // physics
  int nvar;
  real_t dx;
  real_t gamma;
  real_t courant_factor;
  real_t smallc, smallr;

  // numerical scheme
  int niter_riemann;
  int iorder;
  real_t slope_type;

  // char scheme[20];
  int scheme;
  int boundary_right, boundary_left, boundary_down, boundary_up;

  // test case 
  int testCase;
} hydroparam_t;

#define HSCHEME_MUSCL 1
#define HSCHEME_PLMDE 2
#define HSCHEME_COLLELA 3

// Hydrovar holds the whole 2D problem for all variables
typedef struct _hydrovar {
  real_t *uold;                 // nxt, nyt, nvar allocated as (nxt * nyt), nvar
} hydrovar_t;                   // 1:nvar
#ifndef IHv
// #define IHv(i,j,v) ((i) + (j) * H.nxt + (H.nxt * H.nyt) * (v))
#define IHv(i,j,v) ((i) + (H.nxt * (H.nyt * (v)+ (j))))
#define IHvP(i,j,v) ((i) + (j) * H->nxt + (H->nxt * H->nyt) * (v))
#endif /*  */

// work arrays along one direction for all variables
typedef struct _hydrovarwork {
  real_t *u, *q, *qxm, *qxp, *dq;       // (nxt or nyt), nvar
  real_t *qleft, *qright, *qgdnv, *flux;        // (nx+1 or ny+1), nvar
} hydrovarwork_t;               // 1:nvar
#ifndef IHvw
// #define IHvw(i,v) ((i) + (v) * H.nxyt)
// #define IHvwP(i,v) ((i) + (v) * H->nxyt)
#endif /*  */

// works arrays along one direction
typedef struct _hydrowork {
  real_t *c;                    // nxt or nyt
  real_t *e;                    // nxt or nyt
  real_t *tmpm1, *tmpm2;        // for the reduction
  // all others nx+1 or ny+1
  real_t *rl, *ul, *pl, *cl, *wl;
  real_t *rr, *ur, *pr, *cr, *wr;
  real_t *ro, *uo, *po, *co, *wo;
  real_t *rstar, *ustar, *pstar, *cstar;
  real_t *spin, *spout, *ushock;
  int *sgnm;
  int *goon; // convergence indicator for riemann
  real_t *frac, *scr, *delp, *pold;
  int *ind, *ind2;
} hydrowork_t;

// All variables are grouped in structs for clarity sake.
/*
  Warning : no global variables are declared.
  They are passed as arguments.
*/

// useful constants to force double promotion
#ifdef ALWAYS                   // HMPP
static const real_t zero = (real_t) 0.0;
static const real_t one = (real_t) 1.0;
static const real_t two = (real_t) 2.0;
static const real_t three = (real_t) 3.0;
static const real_t hundred = (real_t) 100.0;
static const real_t two3rd = (real_t) 2.0 / (real_t) 3.0;
static const real_t half = (real_t) 1.0 / (real_t) 2.0;
static const real_t third = (real_t) 1.0 / (real_t) 3.0;
static const real_t forth = (real_t) 1.0 / (real_t) 4.0;
static const real_t sixth = (real_t) 1.0 / (real_t) 6.0;

// conservative variables with C indexing
static const int ID = 1 - 1;
static const int IU = 2 - 1;
static const int IV = 3 - 1;
static const int IP = 4 - 1;

// The current scheme ahs two extra layers around the domain.
static const int ExtraLayer = 2;
static const int ExtraLayerTot = 2 * 2;

#else /*  */
#define zero   ((real_t) 0.0)
#define one    ((real_t) 1.0)
#define two    ((real_t) 2.0)
#define three  ((real_t) 3.0)
#define hundred  ((real_t) 100.0)
#define two3rd ((real_t) 2.0 / (real_t) 3.0)
#define half   ((real_t) 1.0 / (real_t) 2.0)
#define third  ((real_t) 1.0 / (real_t) 3.0)
#define forth  ((real_t) 1.0 / (real_t) 4.0)
#define sixth  ((real_t) 1.0 / (real_t) 6.0)
#define ID     (0)
#define IU     (1)
#define IV     (2)
#define IP     (3)
#define ExtraLayer    (2)
#define ExtraLayerTot (2 * 2)
#endif /*  */
void process_args(int argc, char **argv, hydroparam_t * H);


typedef enum {
  TIM_GATCON,
  TIM_CONPRI,
  TIM_EOS,
  TIM_SLOPE,
  TIM_TRACE,
  TIM_QLEFTR,
  TIM_RIEMAN,
  TIM_CMPFLX,
  TIM_UPDCON,
  TIM_COMPDT,
  TIM_MAKBOU,
  TIM_ALLRED,
  TIM_END
} Timers_t;

extern double functim[TIM_END];

#endif // PARAMETRES_H_INCLUDED
//EOF
