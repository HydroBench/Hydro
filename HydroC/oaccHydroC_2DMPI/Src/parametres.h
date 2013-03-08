
#ifndef PARAMETRES_H_INCLUDED
#define PARAMETRES_H_INCLUDED
extern unsigned long flops;


#ifdef USE_DOUBLE
typedef double real;
#else
typedef float real;
#endif



typedef enum
{
  XMIN_BOX, XMAX_BOX,
  YMIN_BOX, YMAX_BOX,
  UP_BOX, DOWN_BOX,
  LEFT_BOX, RIGHT_BOX,
  MAX_BOX
} Box_t;

typedef struct _hydroparam
{
  int prt;

  // time control
  double t, tend;
  int nstep, nstepmax;
  int noutput;
  double dtoutput;

  // dimensions
  int imin, imax, jmin, jmax, nx, ny, nxt, nyt, nxyt, nxystep;

  /*
     nx, ny: real useful size of the domain
     i/j min, max: the total domain index range
     nxt, nyt: the total domain size (includes 2 extra layers around the domain)
     nxyt: maximum of nxt and nyt to minimize allocations
   */
  int nproc, mype;		// MPI version
  int globnx, globny;		// global size of the problem
  int box[MAX_BOX];		// our domain size and its position relative to the others

  // physics
  int nvar;
  real dx;
  real gamma;
  real courant_factor;
  real smallc, smallr;

  // numerical scheme
  int niter_riemann;
  int iorder;
  real slope_type;

  // char scheme[20];
  int scheme;
  int boundary_right, boundary_left, boundary_down, boundary_up;
} hydroparam_t;


#define HSCHEME_MUSCL 1
#define HSCHEME_PLMDE 2
#define HSCHEME_COLLELA 3

// Hydrovar holds the whole 2D problem for all variables
typedef struct _hydrovar
{
  real *uold;			// nxt, nyt, nvar allocated as (nxt * nyt), nvar
} hydrovar_t;			// 1:nvar
#ifndef IHv
// #define IHv(i,j,v) ((i) + (j) * H.nxt + (H.nxt * H.nyt) * (v))
#define IHv(i,j,v) ((i) + (H.nxt * (H.nyt * (v)+ (j))))
#define IHvP(i,j,v) ((i) + (j) * H->nxt + (H->nxt * H->nyt) * (v))
#endif /*  */

// work arrays along one direction for all variables
typedef struct _hydrovarwork
{
  real *u, *q, *qxm, *qxp, *dq;	// (nxt or nyt), nvar
  real *qleft, *qright, *qgdnv, *flux;	// (nx+1 or ny+1), nvar
} hydrovarwork_t;		// 1:nvar
#ifndef IHvw
// #define IHvw(i,v) ((i) + (v) * H.nxyt)
// #define IHvwP(i,v) ((i) + (v) * H->nxyt)
#endif /*  */

// works arrays along one direction
typedef struct _hydrowork
{
  real *c;			// nxt or nyt
  real *e;			// nxt or nyt
  // all others nx+1 or ny+1
  real *rl, *ul, *pl, *cl, *wl;
  real *rr, *ur, *pr, *cr, *wr;
  real *ro, *uo, *po, *co, *wo;
  real *rstar, *ustar, *pstar, *cstar;
  real *spin, *spout, *ushock;
  int *sgnm;
  real *frac, *scr, *delp, *pold;
  int *ind, *ind2;
} hydrowork_t;

// All variables are grouped in structs for clarity sake.
/*
  Warning : no global variables are declared.
  They are passed as arguments.
*/

// useful constants to force double promotion
#ifdef ALWAYS			// HMPP
static const real zero = (real) 0.0;
static const real one = (real) 1.0;
static const real two = (real) 2.0;
static const real three = (real) 3.0;
static const real hundred = (real) 100.0;
static const real two3rd = (real) 2.0 / (real) 3.0;
static const real half = (real) 1.0 / (real) 2.0;
static const real third = (real) 1.0 / (real) 3.0;
static const real forth = (real) 1.0 / (real) 4.0;
static const real sixth = (real) 1.0 / (real) 6.0;

// conservative variables with C indexing
static const int ID = 1 - 1;
static const int IU = 2 - 1;
static const int IV = 3 - 1;
static const int IP = 4 - 1;

// The current scheme ahs two extra layers around the domain.
static const int ExtraLayer = 2;
static const int ExtraLayerTot = 2 * 2;

#else /*  */
#define zero   ((real) 0.0)
#define one    ((real) 1.0)
#define two    ((real) 2.0)
#define three  ((real) 3.0)
#define hundred  ((real) 100.0)
#define two3rd ((real) 2.0 / (real) 3.0)
#define half   ((real) 1.0 / (real) 2.0)
#define third  ((real) 1.0 / (real) 3.0)
#define forth  ((real) 1.0 / (real) 4.0)
#define sixth  ((real) 1.0 / (real) 6.0)
#define ID     (0)
#define IU     (1)
#define IV     (2)
#define IP     (3)
#define ExtraLayer    (2)
#define ExtraLayerTot (2 * 2)
#endif /*  */
void process_args (int argc, char **argv, hydroparam_t * H);

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
