
#ifndef PARAMETRES_H_INCLUDED
#define PARAMETRES_H_INCLUDED
extern unsigned long flops;




//Double/single precision management
#ifdef USE_DOUBLE
typedef double hydro_real_t;
#define autocast(x) x
#else
typedef float hydro_real_t;
#define autocast(x) x ##f
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
  hydro_real_t t, tend;
  int nstep, nstepmax;
  int noutput;
  hydro_real_t dtoutput;

  // dimensions
  int imin, imax, jmin, jmax, nx, ny, nxt, nyt, nxyt, nxystep;

  /*
     nx, ny: hydro_real_t useful size of the domain
     i/j min, max: the total domain index range
     nxt, nyt: the total domain size (includes 2 extra layers around the domain)
     nxyt: maximum of nxt and nyt to minimize allocations
   */
  int nproc, mype;		// MPI version
  int globnx, globny;		// global size of the problem
  int box[MAX_BOX];		// our domain size and its position relative to the others

  // physics
  int nvar;
  hydro_real_t dx;
  hydro_real_t gamma;
  hydro_real_t courant_factor;
  hydro_real_t smallc, smallr;

  // numerical scheme
  int niter_riemann;
  int iorder;
  hydro_real_t slope_type;

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
  hydro_real_t *uold;			// nxt, nyt, nvar allocated as (nxt * nyt), nvar
} hydrovar_t;			// 1:nvar
#ifndef IHv
// #define IHv(i,j,v) ((i) + (j) * H.nxt + (H.nxt * H.nyt) * (v))
#define IHv(i,j,v) ((i) + (H.nxt * (H.nyt * (v)+ (j))))
#define IHvP(i,j,v) ((i) + (j) * H->nxt + (H->nxt * H->nyt) * (v))
#endif /*  */

// work arrays along one direction for all variables
typedef struct _hydrovarwork
{
  hydro_real_t *u, *q, *qxm, *qxp, *dq;	// (nxt or nyt), nvar
  hydro_real_t *qleft, *qright, *qgdnv, *flux;	// (nx+1 or ny+1), nvar
} hydrovarwork_t;		// 1:nvar
#ifndef IHvw
// #define IHvw(i,v) ((i) + (v) * H.nxyt)
// #define IHvwP(i,v) ((i) + (v) * H->nxyt)
#endif /*  */

// works arrays along one direction
typedef struct _hydrowork
{
  hydro_real_t *c;			// nxt or nyt
  hydro_real_t *e;			// nxt or nyt
  // all others nx+1 or ny+1
  hydro_real_t *rl, *ul, *pl, *cl, *wl;
  hydro_real_t *rr, *ur, *pr, *cr, *wr;
  hydro_real_t *ro, *uo, *po, *co, *wo;
  hydro_real_t *rstar, *ustar, *pstar, *cstar;
  hydro_real_t *spin, *spout, *ushock;
  int *sgnm;
  hydro_real_t *frac, *scr, *delp, *pold;
  int *ind, *ind2;
} hydrowork_t;

// All variables are grouped in structs for clarity sake.
/*
  Warning : no global variables are declared.
   They are passed as arguments.
*/

// useful constants to force double promotion
#ifdef ALWAYS			// HMPP
static const hydro_real_t zero = (hydro_real_t) autocast(0.0);
static const hydro_real_t one = (hydro_real_t) autocast(1.0);
static const hydro_real_t two = (hydro_real_t) autocast(2.0);
static const hydro_real_t three = (hydro_real_t) autocast(3.0);
static const hydro_real_t hundred = (hydro_real_t)autocast( 100.0);
static const hydro_real_t two3rd = (hydro_real_t) autocast(2.0) / (hydro_real_t) autocast(3.0);							       	
static const hydro_real_t half = (hydro_real_t) autocast(1.0) / (hydro_real_t) autocast(2.0);
static const hydro_real_t third = (hydro_real_t) autocast(1.0f / (hydro_real_t) autocast(3.0);
							  static const hydro_real_t forth = (hydro_real_t) autocast(1.0f / (hydro_real_t) autocast(4.0);
														    static const hydro_real_t sixth = (hydro_real_t) autocast(1.0) / (hydro_real_t) autocast(6.0);

// conservative variables with C indexing
static const int ID = 1 - 1;
static const int IU = 2 - 1;
static const int IV = 3 - 1;
static const int IP = 4 - 1;

// The current scheme ahs two extra layers around the domain.
static const int ExtraLayer = 2;
static const int ExtraLayerTot = 2 * 2;

#else /*  */
#define zero   ((hydro_real_t) autocast(0.0))
#define one    ((hydro_real_t) autocast(1.0))
#define two    ((hydro_real_t) autocast(2.0))
#define three  ((hydro_real_t) autocast(3.0))
#define hundred  ((hydro_real_t) autocast(100.0))
#define two3rd ((hydro_real_t) autocast(2.0) / (hydro_real_t)autocast( 3.0))
#define half   ((hydro_real_t) autocast(1.0) / (hydro_real_t) autocast(2.0))
#define third  ((hydro_real_t) autocast(1.0) / (hydro_real_t) autocast(3.0))
#define forth  ((hydro_real_t) autocast(1.0) / (hydro_real_t) autocast(4.0))
#define sixth  ((hydro_real_t) autocast(1.0) / (hydro_real_t) autocast(6.0))
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
