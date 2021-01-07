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
