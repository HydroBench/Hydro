#ifndef PARAMETRES_H_INCLUDED
#define PARAMETRES_H_INCLUDED
extern unsigned long flops;

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
  double dx;
  double gamma;
  double courant_factor;
  double smallc, smallr;

  // numerical scheme
  int niter_riemann;
  int iorder;
  double slope_type;

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
  double *uold;			// nxt, nyt, nvar allocated as (nxt * nyt), nvar
} hydrovar_t;			// 1:nvar
#ifndef IHv
// #define IHv(i,j,v) ((i) + (j) * H.nxt + (H.nxt * H.nyt) * (v))
#define IHv(i,j,v) ((i) + (H.nxt * (H.nyt * (v)+ (j))))
#define IHvP(i,j,v) ((i) + (j) * H->nxt + (H->nxt * H->nyt) * (v))
#endif /*  */

// work arrays along one direction for all variables
typedef struct _hydrovarwork
{
  double *u, *q, *qxm, *qxp, *dq;	// (nxt or nyt), nvar
  double *qleft, *qright, *qgdnv, *flux;	// (nx+1 or ny+1), nvar
} hydrovarwork_t;		// 1:nvar
#ifndef IHvw
// #define IHvw(i,v) ((i) + (v) * H.nxyt)
// #define IHvwP(i,v) ((i) + (v) * H->nxyt)
#endif /*  */

// works arrays along one direction
typedef struct _hydrowork
{
  double *c;			// nxt or nyt
  double *e;			// nxt or nyt
  // all others nx+1 or ny+1
  double *rl, *ul, *pl, *cl, *wl;
  double *rr, *ur, *pr, *cr, *wr;
  double *ro, *uo, *po, *co, *wo;
  double *rstar, *ustar, *pstar, *cstar;
  double *spin, *spout, *ushock;
  int *sgnm;
  double *frac, *scr, *delp, *pold;
  int *ind, *ind2;
} hydrowork_t;

// All variables are grouped in structs for clarity sake.
/*
  Warning : no global variables are declared.
  They are passed as arguments.
*/

// useful constants to force double promotion
#ifdef ALWAYS			// HMPP
static const double zero = (double) 0.0;
static const double one = (double) 1.0;
static const double two = (double) 2.0;
static const double three = (double) 3.0;
static const double hundred = (double) 100.0;
static const double two3rd = (double) 2.0 / (double) 3.0;
static const double half = (double) 1.0 / (double) 2.0;
static const double third = (double) 1.0 / (double) 3.0;
static const double forth = (double) 1.0 / (double) 4.0;
static const double sixth = (double) 1.0 / (double) 6.0;

// conservative variables with C indexing
static const int ID = 1 - 1;
static const int IU = 2 - 1;
static const int IV = 3 - 1;
static const int IP = 4 - 1;

// The current scheme ahs two extra layers around the domain.
static const int ExtraLayer = 2;
static const int ExtraLayerTot = 2 * 2;

#else /*  */
#define zero   ((double) 0.0)
#define one    ((double) 1.0)
#define two    ((double) 2.0)
#define three  ((double) 3.0)
#define hundred  ((double) 100.0)
#define two3rd ((double) 2.0 / (double) 3.0)
#define half   ((double) 1.0 / (double) 2.0)
#define third  ((double) 1.0 / (double) 3.0)
#define forth  ((double) 1.0 / (double) 4.0)
#define sixth  ((double) 1.0 / (double) 6.0)
#define ID     (0)
#define IU     (1)
#define IV     (2)
#define IP     (3)
#define ExtraLayer    (2)
#define ExtraLayerTot (2 * 2)
#endif /*  */
void process_args (int argc, char **argv, hydroparam_t * H);

#endif // PARAMETRES_H_INCLUDED
//EOF
