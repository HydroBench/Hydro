/*
  A simple 2D hydro code
  (C) Romain Teyssier : CEA/IRFU           -- original F90 code
  (C) Pierre-Francois Lavallee : IDRIS      -- original F90 code
  (C) Guillaume Colin de Verdiere : CEA/DAM -- for the C version
*/
#include <stdio.h>
#include <time.h>
#include <mpi.h>

#include "parametres.h"
#include "hydro_funcs.h"
#include "vtkfile.h"
#include "compute_deltat.h"
#include "hydro_godunov.h"
#include "utils.h"
#include "cclock.h"
hydroparam_t H;
hydrovar_t Hv;			// nvar
hydrovarwork_t Hvw;		// nvar
hydrowork_t Hw;
double functim[TIM_END];
int sizeLabel(double *tim, const int N) {
  double maxi = 0;
  int i;

  for (i = 0; i < N; i++)
    if (maxi < tim[i]) maxi = tim[i];

  // if (maxi < 100) return 8;
  // if (maxi < 1000) return 9;
  // if (maxi < 10000) return 10;
  return 9;
}
void percentTimings(double *tim, const int N)
{
  double sum = 0;
  int i;

  for (i = 0; i < N; i++)
    sum += tim[i];

  for (i = 0; i < N; i++)
    tim[i] = 100.0 * tim[i] / sum;
}

void avgTimings(double *tim, const int N, const int nbr)
{
  int i;

  for (i = 0; i < N; i++)
    tim[i] = tim[i] / nbr;
}

void printTimings(double *tim, const int N, const int sizeFmt)
{
  double sum = 0;
  int i;
  char fmt[256];

  sprintf(fmt, "%%-%dlf ", sizeFmt);

  for (i = 0; i < N; i++)
    fprintf(stdout, fmt, tim[i]);
}
void printTimingsLabel(const int N, const int fmtSize)
{
  int i;
  char *txt;
  char fmt[256];

  sprintf(fmt, "%%-%ds ", fmtSize);
  for (i = 0; i < N; i++) {
    switch(i) {
    case TIM_COMPDT: txt = "COMPDT"; break;
    case TIM_MAKBOU: txt = "MAKBOU"; break;
    case TIM_GATCON: txt = "GATCON"; break;
    case TIM_CONPRI: txt = "CONPRI"; break;
    case TIM_EOS: txt = "EOS"; break;
    case TIM_SLOPE: txt = "SLOPE"; break;
    case TIM_TRACE: txt = "TRACE"; break;
    case TIM_QLEFTR: txt = "QLEFTR"; break;
    case TIM_RIEMAN: txt = "RIEMAN"; break;
    case TIM_CMPFLX: txt = "CMPFLX"; break;
    case TIM_UPDCON: txt = "UPDCON"; break;
    case TIM_ALLRED: txt = "ALLRED"; break;
    default:;
    }
    fprintf(stdout, fmt, txt);
  }
}

unsigned long flops = 0;
int
main (int argc, char **argv)
{
  hydro_real_t dt = 0;
  int nvtk = 0;
  char outnum[80];
  int time_output = 0;

  // double output_time = 0.0;
  hydro_real_t next_output_time = zero;
  double start_time = 0, start_time_2=0, end_time = 0;
  double start_iter = 0, end_iter = 0;
  double elaps = 0;
  struct timespec start, end;

  // array of timers to profile the code
  memset(functim, 0, TIM_END * sizeof(functim[0]));

  MPI_Init (&argc, &argv);

  start_time = dcclock ();
  if (H.mype == 1)
    fprintf (stdout, "Hydro starts.\n");
  process_args (argc, argv, &H);
  hydro_init (&H, &Hv);
  // PRINTUOLD(H, &Hv);
  if (H.nproc > 1)
    MPI_Barrier (MPI_COMM_WORLD);

  if (H.dtoutput > 0)
    {
      // outputs are in physical time not in time steps
      time_output = 1;
      next_output_time = next_output_time + H.dtoutput;
    }
  if (H.dtoutput || H.noutput)
    vtkfile (++nvtk, H, &Hv);
  if (H.mype == 1)
    fprintf (stdout, "Hydro starts main loop.\n");
    
    
  //Data management tweaking
  hydro_real_t (*e)[H.nxyt];
  hydro_real_t (*flux)[H.nxystep][H.nxyt];
  hydro_real_t (*qleft)[H.nxystep][H.nxyt];
  hydro_real_t (*qright)[H.nxystep][H.nxyt];
  hydro_real_t (*c)[H.nxyt];
  hydro_real_t *uold;
  int (*sgnm)[H.nxyt];
  hydro_real_t (*qgdnv)[H.nxystep][H.nxyt];
  hydro_real_t (*u)[H.nxystep][H.nxyt];
  hydro_real_t (*qxm)[H.nxystep][H.nxyt];
  hydro_real_t (*qxp)[H.nxystep][H.nxyt];
  hydro_real_t (*q)[H.nxystep][H.nxyt];
  hydro_real_t (*dq)[H.nxystep][H.nxyt];
  
  start = cclock();
  allocate_work_space (H.nxyt, H, &Hw, &Hvw);
  end = cclock();
  if (H.mype == 0) fprintf(stdout, "Hydro: init mem %lfs\n", ccelaps(start, end));
  
  
  uold = Hv.uold;
  qgdnv = (hydro_real_t (*)[H.nxystep][H.nxyt]) Hvw.qgdnv;
  flux = (hydro_real_t (*)[H.nxystep][H.nxyt]) Hvw.flux;
  c = (hydro_real_t (*)[H.nxyt]) Hw.c;
  e = (hydro_real_t (*)[H.nxyt]) Hw.e;
  qleft = (hydro_real_t (*)[H.nxystep][H.nxyt]) Hvw.qleft;
  qright = (hydro_real_t (*)[H.nxystep][H.nxyt]) Hvw.qright;
  sgnm = (int (*)[H.nxyt]) Hw.sgnm;
  q = (hydro_real_t (*)[H.nxystep][H.nxyt]) Hvw.q;
  dq = (hydro_real_t (*)[H.nxystep][H.nxyt]) Hvw.dq;
  u = (hydro_real_t (*)[H.nxystep][H.nxyt]) Hvw.u;
  qxm = (hydro_real_t (*)[H.nxystep][H.nxyt]) Hvw.qxm;
  qxp = (hydro_real_t (*)[H.nxystep][H.nxyt]) Hvw.qxp;
  
  start = cclock();
#pragma acc data						\
  create(qleft[0:H.nvar], qright[0:H.nvar],			\
         q[0:H.nvar], qgdnv[0:H.nvar],				\
         flux[0:H.nvar], u[0:H.nvar],				\
         dq[0:H.nvar], e[0:H.nxystep], c[0:H.nxystep],		\
         sgnm[0:H.nxystep], qxm[0:H.nvar], qxp[0:H.nvar])	\
  copyin(uold[0:H.nvar*H.nxt*H.nyt]) 
  {
    end = cclock();
    fprintf(stdout, "Hydro %d: initialize acc %lfs\n", H.mype, ccelaps(start, end));
    start_time_2 = dcclock ();
    while ((H.t < H.tend) && (H.nstep < H.nstepmax))
      {
	start_iter = dcclock ();
	outnum[0] = 0;
	flops = 0;
	if ((H.nstep % 2) == 0)
	  {
	    // if (H.mype == 1) fprintf(stdout, "Hydro computes deltat.\n");
	    start = cclock();
	    compute_deltat (&dt, H, &Hw, &Hv, &Hvw);
	    end = cclock();
	    functim[TIM_COMPDT] += ccelaps(start, end);
	    if (H.nstep == 0){
	      dt = dt / two;
	    }
	    if (H.nproc > 1)
	      {
	      	volatile hydro_real_t dtmin;
		start = cclock();
	      	MPI_Allreduce (&dt, &dtmin, 1, MPI_DOUBLE, MPI_MIN,
			       MPI_COMM_WORLD);
		end = cclock();
		functim[TIM_ALLRED] += ccelaps(start, end);
	      	dt = dtmin;
	      }
	  }
	// if (H.mype == 1) fprintf(stdout, "Hydro starts godunov.\n");
	if ((H.nstep % 2) == 0)
	  {
	    hydro_godunov (1, dt, H, &Hv, &Hw, &Hvw);
	    //            hydro_godunov(2, dt, H, &Hv, &Hw, &Hvw);
	  }
	else
	  {
	    hydro_godunov (2, dt, H, &Hv, &Hw, &Hvw);
	    //            hydro_godunov(1, dt, H, &Hv, &Hw, &Hvw);
	  }
	end_iter = dcclock ();
	H.nstep++;
	H.t += dt;
	double iter_time = (double) (end_iter - start_iter);
	if (flops > 0) {
	  if (iter_time > 1.e-9) {
	    double mflops = (double) flops / (double) 1.e+6 / iter_time;
	    sprintf (outnum, "%s {%.3lf Mflops %lu} (%.3lfs)", outnum,
		     mflops, flops, iter_time);
	  }
	} else {
	  if (H.nx == 400 && H.ny == 400){			/* LM -- Got from input !! REMOVE !!  */
	    flops = 31458268;
	    double mflops = (double) flops / (double) 1.e+6 / iter_time;
	    sprintf (outnum, "%s {~%.3lf Mflops} (%.3lfs)", outnum, mflops,
		     iter_time);
	  } else {
	    sprintf (outnum, "%s (%.3lfs)", outnum, iter_time);
	  }
	}
	if (time_output == 0) {
	  if ((H.nstep % H.noutput) == 0)
	    {
#pragma acc update host(uold[0:H.nvar*H.nxt*H.nyt])
	      vtkfile (++nvtk, H, &Hv);
	      sprintf (outnum, "%s [%04d]", outnum, nvtk);
	    }
	} else {
	  if (time_output == 1 && H.t >= next_output_time) {
#pragma acc update host(uold[0:H.nvar*H.nxt*H.nyt])
	    vtkfile (++nvtk, H, &Hv);
	    next_output_time = next_output_time + H.dtoutput;
	    sprintf (outnum, "%s [%04d]", outnum, nvtk);
	  }
	}
	if (H.mype == 0) {
	  fprintf (stdout, "--> Step=%4d, %12.5e, %10.5e %f %s\n", H.nstep, H.t,
		   dt, dt, outnum);
	  fflush (stdout);
	}
      }
   
  }// data region
  hydro_finish (H, &Hv);
  end_time = dcclock ();
  elaps = (double) (end_time - start_time);
  timeToString (outnum, elaps);
  if (H.mype == 0){
    fprintf (stdout, "Hydro ends in %ss(%.3lf) without init: %.3lfs.\n", outnum, elaps, (double) (end_time - start_time_2));
    fprintf(stdout, "    ");
  }
  if (H.nproc == 1) {
    int sizeFmt = sizeLabel(functim, TIM_END);
    printTimingsLabel(TIM_END, sizeFmt);
    fprintf(stdout, "\n");
    fprintf(stdout, "PE0 ");
    printTimings(functim, TIM_END, sizeFmt);
    fprintf(stdout, "\n");
    fprintf(stdout, "%%   ");
    percentTimings(functim, TIM_END);
    printTimings(functim, TIM_END, sizeFmt);
    fprintf(stdout, "\n");
  }
  if (H.nproc > 1) {
    double timMAX[TIM_END];
    double timMIN[TIM_END];
    double timSUM[TIM_END];
    MPI_Allreduce(functim, timMAX, TIM_END, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(functim, timMIN, TIM_END, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(functim, timSUM, TIM_END, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    if (H.mype == 0) {
      int sizeFmt = sizeLabel(timMAX, TIM_END);
      printTimingsLabel(TIM_END, sizeFmt);
      fprintf(stdout, "\n");
      fprintf(stdout, "MIN ");
      printTimings(timMIN, TIM_END, sizeFmt);
      fprintf(stdout, "\n");
      fprintf(stdout, "MAX ");
      printTimings(timMAX, TIM_END, sizeFmt);
      fprintf(stdout, "\n");
      fprintf(stdout, "AVG ");
      avgTimings(timSUM, TIM_END, H.nproc);
      printTimings(timSUM, TIM_END, sizeFmt);
      fprintf(stdout, "\n");
    }
  }

  MPI_Finalize ();
  return 0;
}
