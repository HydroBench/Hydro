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

#include <stdio.h>
#include <time.h>
#ifdef MPI
#include <mpi.h>
#endif

#include "parametres.h"
#include "hydro_funcs.h"
#include "vtkfile.h"
#include "oclComputeDeltat.h"
#include "oclHydroGodunov.h"
#include "utils.h"
#include "cclock.h"
#include "oclInit.h"

#ifdef NVIDIA
OclUnit_t runUnit = RUN_GPU;
#else
OclUnit_t runUnit = RUN_CPU;
#endif
hydroparam_t H;
hydrovar_t Hv;                  // nvar
hydrovarwork_t Hvw;             // nvar
hydrowork_t Hw;
unsigned long flops = 0;

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

int
main(int argc, char **argv)
{
  real_t dt = 0;
  long nvtk = 0;
  char outnum[240];
  long time_output = 0;

  // double output_time = 0.0;
  real_t next_output_time = 0;
  double start_time = 0, start_time_2 = 0, end_time = 0;
  double start_iter = 0, end_iter = 0;
  double elaps = 0;
  double avgMcps = 0;
  long nAvgMcps = 0;

  char cdt;
  struct timespec start, end;

#ifdef MPI
#pragma message "Building an MPI version"
  MPI_Init(&argc, &argv);
#endif
  start_time = dcclock ();
  process_args(argc, argv, &H);
  if (H.mype == 0) {
#ifdef MPI
    fprintf(stdout, "Hydro: build made for MPI usage\n");
#else
    fprintf(stdout, "Hydro: single node build (no MPI)\n");  
#endif
    fprintf(stdout, "Hydro starts in %s.\n", (sizeof(real_t) == sizeof(double))? "double precision": "single precision");
    fflush(stdout);
  }

#ifdef MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  oclInitCode(H.nproc, H.mype);
#ifdef MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  hydro_init(&H, &Hv);
  // PRINTUOLD(stdout, H, &Hv);

  // Allocate work space for 1D sweeps
  allocate_work_space(H, &Hw, &Hvw);
  start = cclock();
  oclAllocOnDevice(H);

  // vtkfile(nvtk, H, &Hv);
  if (H.dtoutput > 0) {

    // outputs are in physical time not in time steps
    time_output = 1;
    next_output_time = next_output_time + H.dtoutput;
  }
  oclPutUoldOnDevice(H, &Hv);
  end = cclock();
  fprintf(stdout, "Hydro %d: initialize acc %lfs\n", H.mype, ccelaps(start, end));

  if (H.dtoutput > 0 || H.noutput > 0)
    vtkfile(++nvtk, H, &Hv);

  if (H.mype == 1)
    fprintf(stdout, "Hydro starts main loop.\n");

  start_time_2 = dcclock();
  while ((H.t < H.tend) && (H.nstep < H.nstepmax)) {
	  double iter_time = 0;
    start_iter = dcclock();
    outnum[0] = 0;
    flops = 0;
    cdt = ' ';
    if ((H.nstep % 2) == 0) {
      start = cclock();
      oclComputeDeltat(&dt, H, &Hw, &Hv, &Hvw);
      end = cclock();
      functim[TIM_COMPDT] += ccelaps(start, end);
      cdt = '*';
      // fprintf(stdout, "dt=%lg\n", dt);
      if (H.nstep == 0) {
        dt = dt / 2.0;
      }
      if (H.nproc > 1) {
#ifdef MPI
	real_t dtmin;
	int uno = 1;
	start = cclock();
	MPI_Allreduce(&dt, &dtmin, uno, 
		      (sizeof(real_t) == sizeof(double))? MPI_DOUBLE: MPI_FLOAT, 
		       MPI_MIN, MPI_COMM_WORLD);
	end = cclock();
	functim[TIM_ALLRED] += ccelaps(start, end);
	dt = dtmin;
#endif
      }
    }
    if ((H.nstep % 2) == 0) {
      oclHydroGodunov(1, dt, H, &Hv, &Hw, &Hvw);
    } else {
      oclHydroGodunov(2, dt, H, &Hv, &Hw, &Hvw);
    }
    end_iter = dcclock();
    H.nstep++;
    H.t += dt;
    iter_time = (double) (end_iter - start_iter);
    if (flops > 0) {
      if (iter_time > 1.e-9) {
        double mflops = (double) flops / (double) 1.e+6 / iter_time;
        sprintf(outnum, "%s {%.3f Mflops} (%.3fs)", outnum, mflops, iter_time);
      }
    } else {
      sprintf(outnum, "%s (%.3fs)", outnum, iter_time);
    }
    if (iter_time > 1.e-9) {
	    double mcps = ((double) H.globnx * (double) H.globny) / iter_time / 1e6l;
	    if (H.nstep > 5) {
		    sprintf(outnum, "%s (%.1lf MC/s)", outnum, mcps);
		    nAvgMcps++;
		    avgMcps += mcps;
	    }
    }
    if (time_output == 0 && H.noutput > 0) {
      if ((H.nstep % H.noutput) == 0) {
        oclGetUoldFromDevice(H, &Hv);
        vtkfile(++nvtk, H, &Hv);
        sprintf(outnum, "%s [%04ld]", outnum, nvtk);
      }
    } else {
      if (time_output == 1 && H.t >= next_output_time) {
        oclGetUoldFromDevice(H, &Hv);
        vtkfile(++nvtk, H, &Hv);
        next_output_time = next_output_time + H.dtoutput;
        sprintf(outnum, "%s [%04ld]", outnum, nvtk);
      }
    }
    if (H.mype == 0) {
      fprintf(stdout, "--> step=%-4ld %12.5e, %10.5e %s %c\n", H.nstep, H.t, dt, outnum, cdt);
    }
  }

  hydro_finish(H, &Hv);
  end_time = dcclock();
  elaps = (double) (end_time - start_time);
  timeToString(outnum, elaps);  
  if (H.mype == 0) {
    fprintf(stdout, "Hydro ends in %ss(%.3lf) without init: %.3lfs. [%s]\n", outnum, elaps, (double) (end_time - start_time_2), (sizeof(real_t) == sizeof(double))? "DP": "SP");
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
#ifdef MPI
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
#endif
  }

  oclFreeOnDevice();
  oclCloseupCode();
  // Deallocate work space
  deallocate_work_space(H, &Hw, &Hvw);
  if (H.mype == 0) {
	  avgMcps /= nAvgMcps;
	  fprintf(stdout, "Average MC/s: %.1lf\n", avgMcps);
  }
#ifdef MPI
  MPI_Finalize();
#endif
  return 0;
}
