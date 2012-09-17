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
#ifdef _OPENMP
#include <omp.h>
#endif

#include "parametres.h"
#include "hydro_funcs.h"
#include "vtkfile.h"
#include "compute_deltat.h"
#include "hydro_godunov.h"
#include "perfcnt.h"
#include "utils.h"
hydroparam_t H;
hydrovar_t Hv;                  // nvar
hydrovarwork_t Hvw;             // nvar
hydrowork_t Hw;

int
main(int argc, char **argv) {
  double dt = 0;
  int nvtk = 0;
  char outnum[80];
  int time_output = 0;
  long flops = 0;

  // double output_time = 0.0;
  double next_output_time = 0;
  double start_time = 0, end_time = 0;
  double start_iter = 0, end_iter = 0;
  double elaps = 0;

#ifdef MPI
  MPI_Init(&argc, &argv);
#endif

  start_time = cclock();
  process_args(argc, argv, &H);
  hydro_init(&H, &Hv);

  if (H.mype == 0)
    fprintf(stdout, "Hydro starts.\n");

#ifdef _OPENMP
  if (H.mype == 0) {
    fprintf(stdout, "Hydro:    OpenMP mode ON\n");
    fprintf(stdout, "Hydro: OpenMP %d max threads\n", omp_get_max_threads());
    fprintf(stdout, "Hydro: OpenMP %d num threads\n", omp_get_num_threads());
    fprintf(stdout, "Hydro: OpenMP %d num procs\n", omp_get_num_procs());
  }
#endif
#ifdef MPI
  if (H.mype == 0) {
    fprintf(stdout, "Hydro: MPI run with %d procs\n", H.nproc);
  }
#else
  fprintf(stdout, "Hydro: standard build\n");
#endif


  // PRINTUOLD(H, &Hv);
#ifdef MPI
  if (H.nproc > 1)
    MPI_Barrier(MPI_COMM_WORLD);
#endif

  if (H.dtoutput > 0) {
    // outputs are in physical time not in time steps
    time_output = 1;
    next_output_time = next_output_time + H.dtoutput;
  }

  if (H.dtoutput > 0 || H.noutput > 0)
    vtkfile(++nvtk, H, &Hv);

  if (H.mype == 1)
    fprintf(stdout, "Hydro starts main loop.\n");
  while ((H.t < H.tend) && (H.nstep < H.nstepmax)) {
    // reset perf counter for this iteration
    flopsAri = flopsSqr = flopsMin = flopsTra = 0;
    start_iter = cclock();
    outnum[0] = 0;
    if ((H.nstep % 2) == 0) {
      dt=0;
      // if (H.mype == 1) fprintf(stdout, "Hydro computes deltat.\n");
      compute_deltat(&dt, H, &Hw, &Hv, &Hvw);
      if (H.nstep == 0) {
        dt = dt / 2.0;
      }
#ifdef MPI
      if (H.nproc > 1) {
        double dtmin;
	// printf("pe=%4d\tdt=%lg\n",H.mype, dt);
        MPI_Allreduce(&dt, &dtmin, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        dt = dtmin;
      }
#endif
    }
    // if (H.mype == 1) fprintf(stdout, "Hydro starts godunov.\n");
    if ((H.nstep % 2) == 0) {
      hydro_godunov(1, dt, H, &Hv, &Hw, &Hvw);
      //            hydro_godunov(2, dt, H, &Hv, &Hw, &Hvw);
    } else {
      hydro_godunov(2, dt, H, &Hv, &Hw, &Hvw);
      //            hydro_godunov(1, dt, H, &Hv, &Hw, &Hvw);
    }
    end_iter = cclock();
    H.nstep++;
    H.t += dt;
    {
      double iter_time = (double) (end_iter - start_iter);
#ifdef MPI
      long flopsAri_t, flopsSqr_t, flopsMin_t, flopsTra_t;
      MPI_Allreduce(&flopsAri, &flopsAri_t, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(&flopsSqr, &flopsSqr_t, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(&flopsMin, &flopsMin_t, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(&flopsTra, &flopsTra_t, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
      //       if (H.mype == 1)
      //        printf("%ld %ld %ld %ld %ld %ld %ld %ld \n", flopsAri, flopsSqr, flopsMin, flopsTra, flopsAri_t, flopsSqr_t, flopsMin_t, flopsTra_t);
      flops = flopsAri_t * FLOPSARI + flopsSqr_t * FLOPSSQR + flopsMin_t * FLOPSMIN + flopsTra_t * FLOPSTRA;
#else
      flops = flopsAri * FLOPSARI + flopsSqr * FLOPSSQR + flopsMin * FLOPSMIN + flopsTra * FLOPSTRA;
#endif
      nbFLOPS++;

      if (flops > 0) {
        if (iter_time > 1.e-9) {
          double mflops = (double) flops / (double) 1.e+6 / iter_time;
	  MflopsSUM += mflops;
          sprintf(outnum, "%s {%.2f Mflops %ld Ops} (%.3fs)", outnum, mflops, flops, iter_time);
        }
      } else {
        sprintf(outnum, "%s (%.3fs)", outnum, iter_time);
      }
    }
    if (time_output == 0 && H.noutput > 0) {
      if ((H.nstep % H.noutput) == 0) {
        vtkfile(++nvtk, H, &Hv);
        sprintf(outnum, "%s [%04d]", outnum, nvtk);
      }
    } else {
      if (time_output == 1 && H.t >= next_output_time) {
        vtkfile(++nvtk, H, &Hv);
        next_output_time = next_output_time + H.dtoutput;
        sprintf(outnum, "%s [%04d]", outnum, nvtk);
      }
    }
    if (H.mype == 0) {
      fprintf(stdout, "--> step=%4d, %12.5e, %10.5e %s\n", H.nstep, H.t, dt, outnum);
      fflush(stdout);
    }
  }
  hydro_finish(H, &Hv);
  end_time = cclock();
  elaps = (double) (end_time - start_time);
  timeToString(outnum, elaps);
  if (H.mype == 0)
    fprintf(stdout, "Hydro ends in %ss (%.3lf) <%.2lf MFlops>.\n", outnum, elaps, (float) (MflopsSUM / nbFLOPS));

#ifdef MPI
  MPI_Finalize();
#endif
  return 0;
}
