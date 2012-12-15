/*
  A simple 2D hydro code
  (C) Romain Teyssier : CEA/IRFU           -- original F90 code
  (C) Pierre-Francois Lavallee : IDRIS      -- original F90 code
  (C) Guillaume Colin de Verdiere : CEA/DAM -- for the C version
  (C) Jeffrey Poznanovic : CSCS             -- for the OpenACC version
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
#include <mpi.h>

#include "parametres.h"
#include "hydro_funcs.h"
#include "vtkfile.h"
#include "compute_deltat.h"
#include "hydro_godunov.h"
#include "utils.h"
hydroparam_t H;
hydrovar_t Hv;                  // nvar
hydrovarwork_t Hvw;             // nvar
hydrowork_t Hw;
unsigned long flops = 0;
int
main(int argc, char **argv) {
  double dt = 0;
  int nvtk = 0;
  char outnum[80];
  int time_output = 0;

  // double output_time = 0.0;
  double next_output_time = 0;
  double start_time = 0, end_time = 0;
  double start_iter = 0, end_iter = 0;
  double elaps = 0;

  MPI_Init(&argc, &argv);

  start_time = cclock();
  if (H.mype == 1)
    fprintf(stdout, "Hydro starts.\n");
  process_args(argc, argv, &H);
  hydro_init(&H, &Hv);
  // PRINTUOLD(H, &Hv);
  if (H.nproc > 1)
    MPI_Barrier(MPI_COMM_WORLD);

  if (H.dtoutput > 0) {

    // outputs are in physical time not in time steps
    time_output = 1;
    next_output_time = next_output_time + H.dtoutput;
  }
  if (H.dtoutput || H.noutput)
    vtkfile(++nvtk, H, &Hv);
  if (H.mype == 1)
    fprintf(stdout, "Hydro starts main loop.\n");

  double *restrict uold = Hv.uold;
#pragma acc data copyin(uold[0:H.nvar*H.nxt*H.nyt])

  while ((H.t < H.tend) && (H.nstep < H.nstepmax)) {
    start_iter = cclock();
    outnum[0] = 0;
    flops = 0;
    if ((H.nstep % 2) == 0) {
      // if (H.mype == 1) fprintf(stdout, "Hydro computes deltat.\n");
      compute_deltat(&dt, H, &Hw, &Hv, &Hvw);
      if (H.nstep == 0) {
        dt = dt / 2.0;
      }
      if (H.nproc > 1) {
        volatile double dtmin;
        MPI_Allreduce(&dt, &dtmin, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        dt = dtmin;
      }
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
      if (flops > 0) {
        if (iter_time > 1.e-9) {
          double mflops = (double) flops / (double) 1.e+6 / iter_time;
	  sprintf(outnum, "%s (%.3fs)", outnum, iter_time);  // Flops not supported 
          //sprintf(outnum, "%s {%.3f Mflops %lu} (%.3fs)", outnum, mflops, flops, iter_time);
        }
      } else {
        if (H.nx == 400 && H.ny == 400) {       /* LM -- Got from input !! REMOVE !!  */
          flops = 31458268;
          double mflops = (double) flops / (double) 1.e+6 / iter_time;
          sprintf(outnum, "%s {~%.3f Mflops} (%.3fs)", outnum, mflops, iter_time);
        } else
          sprintf(outnum, "%s (%.3fs)", outnum, iter_time);
      }
    }
    if (time_output == 0) {
      if ((H.nstep % H.noutput) == 0) {
        #pragma acc update host(uold[0:H.nvar*H.nxt*H.nyt])
        vtkfile(++nvtk, H, &Hv);
        sprintf(outnum, "%s [%04d]", outnum, nvtk);
      }
    } else {
      if (H.t >= next_output_time) {
        #pragma acc update host(uold[0:H.nvar*H.nxt*H.nyt])
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
    fprintf(stdout, "Hydro ends in %ss (%.3lf).\n", outnum, elaps);
  MPI_Finalize();
  return 0;
}
