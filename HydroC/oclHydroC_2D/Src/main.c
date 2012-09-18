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
#include <mpi.h>

#include "parametres.h"
#include "hydro_funcs.h"
#include "vtkfile.h"
#include "oclComputeDeltat.h"
#include "oclHydroGodunov.h"
#include "utils.h"
#include "oclInit.h"

hydroparam_t H;
hydrovar_t Hv;                  // nvar
hydrovarwork_t Hvw;             // nvar
hydrowork_t Hw;
unsigned long flops = 0;
int
main(int argc, char **argv)
{
  double dt = 0;
  long nvtk = 0;
  char outnum[80];
  long time_output = 0;

  // double output_time = 0.0;
  double next_output_time = 0;
  double start_time = 0, end_time = 0;
  double start_iter = 0, end_iter = 0;
  double elaps = 0;
  char cdt;
  start_time = cclock();

  MPI_Init(&argc, &argv);
  process_args(argc, argv, &H);
  if (H.mype == 0) {
    fprintf(stdout, "Hydro starts.\n");
    fflush(stdout);
  }
  
  MPI_Barrier(MPI_COMM_WORLD);

  oclInitCode(H.nproc, H.mype);
  MPI_Barrier(MPI_COMM_WORLD);

  hydro_init(&H, &Hv);
  // PRINTUOLD(stdout, H, &Hv);

  // Allocate work space for 1D sweeps
  allocate_work_space(H, &Hw, &Hvw);
  oclAllocOnDevice(H);

  // vtkfile(nvtk, H, &Hv);
  if (H.dtoutput > 0) {

    // outputs are in physical time not in time steps
    time_output = 1;
    next_output_time = next_output_time + H.dtoutput;
  }
  oclPutUoldOnDevice(H, &Hv);

  if (H.dtoutput > 0 || H.noutput > 0)
    vtkfile(++nvtk, H, &Hv);

  if (H.mype == 1)
    fprintf(stdout, "Hydro starts main loop.\n");

  while ((H.t < H.tend) && (H.nstep < H.nstepmax)) {
    start_iter = cclock();
    outnum[0] = 0;
    flops = 0;
    cdt = ' ';
    if ((H.nstep % 2) == 0) {
      oclComputeDeltat(&dt, H, &Hw, &Hv, &Hvw);
      cdt = '*';
      // fprintf(stdout, "dt=%lg\n", dt);
      if (H.nstep == 0) {
        dt = dt / 2.0;
      }
      if (H.nproc > 1) {
	double dtmin;
	int uno = 1;
	MPI_Allreduce(&dt, &dtmin, uno, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
	dt = dtmin;
      }
    }
    if ((H.nstep % 2) == 0) {
      oclHydroGodunov(1, dt, H, &Hv, &Hw, &Hvw);
    } else {
      oclHydroGodunov(2, dt, H, &Hv, &Hw, &Hvw);
    }
    end_iter = cclock();
    H.nstep++;
    H.t += dt;
    if (flops > 0) {
      double iter_time = (double) (end_iter - start_iter);
      if (iter_time > 1.e-9) {
        double mflops = (double) flops / (double) 1.e+6 / iter_time;
        sprintf(outnum, "%s {%.3f Mflops} (%.3fs)", outnum, mflops, iter_time);
      }
    } else {
      double iter_time = (double) (end_iter - start_iter);
      sprintf(outnum, "%s (%.3fs)", outnum, iter_time);
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
    if (H.mype == 0) fprintf(stdout, "--> step=%-4ld %12.5e, %10.5e %s %c\n", H.nstep, H.t, dt, outnum, cdt);
  }

  hydro_finish(H, &Hv);
  oclFreeOnDevice();
  // Deallocate work space
  deallocate_work_space(H, &Hw, &Hvw);

  end_time = cclock();
  elaps = (double) (end_time - start_time);
  timeToString(outnum, elaps);
  if (H.mype == 0) fprintf(stdout, "Hydro ends in %ss (%.3lf).\n", outnum, elaps);
  return 0;
}
