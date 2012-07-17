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
hydroparam_t H;
hydrovar_t Hv;			// nvar
hydrovarwork_t Hvw;		// nvar
hydrowork_t Hw;
unsigned long flops = 0;
int
main (int argc, char **argv)
{
  double dt = 0;
  int nvtk = 0;
  char outnum[80];
  int time_output = 0;

  // double output_time = 0.0;
  double next_output_time = 0;
  double start_time = 0, start_time_2=0, end_time = 0;
  double start_iter = 0, end_iter = 0;
  double elaps = 0;

  MPI_Init (&argc, &argv);

  start_time = cclock ();
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
      double (*e)[H.nxyt];
  double (*flux)[H.nxystep][H.nxyt];
  double (*qleft)[H.nxystep][H.nxyt];
  double (*qright)[H.nxystep][H.nxyt];
  double (*c)[H.nxyt];
  double *uold;
  int (*sgnm)[H.nxyt];
  double (*qgdnv)[H.nxystep][H.nxyt];
  double (*u)[H.nxystep][H.nxyt];
  double (*qxm)[H.nxystep][H.nxyt];
  double (*qxp)[H.nxystep][H.nxyt];
  double (*q)[H.nxystep][H.nxyt];
  double (*dq)[H.nxystep][H.nxyt];
  
  allocate_work_space (H.nxyt, H, &Hw, &Hvw);
  
  
        uold = Hv.uold;
      qgdnv = (double (*)[H.nxystep][H.nxyt]) Hvw.qgdnv;
      flux = (double (*)[H.nxystep][H.nxyt]) Hvw.flux;
      c = (double (*)[H.nxyt]) Hw.c;
      e = (double (*)[H.nxyt]) Hw.e;
      qleft = (double (*)[H.nxystep][H.nxyt]) Hvw.qleft;
      qright = (double (*)[H.nxystep][H.nxyt]) Hvw.qright;
      sgnm = (int (*)[H.nxyt]) Hw.sgnm;
      q = (double (*)[H.nxystep][H.nxyt]) Hvw.q;
      dq = (double (*)[H.nxystep][H.nxyt]) Hvw.dq;
      u = (double (*)[H.nxystep][H.nxyt]) Hvw.u;
      qxm = (double (*)[H.nxystep][H.nxyt]) Hvw.qxm;
      qxp = (double (*)[H.nxystep][H.nxyt]) Hvw.qxp;
    

#pragma acc data
{
   start_time_2 = cclock ();
	
    
#pragma acc data \
  create(qleft[0:H.nvar], qright[0:H.nvar], \
         q[0:H.nvar], qgdnv[0:H.nvar], \
         flux[0:H.nvar], u[0:H.nvar], \
         dq[0:H.nvar], e[0:H.nxystep], c[0:H.nxystep], \
         sgnm[0:H.nxystep], qxm[0:H.nvar], qxp[0:H.nvar]) \
  copy(uold[0:H.nvar*H.nxt*H.nyt]) 
  while ((H.t < H.tend) && (H.nstep < H.nstepmax))
  {
      start_iter = cclock ();
      outnum[0] = 0;
      flops = 0;
      if ((H.nstep % 2) == 0)
			{
			  // if (H.mype == 1) fprintf(stdout, "Hydro computes deltat.\n");
	  		compute_deltat (&dt, H, &Hw, &Hv, &Hvw);
	  		if (H.nstep == 0){
	      	dt = dt / 2.0;
	    	}
	  		if (H.nproc > 1)
	    	{
	      	volatile double dtmin;
	      	MPI_Allreduce (&dt, &dtmin, 1, MPI_DOUBLE, MPI_MIN,
			     	MPI_COMM_WORLD);
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
      end_iter = cclock ();
      H.nstep++;
      H.t += dt;
			double iter_time = (double) (end_iter - start_iter);
			if (flops > 0) {
		  	if (iter_time > 1.e-9) {
					double mflops = (double) flops / (double) 1.e+6 / iter_time;
					sprintf (outnum, "%s {%.3f Mflops %lu} (%.3fs)", outnum,
			 			mflops, flops, iter_time);
		    }
	 	 	} else {
		  	if (H.nx == 400 && H.ny == 400){			/* LM -- Got from input !! REMOVE !!  */
					flops = 31458268;
					double mflops = (double) flops / (double) 1.e+6 / iter_time;
					sprintf (outnum, "%s {~%.3f Mflops} (%.3fs)", outnum, mflops,
			 		iter_time);
		    } else {
		      sprintf (outnum, "%s (%.3fs)", outnum, iter_time);
		    }
			}
      if (time_output == 0) {
	 			if ((H.nstep % H.noutput) == 0)
	    	{
	     		vtkfile (++nvtk, H, &Hv);
	      	sprintf (outnum, "%s [%04d]", outnum, nvtk);
	    	}
			} else {
	  		if (H.t >= next_output_time) {
	      	vtkfile (++nvtk, H, &Hv);
	     	 	next_output_time = next_output_time + H.dtoutput;
	      	sprintf (outnum, "%s [%04d]", outnum, nvtk);
	    	}
			}
      if (H.mype == 0) {
				fprintf (stdout, "--> Step=%4d, %12.5e, %10.5e %s\n", H.nstep, H.t,
					 dt, outnum);
				fflush (stdout);
			}
   }//data region
   
 }//bogus data region
  hydro_finish (H, &Hv);
  end_time = cclock ();
  elaps = (double) (end_time - start_time);
  timeToString (outnum, elaps);
  if (H.mype == 0){
    fprintf (stdout, "Hydro ends in %ss(%.3lf) without device aquirement: %.3lfs.\n", outnum, elaps, (double) (end_time - start_time_2));
  }
  MPI_Finalize ();
  return 0;
}
