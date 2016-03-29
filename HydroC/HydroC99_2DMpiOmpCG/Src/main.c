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
#include <string.h>
#include <values.h>
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
#include "cclock.h"
#include "utils.h"

hydroparam_t H;
hydrovar_t Hv;			// nvar
//for compute_delta
hydrovarwork_t Hvw_deltat;	// nvar
hydrowork_t Hw_deltat;
hydrovarwork_t Hvw_godunov;	// nvar
hydrowork_t Hw_godunov;
double functim[TIM_END];

int sizeLabel(double *tim, const int N)
{
	double maxi = 0;
	int i;

	for (i = 0; i < N; i++)
		if (maxi < tim[i])
			maxi = tim[i];

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
		switch (i) {
		case TIM_COMPDT:
			txt = "COMPDT";
			break;
		case TIM_MAKBOU:
			txt = "MAKBOU";
			break;
		case TIM_GATCON:
			txt = "GATCON";
			break;
		case TIM_CONPRI:
			txt = "CONPRI";
			break;
		case TIM_EOS:
			txt = "EOS";
			break;
		case TIM_SLOPE:
			txt = "SLOPE";
			break;
		case TIM_TRACE:
			txt = "TRACE";
			break;
		case TIM_QLEFTR:
			txt = "QLEFTR";
			break;
		case TIM_RIEMAN:
			txt = "RIEMAN";
			break;
		case TIM_CMPFLX:
			txt = "CMPFLX";
			break;
		case TIM_UPDCON:
			txt = "UPDCON";
			break;
		case TIM_ALLRED:
			txt = "ALLRED";
			break;
		default:;
		}
		fprintf(stdout, fmt, txt);
	}
}

int main(int argc, char **argv)
{
	real_t dt = 0;
	int nvtk = 0;
	char outnum[80];
	int time_output = 0;
	long flops = 0;

	// real_t output_time = 0.0;
	real_t next_output_time = 0;
	double start_time = 0, end_time = 0;
	double start_iter = 0, end_iter = 0;
	double elaps = 0;
	double avgMCbys = 0, aMCs = 0;
	int nbMCbys = 0;
	struct timespec start, end;

	// array of timers to profile the code
	memset(functim, 0, TIM_END * sizeof(functim[0]));

#ifdef MPI
	MPI_Init(&argc, &argv);
#endif

	process_args(argc, argv, &H);
	hydro_init(&H, &Hv);

	if (H.mype == 0)
		fprintf(stdout, "Hydro starts in %s precision.\n", ((sizeof(real_t) == sizeof(double)) ? "double" : "single"));

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

	if (H.mype == 0)
		fprintf(stdout, "Hydro starts main loop.\n");

	//pre-allocate memory before entering in loop
	//For godunov scheme
	start = cclock();
	start = cclock();
	allocate_work_space(H.nxyt, H, &Hw_godunov, &Hvw_godunov);
	compute_deltat_init_mem(H, &Hw_deltat, &Hvw_deltat);
	end = cclock();
	if (H.mype == 0)
		fprintf(stdout, "Hydro: init mem %lfs\n", ccelaps(start, end));
	// we start timings here to avoid the cost of initial memory allocation
	start_time = dcclock();
	{
		while ((H.t < H.tend) && (H.nstep < H.nstepmax)) {
			// reset perf counter for this iteration
			flopsAri = flopsSqr = flopsMin = flopsTra = 0;
			start_iter = dcclock();
			outnum[0] = 0;
			if ((H.nstep % 2) == 0) {
				dt = FLT_MAX;
				// if (H.mype == 0) fprintf(stdout, "Hydro computes local deltat %lg.\n", ldt);
				start = cclock();
				compute_deltat(&dt, H, &Hw_deltat, &Hv, &Hvw_deltat);
				// if (H.mype == 0) fprintf(stdout, "Hydro computes local deltat %lg.\n", ldt);
				end = cclock();
				functim[TIM_COMPDT] += ccelaps(start, end);
				if (H.nstep == 0) {
					dt = dt / 2.0;
				}
#ifdef MPI
				if (H.nproc > 1) {
					real_t dtmin = FLT_MAX;
					// printf("pe=%4d\tdt=%lg\n",H.mype, dt);
					if (sizeof(real_t) == sizeof(double)) {
						MPI_Allreduce(&dt, &dtmin, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
					} else {
						MPI_Allreduce(&dt, &dtmin, 1, MPI_FLOAT, MPI_MIN, MPI_COMM_WORLD);
					}
					// fprintf(stdout, "Hydro computes deltat MPI min %lg.\n", dtmin);
					dt = dtmin;
				}
#endif
			}
			// if (H.mype == 1) fprintf(stdout, "Hydro starts godunov.\n");
			if ((H.nstep % 2) == 0) {
				hydro_godunov(1, dt, H, &Hv, &Hw_godunov, &Hvw_godunov);
				//            hydro_godunov(2, dt, H, &Hv, &Hw, &Hvw);
			} else {
				hydro_godunov(2, dt, H, &Hv, &Hw_godunov, &Hvw_godunov);
				//            hydro_godunov(1, dt, H, &Hv, &Hw, &Hvw);
			}
			end_iter = dcclock();
			{
				H.nstep++;
				H.t += dt;
			}
			{
				real_t iter_time = (real_t) (end_iter - start_iter);
				avgMCbys = ((H.globnx * H.globny) / iter_time / 1000000);
				aMCs += avgMCbys;
				nbMCbys += 1;
#ifdef MPIO
				{
					long flopsAri_t, flopsSqr_t, flopsMin_t, flopsTra_t;
					start = cclock();
					MPI_Allreduce(&flopsAri, &flopsAri_t, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
					MPI_Allreduce(&flopsSqr, &flopsSqr_t, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
					MPI_Allreduce(&flopsMin, &flopsMin_t, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
					MPI_Allreduce(&flopsTra, &flopsTra_t, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
					//       if (H.mype == 1)
					//        printf("%ld %ld %ld %ld %ld %ld %ld %ld \n", flopsAri, flopsSqr, flopsMin, flopsTra, flopsAri_t, flopsSqr_t, flopsMin_t, flopsTra_t);
					flops = flopsAri_t * FLOPSARI + flopsSqr_t * FLOPSSQR + flopsMin_t * FLOPSMIN + flopsTra_t * FLOPSTRA;
					end = cclock();
					functim[TIM_ALLRED] += ccelaps(start, end);
				}
#else
				flops = flopsAri * FLOPSARI + flopsSqr * FLOPSSQR + flopsMin * FLOPSMIN + flopsTra * FLOPSTRA;
#endif
				nbFLOPS++;

				if (flops > 0) {
					if (iter_time > 1.e-9) {
						double mflops = (double)flops / (double)1.e+6 / iter_time;
						MflopsSUM += mflops;
						sprintf(outnum, "%s %.2lfMC/s {%.2f Mflops %ld Ops} (%.3fs)", avgMCbys, outnum, mflops, flops, iter_time);
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
				{
					fprintf(stdout, "--> step=%4d, %12.5e, %10.5e %s\n", H.nstep, H.t, dt, outnum);
				}
			}
		}		// while
		fflush(stdout);
	}
	end_time = dcclock();
	aMCs = aMCs / nbMCbys;
	if (H.mype ==0) fprintf(stdout, "Average MC/s: %.2lf\n", aMCs);

	// Deallocate work spaces
	deallocate_work_space(H.nxyt, H, &Hw_godunov, &Hvw_godunov);
	compute_deltat_clean_mem(H, &Hw_deltat, &Hvw_deltat);

	hydro_finish(H, &Hv);
	elaps = (double)(end_time - start_time);
	timeToString(outnum, elaps);
	if (H.mype == 0) {
		fprintf(stdout, "Hydro ends in %ss (%.3lf) <%.2lf MFlops>.\n", outnum, elaps, (float)(MflopsSUM / nbFLOPS));
		fprintf(stdout, "    ");
	}
	// Average timings according to number of threads
	avgTimings(functim, TIM_END, omp_get_max_threads());
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
#ifdef MPI
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
#endif

#ifdef MPI
	MPI_Finalize();
#endif
	return 0;
}
