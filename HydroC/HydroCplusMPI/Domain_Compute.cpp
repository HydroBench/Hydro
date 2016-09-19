#ifdef MPI_ON
#include <mpi.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

#include <cstring>
#include <cstdlib>
#include <unistd.h>
#include <cmath>
#include <cstdio>
#include <climits>
#include <cerrno>
#include <iostream>
#include <iomanip>

#include <strings.h>
#include <unistd.h>
#include <malloc.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <float.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <stdarg.h>

/*
 */
using namespace std;

#ifndef LIGHTSYNC
#define LIGHTSYNC 0
#endif

#include "EnumDefs.hpp"
#include "Domain.hpp"
#include "Soa.hpp"
#include "cclock.h"

void Domain::changeDirection()
{

	// reverse the private storage direction of the tile
#pragma omp parallel for SCHEDULE
	for (int32_t i = 0; i < m_nbtiles; i++) {
		m_tiles[i]->swapScan();
		m_tiles[i]->swapStorageDims();
	}

	// reverse the shared storage direction of the threads
#pragma omp parallel for SCHEDULE
	for (int32_t i = 0; i < m_numThreads; i++) {
		m_buffers[i]->swapStorageDims();
	}
	swapScan();
}

void Domain::computeDt()
{
	real_t dt;

#pragma omp parallel for SCHEDULE
	for (int32_t i = 0; i < m_nbtiles; i++) {
		m_tiles[i]->setBuffers(m_buffers[myThread()]);
		m_tiles[i]->setTcur(m_tcur);
		m_tiles[i]->setDt(m_dt);
		m_localDt[i] = m_tiles[i]->computeDt();
	}

	dt = m_localDt[0];
	for (int32_t i = 0; i < m_nbtiles; i++) {
		dt = Min(dt, m_localDt[i]);
	}
	m_dt = reduceMin(dt);
}

real_t Domain::reduceMin(real_t dt)
{
	real_t dtmin = dt;
#ifdef MPI_ON
	if (sizeof(real_t) == sizeof(double)) {
		MPI_Allreduce(&dt, &dtmin, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
	} else {
		MPI_Allreduce(&dt, &dtmin, 1, MPI_FLOAT, MPI_MIN, MPI_COMM_WORLD);
	}
#endif
	return dtmin;
}

real_t Domain::reduceMaxAndBcast(real_t dt)
{
	real_t dtmax = dt;
#ifdef MPI_ON
	if (sizeof(real_t) == sizeof(double)) {
		MPI_Allreduce(&dt, &dtmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
		MPI_Bcast(&dtmax, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	} else {
		MPI_Allreduce(&dt, &dtmax, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
		MPI_Bcast(&dtmax, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
	}
#endif
	return dtmax;
}

int32_t Domain::tileFromMorton(int32_t t)
{
	int32_t i = t;
	int32_t m = m_mortonIdx[t];
	int32_t x, y;
	(*m_morton).idxFromMorton(x, y, m);
	i = (*m_morton) (x, y);
	return i;
}

real_t Domain::computeTimeStep()
{
	real_t dt = 0;
	int32_t t;

	for (int32_t pass = 0; pass < 2; pass++) {
		Matrix2 < real_t > &uold = *(*m_uold) (IP_VAR);
		if (m_prt)
			uold.printFormatted("uold computeTimeStep");

		boundary_init();
		boundary_process();

		real_t *pm_localDt = m_localDt;
#pragma omp parallel for private(t) SCHEDULE
		for (t = 0; t < m_nbtiles; t++) {
			// int lockStep = 0;
			int32_t i = t;
			int32_t thN = 0;
#if WITH_TIMERS == 1
			double startT = dcclock(), endT;
#ifdef _OPENMP
			thN = omp_get_thread_num();
#endif
#endif
			if (m_withMorton) {
				i = tileFromMorton(t);
			}
			m_tiles[i]->setBuffers(m_buffers[myThread()]);
			m_tiles[i]->setTcur(m_tcur);
			m_tiles[i]->setDt(m_dt);
			// cerr << i << " demarre " << endl; cerr.flush();
			// lockStep = 1;
			// m_tiles[i]->notProcessed();
			m_tiles[i]->gatherconserv();	// input uold      output u
			// m_tiles[i]->doneProcessed(lockStep);
			m_tiles[i]->godunov();
			// m_tiles[i]->doneProcessed(lockStep);
#if WITH_TIMERS == 1
			endT = dcclock();
			(m_timerLoops[thN])[LOOP_GODUNOV] += (endT - startT);
#endif
		}
		// we have to wait here that all tiles are ready to update uold
#pragma omp parallel for private(t) SCHEDULE
		for (t = 0; t < m_nbtiles; t++) {
			int32_t i = t;
#if WITH_TIMERS == 1
			int32_t thN = 0;
			double startT = dcclock(), endT;
#ifdef _OPENMP
			thN = omp_get_thread_num();
#endif
#endif
			if (m_withMorton) {
				i = tileFromMorton(t);
			}
			m_tiles[i]->setBuffers(m_buffers[myThread()]);
			m_tiles[i]->updateconserv();	// input u, flux       output uold
			if (pass == 1) {
				m_localDt[i] = m_tiles[i]->computeDt();
			}
#if WITH_TIMERS == 1
			endT = dcclock();
			(m_timerLoops[thN])[LOOP_UPDATE] += (endT - startT);
#endif
		}
// we have to wait here that uold has been fully updated by all tiles
		if (m_prt) {
			cout << "After pass " << pass << " direction [" << m_scan << "]" << endl;
		}
		changeDirection();
	}			// X_SCAN - Y_SCAN
	changeDirection();	// to do X / Y then Y / X then X / Y ...

	// final estimation of the time step
	dt = m_localDt[0];
	for (int32_t i = 0; i < m_nbtiles; i++) {
		dt = Min(dt, m_localDt[i]);
	}

	// inquire the other MPI domains
	dt = reduceMin(dt);

	return dt;
}

void Domain::compute()
{
	int32_t n = 0;
	real_t dt = 0;
	char vtkprt[64];
	double start, end;
	double startstep, endstep, elpasstep;
	struct rusage myusage;
	double giga = 1024 * 1024 * 1024;
	double totalCellPerSec = 0.0;
	long nbTotCelSec = 0;
	memset(vtkprt, 0, 64);

#ifdef _OPENMP
	if (m_myPe == 0) {
		cout << "Hydro: OpenMP max threads " << omp_get_max_threads() << endl;
		// cout << "Hydro: OpenMP num threads " << omp_get_num_threads() << endl;
		cout << "Hydro: OpenMP num procs   " << omp_get_num_procs() << endl;
		cout << "Hydro: OpenMP " << Schedule << endl;
	}
#endif
#ifdef MPI_ON
	if (m_myPe == 0) {
		cout << "Hydro: MPI is present with " << m_nProc << " tasks" << endl;;
	}
#endif

	start = dcclock();
	vtkprt[0] = '\0';

	if (m_tcur == 0) {
//              if (m_dtImage > 0) {
//                      char pngName[256];
//                      pngProcess();
// #if WITHPNG > 0
//                      sprintf(pngName, "%s_%06d.png", "Image", m_npng);
// #else
//                      sprintf(pngName, "%s_%06d.ppm", "Image", m_npng);
// #endif
//                      pngWriteFile(pngName);  
//                      pngCloseFile();
//                      sprintf(vtkprt, "%s   (%05d)", vtkprt, m_npng);
//                      m_npng++;
//              }
		// only for the start of the test case

		computeDt();
		m_dt /= 2.0;
		assert(m_dt > 1.e-15);
		if (m_myPe == 0) {
			cout << " Initial dt " << setiosflags(ios::scientific) << setprecision(5) << m_dt << endl;;
		}
	}
	dt = m_dt;

	while (m_tcur < m_tend) {
		vtkprt[0] = '\0';
		if ((m_iter % 2) == 0) {
			m_dt = dt;	// either the initial one or the one computed by the time step
		}

		startstep = dcclock();
		dt = computeTimeStep();
		endstep = dcclock();
		elpasstep = endstep - startstep;
		// m_dt = 1.e-3;
		m_tcur += m_dt;
		n++;		// iteration of this run
		m_iter++;	// global iteration of the computation (accross runs)

		if (m_nStepMax > 0) {
			if (m_iter > m_nStepMax)
				break;
		}

		int outputVtk = 0;
		if (m_nOutput > 0) {
			if ((m_iter % m_nOutput) == 0) {
				outputVtk++;
			}
		}
		if (m_dtOutput > 0) {
			if (m_tcur > m_nextOutput) {
				m_nextOutput += m_dtOutput;
				outputVtk++;
			}
		}
		if (outputVtk) {
			vtkOutput(m_nvtk);
			sprintf(vtkprt, "[%05d]", m_nvtk);
			m_nvtk++;
		}

		int outputImage = 0;
		if (m_dtImage > 0) {
			if (m_tcur > m_nextImage) {
				m_nextImage += m_dtImage;
				outputImage++;
			}
		}
		if (m_nImage > 0) {
			if ((m_iter % m_nImage) == 0) {
				outputImage++;
			}
		}

		if (outputImage) {
			char pngName[256];
			m_npng++;
			pngProcess();
#if WITHPNG > 0
			sprintf(pngName, "%s_%06d.png", "Image", m_npng);
#else
			sprintf(pngName, "%s_%06d.ppm", "Image", m_npng);
#endif
			pngWriteFile(pngName);
			pngCloseFile();
			sprintf(vtkprt, "%s (%05d)", vtkprt, m_npng);
		}
		double resteAll = m_tr.timeRemain() - m_timeGuard;
		if (m_myPe == 0) {
			int64_t totCell = int64_t(m_globNx) * int64_t(m_globNy);
			double cellPerSec = totCell / elpasstep / 1000000;
			if (n > 4) {
				// skip the 4 first iterations to let the system stabilize
				totalCellPerSec += cellPerSec;
				nbTotCelSec++;
			}
			fprintf(stdout, "Iter %6d Time %-13.6g Dt %-13.6g (%f %f Mc/s %f GB) %lf %s \n",
				m_iter, m_tcur, m_dt, elpasstep, cellPerSec, float (getMemUsed() / giga), resteAll, vtkprt);
			fflush(stdout);
		}
		{
			int needToStopGlob = false;
			if (m_myPe == 0) {
				double reste = m_tr.timeRemain();
				int needToStop = false;
				if (reste < m_timeGuard) {
					needToStop = true;
				}
				needToStopGlob = needToStop;
			}
#ifdef MPI_ON
			MPI_Bcast(&needToStopGlob, 1, MPI_INT, 0, MPI_COMM_WORLD);
#endif
			if (needToStopGlob) {
				if (m_myPe == 0) {
					cerr << " Hydro stops by time limit " << m_tr.timeRemain() << " < " << m_timeGuard << endl;
				}
				cout.flush();
				cerr.flush();
				break;
			}
		}
		// cout << "suivant"<< m_myPe<< endl; cout.flush();
	}			// while (m_tcur < m_tend)
	end = dcclock();
	m_nbRun++;
	m_elapsTotal += (end - start);
	// TODO: temporaire a equiper de mesure de temps
	if (m_checkPoint) {
		double start, end;
		start = dcclock();
		m_dt = dt;
		writeProtection();
		end = dcclock();
		if (m_myPe == 0) {
			char txt[256];
			double elaps = (end - start);
			convertToHuman(txt, elaps);
			cerr << "Write protection in " << txt << " (" << elaps << "s)" << endl;
		}
	}

	if (m_nStepMax > 0) { 
		if (m_iter > m_nStepMax) { 
			if (m_forceStop) {
				StopComputation();
			}
		}
	}
	if (m_tcur >= m_tend) {
		StopComputation();
	}

	if (getrusage(RUSAGE_SELF, &myusage) != 0) {
		cerr << "error getting my resources usage" << endl;
		exit(1);
	}
	m_maxrss = myusage.ru_maxrss;
	m_ixrss = myusage.ru_ixrss;

	if (m_myPe == 0) {
		char timeHuman[256];
		long maxMemUsed = getMemUsed();
		cout << "End of computations in " << setiosflags(ios::fixed) << setprecision(3) << (end - start);
		cout << " s ";
		cout << " (";
		convertToHuman(timeHuman, (end - start));
		cout << timeHuman;
		cout << ")";
		cout << " with " << m_nbtiles << " tiles";
#ifdef _OPENMP
		cout << " using " << m_numThreads << " threads";
#endif
#ifdef MPI_ON
		cout << " and " << m_nProc << " MPI tasks";
#endif
		// cout << " maxRSS " << m_maxrss;
		cout << std::resetiosflags(std::ios::showbase) << setprecision(3) << setiosflags(ios::fixed);
		cout << " maxMEMproc " << float (maxMemUsed / giga) << "GB";
		if (getNbpe() > 1) {
			cout << " maxMEMtot " << float (maxMemUsed * getNbpe() / giga) << "GB";
		}
		cout << endl;
		convertToHuman(timeHuman, m_elapsTotal);
		cout << "Total simulation time: " << timeHuman << " in " << m_nbRun << " runs" << endl;
		cout << "Average MC/s: " << totalCellPerSec / nbTotCelSec << endl;

#if WITH_TIMERS == 1
		cout.precision(4);
		for (int32_t i = 0; i < m_numThreads; i++) {
			cout << "THread " << i << " ";
			for (int32_t j = 0; j < LOOP_END; j++) {
				cout << (m_timerLoops[i])[j] << " ";
			}
			cout << endl;
		}
#endif

	}
	// cerr << "End compute " << m_myPe << endl;
}

// EOF
