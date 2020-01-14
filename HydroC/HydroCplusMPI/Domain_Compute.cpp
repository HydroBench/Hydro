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
#include "FakeRead.hpp"
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

    if ((m_tasked == 0) && (m_taskeddep == 0)) {
#pragma omp parallel for SCHEDULE
	for (int32_t t = 0; t < m_nbtiles; t++) {
	    int32_t i = t;
	    if (m_withMorton) {
		i = m_mortonIdx[t];
		assert(i >= 0);
		assert(i < m_nbtiles);
	    }
	    m_tiles[i]->setBuffers(m_buffers[myThread()]);
	    m_tiles[i]->setTcur(m_tcur);
	    m_tiles[i]->setDt(m_dt);
	    m_localDt[i] = m_tiles[i]->computeDt();
	}
    } else {
#pragma omp parallel
	{
#pragma omp single nowait
	    {
		for (int32_t t = 0; t < m_nbtiles; t++) {
#pragma omp task
		    {
			int32_t i = t;
			if (m_withMorton) {
			    i = m_mortonIdx[t];
			    assert(i >= 0);
			    assert(i < m_nbtiles);
			}
			m_tiles[i]->setBuffers(m_buffers[myThread()]);
			m_tiles[i]->setTcur(m_tcur);
			m_tiles[i]->setDt(m_dt);
			m_localDt[i] = m_tiles[i]->computeDt();
		    }
		}
	    }
	}
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
    double t1 = MPI_Wtime();
    if (sizeof(real_t) == sizeof(double)) {
	MPI_Allreduce(&dt, &dtmin, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    } else {
	MPI_Allreduce(&dt, &dtmin, 1, MPI_FLOAT, MPI_MIN, MPI_COMM_WORLD);
    }
    double t2 = MPI_Wtime();
    m_threadTimers[myThread()].add(REDUCEMIN, (t2 - t1));
#endif
    return dtmin;
}

real_t Domain::reduceMaxAndBcast(real_t dt)
{
    real_t dtmax = dt;
#ifdef MPI_ON
    double t1 = MPI_Wtime();
    if (sizeof(real_t) == sizeof(double)) {
	MPI_Allreduce(&dt, &dtmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
	// MPI_Bcast(&dtmax, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    } else {
	MPI_Allreduce(&dt, &dtmax, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
	// MPI_Bcast(&dtmax, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    }
    double t2 = MPI_Wtime();
    m_threadTimers[myThread()].add(REDUCEMAX, (t2 - t1));
#endif
    return dtmax;
}

int32_t Domain::tileFromMorton(int32_t t)
{
    int32_t it = t;
    int32_t m = m_mortonIdx[t];
    int32_t x, y;
    // int rc = (*m_morton).idxFromMorton(x, y, m);
    // assert(rc == true);
    // i = (*m_morton) (x, y);
    int32_t mortonH = (m_ny + m_tileSize - 1) / m_tileSize;
    int32_t mortonW = (m_nx + m_tileSize - 1) / m_tileSize;
    it = 0;
    for (int32_t j = 0; j < mortonH; j++) {
	for (int32_t i = 0; i < mortonW; i++) {
	    if (m == morton2(i, j))
		return it;
	    it++;
	}
    }

    return it;
}

void Domain::compTStask1(int32_t tile)
{
    // int lockStep = 0;
    int32_t i = tile;
    int32_t thN = 0;
#if WITH_TIMERS == 1
    double startT = dcclock(), endT;
    thN = myThread();
#endif
    if (m_withMorton) {
	i = m_mortonIdx[tile];
	assert(i >= 0);
	assert(i < m_nbtiles);
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

void Domain::compTStask2(int32_t tile, int32_t mydep, int32_t mine)
{
    int32_t i = tile;
#if WITH_TIMERS == 1
    int32_t thN = 0;
    double startT = dcclock(), endT;
    thN = myThread();
#endif
    if (m_withMorton) {
	i = m_mortonIdx[tile];
	assert(i >= 0);
	assert(i < m_nbtiles);
    }
    m_tiles[i]->setBuffers(m_buffers[myThread()]);
    m_tiles[i]->updateconserv();	// input u, flux       output uold
    // if (pass == 1) {
    m_localDt[i] = m_tiles[i]->computeDt();
    //}
#if WITH_TIMERS == 1
    endT = dcclock();
    (m_timerLoops[thN])[LOOP_UPDATE] += (endT - startT);
#endif
    // char txt[256]; sprintf(txt, "%03d prev %03d done\n", mine, mydep); cerr << txt;
}

real_t Domain::computeTimeStep()
{
    real_t dt = 0;

    for (int32_t pass = 0; pass < 2; pass++) {
	Matrix2 < real_t > &uold = *(*m_uold) (IP_VAR);

	if (m_prt)
	    uold.printFormatted("uold computeTimeStep");

	boundary_init();
	boundary_process();

	real_t *pm_localDt = m_localDt;
	double start = dcclock();
	int32_t t;
	if (m_tasked == 0) {
#pragma omp parallel for private(t) SCHEDULE
	    for (t = 0; t < m_nbtiles; t++) {
		compTStask1(t);
	    }
	    // we have to wait here that all tiles are ready to update uold
#pragma omp parallel for private(t) SCHEDULE
	    for (t = 0; t < m_nbtiles; t++) {
		compTStask2(t, 0, 0);
	    }

	} else if (m_taskeddep > 0) {
	    int32_t *tileProcessed =
		(int32_t *) alloca(m_nbtiles * sizeof(int32_t));
	    assert(tileProcessed != 0);
	    for (int tile = 0; tile < m_nbtiles; tile++) {
		// reset the tasks flags
		tileProcessed[tile] = 0;
	    }
// #pragma message "Version avec TASK et dependance"
	    int32_t mydep = 0, mine = 0;
#pragma omp parallel private(t) firstprivate(mine, mydep) shared(tileProcessed)
	    {
#pragma omp single nowait
		{
		    for (t = 0; t < m_nbtiles; t++) {
// #pragma omp task depend(out: tileProcessed[t])
#pragma omp task 
			{
			    compTStask1(t);
			    tileProcessed[t]++;
			    // char txt[256]; sprintf(txt, "%03d task done\n", tileOrder[t]); cerr << txt;
			}

			mine = t;
			mydep = (t == 0) ? t : t - 1;
		    }
		}
#pragma omp barrier		// we have to wait here that all tiles are ready to update uold
#pragma omp single nowait
		{
		    for (t = 0; t < m_nbtiles; t++) {
// #pragma omp task depend(out: tileProcessed[mydep], tileProcessed[mine])
#pragma omp task
			{
			    compTStask2(t, mydep, mine);
			}
		    }
		}		// omp single
	    }			// omp parallel
	} else {
// #pragma message "Version avec TASK"
#pragma omp parallel private(t)
	    {
#pragma omp single nowait
		{
		    for (t = 0; t < m_nbtiles; t++) {
#pragma omp task
			{
			    compTStask1(t);
			}
		    }
		}
#pragma omp barrier		// we have to wait here that all tiles are ready to update uold
#pragma omp single nowait
		{
		    for (t = 0; t < m_nbtiles; t++) {
#pragma omp task
			{
			    compTStask2(t, 0, 0);
			}
		    }
		}
	    }
	}
	// we have to wait here that uold has been fully updated by all tiles
	double end = dcclock();
	m_mainTimer.add(ALLTILECMP, (end - start));

	if (m_prt) {
	    cout << "After pass " << pass << " direction [" << m_scan << "]" <<
		endl;
	}
	changeDirection();
	// cerr << " new dir\n";
    }				// X_SCAN - Y_SCAN
    changeDirection();		// to do X / Y then Y / X then X / Y ...

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
    double startstep, endstep, elpasstep, elpasStepTot = 0.0;
    struct rusage myusage;
    double giga = 1024 * 1024 * 1024;
    double mega = 1024 * 1024;
    double totalCellPerSec = 0.0;
    double minCellPerSec = FLT_MAX;
    double maxCellPerSec = 0;
    double ecartCellPerSec = 0;
    long nbTotCelSec = 0;
    FakeRead *reader = 0;

    if (m_myPe == 0 && m_fakeRead) {
	fprintf(stderr, "HydroC: Creating fake paging file(s) ...\n");
    }
    start = dcclock();
    if (m_fakeRead)
	reader = new FakeRead(m_fakeReadSize, m_myPe);
#ifdef MPI_ON
    if (m_nProc > 1)
	MPI_Barrier(MPI_COMM_WORLD);
#endif
    end = dcclock();

    if (m_myPe == 0 && m_fakeRead) {
	char txt[64];
	double elaps = end - start;
	convertToHuman(txt, elaps);
	fprintf(stderr, "HydroC: Creating fake paging file(s) done in %s.\n",
		txt);
    }

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
	if (m_nProc == 1)
	    cout << "Hydro: MPI is present with " << m_nProc << " rank" << endl;
	else
	    cout << "Hydro: MPI is present with " << m_nProc << " ranks" <<
		endl;
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
	    cout << " Initial dt " << setiosflags(ios::
						  scientific) << setprecision(5)
		<< m_dt << endl;
	}
    }
    dt = m_dt;

    while (m_tcur < m_tend) {
	int needSync = 0;
	vtkprt[0] = '\0';
	if ((m_iter % 2) == 0) {
	    m_dt = dt;		// either the initial one or the one computed by the time step
	}
	if (reader)
	    reader->Read(m_fakeRead);

	//
	// - - - - - - - - - - - - - - - - - - -
	//
	{
	    startstep = dcclock();
// #ifdef MPI_ON
//                      double t1 =  MPI_Wtime();
// #endif

	    dt = computeTimeStep();
	    endstep = dcclock();
	    elpasstep = endstep - startstep;
// #ifdef MPI_ON
//                      double t2 =  MPI_Wtime();
//                      cerr << (t2 - t1) << " " << elpasstep << endl;
// #endif
	    elpasStepTot += elpasstep;
	}
	//
	// - - - - - - - - - - - - - - - - - - -
	//
	// m_dt = 1.e-3;
	m_tcur += m_dt;
	n++;			// iteration of this run
	m_iter++;		// global iteration of the computation (accross runs)

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
	    needSync++;
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
	// TODO
// #pragma message "Bandwidth monitoring to do properly"
	m_mainTimer.set(BOUNDINITBW, 0);
	if (m_myPe == 0) {
	    int64_t totCell = int64_t(m_globNx) * int64_t(m_globNy);
	    double cellPerSec = totCell / elpasstep / 1000000;
	    char ftxt[32];
	    ftxt[0] = '\0';
	    if (n > 4) {
		// skip the 4 first iterations to let the system stabilize
		totalCellPerSec += cellPerSec;
		nbTotCelSec++;
		if (cellPerSec > maxCellPerSec)
		    maxCellPerSec = cellPerSec;
		if (cellPerSec < minCellPerSec)
		    minCellPerSec = cellPerSec;
		ecartCellPerSec += (cellPerSec * cellPerSec);
	    }
	    if (m_forceSync && needSync) {
		double startflush = dcclock();
		sync();
		sync();
		sync();
		double endflush = dcclock();
		double elapsflush = endflush - startflush;
		sprintf(ftxt, "{f:%.4lf}", elapsflush);
	    }
	    if (reader)
		sprintf(ftxt, "%s r", ftxt);

	    fprintf(stdout, "Iter %6d Time %-13.6lf Dt %-13.6g", m_iter, m_tcur,
		    m_dt);
	    fprintf(stdout, "(%.5lfs %.3lfMc/s %.3lfGB)", elpasstep, cellPerSec,
		    float (Matrix2 < double >::getMax() / giga));
	    fprintf(stdout, " %.1lf %s %s", resteAll, vtkprt, ftxt);
#ifdef MPI_ON
	    // fprintf(stdout, "%.3lf MB/s", m_mainTimer.get(BOUNDINITBW));
#endif
	    fprintf(stdout, "\n");
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
		    cerr << " Hydro stops by time limit " << m_tr.timeRemain()
			<< " < " << m_timeGuard << endl;
		}
		cout.flush();
		cerr.flush();
		break;
	    }
	}
	// cout << "suivant"<< m_myPe<< endl; cout.flush();
    }				// while (m_tcur < m_tend)
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
	    cerr << "Write protection in " << txt << " (" << elaps << "s)" <<
		endl;
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
	double elaps = (end - start);
	printf("End of computations in %.3lf s (", elaps);
	convertToHuman(timeHuman, (end - start));
	printf("%s) with %d tiles", timeHuman, m_nbtiles);
#ifdef _OPENMP
	printf(" using %d threads", m_numThreads);
#endif
#ifdef MPI_ON
	printf(" and %d MPI tasks", m_nProc);
#endif
	// printf(" maxMEMproc %.3fGB", float (maxMemUsed / giga));
	printf(" maxMatrix %.3f MB",
	       float (Matrix2 < double >::getMax() / mega));
	if (getNbpe() > 1) {
	    printf(" maxMatrixTot %.3f GB",
		   float (Matrix2 < double >::getMax() * getNbpe() / giga));
	}
	printf("\n");
	convertToHuman(timeHuman, m_elapsTotal);
	printf("Total simulation time: %s in %d runs\n", timeHuman, m_nbRun);
	if (nbTotCelSec == 0) {
	    nbTotCelSec = 1;	// avoid divide by 0
	    minCellPerSec = 0;	// so that everything is 0
	}
	double avgCellPerSec = totalCellPerSec / nbTotCelSec;
	printf("Average MC/s: %.3lf", avgCellPerSec);
	ecartCellPerSec =
	    sqrt((ecartCellPerSec / nbTotCelSec) -
		 (avgCellPerSec * avgCellPerSec));
	printf(" min %.3lf, max %.3lf, sig %.3lf\n", minCellPerSec,
	       maxCellPerSec, ecartCellPerSec);
#if WITH_THREAD_TIMERS == 1
	// cout.precision(4);
	for (int32_t i = 0; i < m_numThreads; i++) {
	    printf("Thread %4d: ", i);
	    for (int32_t j = 0; j < LOOP_END; j++) {
		printf("loop %d: %lfs ", j, (m_timerLoops[i])[j]);
	    }
	    printf("\n");
	}
#endif
    }
    {
	// get the threads timers values and add them to our timer
	for (int32_t i = 0; i < m_numThreads; i++) {

	    // m_threadTimers[i].print();
	    m_mainTimer += m_threadTimers[i];
	}
	// TODO
// #pragma message "Bandwidth monitoring to do properly"
	m_mainTimer.set(BOUNDINITBW, 0);
	m_mainTimer.getStats();	// all processes involved
	// cout << endl;
	if (m_myPe == 0) {
#ifdef MPI_ON
	    m_mainTimer.printStats();
#else
	    m_mainTimer.print();
#endif
	}
	if (m_myPe == 0) {
	    double elapsParallelOMP = m_mainTimer.get(ALLTILECMP);
	    double seenParallel = 0;
	    for (int32_t i = 0; i < TILEOMP; ++i) {
		m_mainTimer.div(Fname_t(i), m_numThreads);
		seenParallel += m_mainTimer.get(Fname_t(i));
	    }
	    double efficiency = 100.0 * seenParallel / elapsParallelOMP;
	    printf("TotalOMP//: %lf, SeenOMP//: %lf effOMP%%=%.2lf\n",
		   elapsParallelOMP, seenParallel, efficiency);
#ifdef MPI_ON
	    double seenMPI = 0.0;
	    for (int32_t i = 0; i < BANDWIDTH; ++i) {
		seenMPI += m_mainTimer.get(Fname_t(i));
	    }
	    efficiency = 100.0 * seenMPI / (end - start);
	    printf("TotalMPI//: %lf, SeenMPI//: %lf effMPI%%=%.2lf\n",
		   (end - start), seenMPI, efficiency);
#endif

	}
    }

    if (reader)
	delete reader;
    // cerr << "End compute " << m_myPe << endl;
}

// EOF
