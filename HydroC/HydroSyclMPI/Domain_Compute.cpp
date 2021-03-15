
#include "Domain.hpp"
#include "FakeRead.hpp"
#include "ParallelInfo.hpp"
#include "ParallelInfoOpaque.hpp"
#include "Tile_Shared_Variables.hpp"
#include "cclock.hpp"

#ifdef MPI_ON
#include <mpi.h>
#endif

#include <CL/sycl.hpp>

#include <algorithm>
#include <cmath>
#include <iomanip>

#include <sys/resource.h>
#include <sys/time.h>
#include <unistd.h>

#ifndef LIGHTSYNC
#define LIGHTSYNC 0
#endif

void Domain::changeDirection() {

    // We have to change direction of the different arrays that are on the devices

    auto the_tiles = m_tilesOnDevice;
    auto queue = ParallelInfo::extraInfos()->m_queue;
    int32_t nbTiles = m_nbTiles;

    auto first_call = queue.parallel_for(nbTiles, [=](sycl::id<1> i) {
        the_tiles[i].swapStorageDims();
        the_tiles[i].swapScan();
    });

    auto second_call = queue.parallel_for(m_nbWorkItems, [=](sycl::id<1> i) {
        the_tiles[0].deviceSharedVariables()->buffers(i)->swapStorageDims();
    });

    queue
        .submit([&](sycl::handler &handler) {
            handler.depends_on({first_call, second_call});

            handler.single_task([=]() { /* nothing more */ });
        })
        .wait();

    swapScan();
}

void Domain::computeDt() {

    sycl::queue queue = ParallelInfo::extraInfos()->m_queue;
    int nb_virtual_tiles = m_nbTiles;
    if (nb_virtual_tiles % m_nbWorkItems != 0)
        nb_virtual_tiles += m_nbTiles + (m_nbWorkItems - m_nbTiles % m_nbWorkItems);

    auto the_tiles = m_tilesOnDevice;

    real_t maximum = std::numeric_limits<float>::max();

    real_t *localDt = m_localDt;
    std::fill(localDt, localDt + m_nbWorkItems, maximum);

    auto nbTiles = m_nbTiles;
    auto tcur = m_tcur;
    auto dt = m_dt;
    queue
        .submit([&](sycl::handler &handler) {
            auto global_range = sycl::nd_range<1>(nb_virtual_tiles, m_nbWorkItems);
            sycl::stream out(1024, 256, handler);
            handler.parallel_for(global_range, [=](sycl::nd_item<1> it) {
                int tile_idx = it.get_global_id(0);
                if (tile_idx < nbTiles) {
                    auto local = the_tiles[0].deviceSharedVariables();

                    int32_t workitem = it.get_local_id(0);

                    auto &my_tile = the_tiles[tile_idx];
                    my_tile.setOut(out);
                    my_tile.setBuffers(local->buffers(workitem));
                    my_tile.setTcur(tcur);
                    my_tile.setDt(dt);
                    real_t local_dt = my_tile.computeDt();

                    localDt[workitem] = sycl::min(localDt[workitem], local_dt);
                }
            });
        })
        .wait();

    dt = *std::min_element(localDt, localDt + m_nbWorkItems);
    m_dt = reduceMin(dt);
}

real_t Domain::reduceMin(real_t dt) {
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

real_t Domain::reduceMaxAndBcast(real_t dt) {
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

real_t Domain::computeTimeStep() {
    real_t dt = 0, last_dt;

    for (int32_t pass = 0; pass < 2; pass++) {

        boundary_init();
        boundary_process();

        sendUoldToDevice(); // Since Uold is modified by the two previous routines

        double start = Custom_Timer::dcclock();

        sycl::queue queue = ParallelInfo::extraInfos()->m_queue;
        int nb_virtual_tiles = m_nbTiles;
        if (nb_virtual_tiles % m_nbWorkItems != 0)
            nb_virtual_tiles += m_nbTiles + (m_nbWorkItems - m_nbTiles % m_nbWorkItems);

        auto the_tiles = m_tilesOnDevice;
        auto nb_tiles = m_nbTiles;
        auto tcur = m_tcur;
        auto dt = m_dt;

        queue
            .submit([&](sycl::handler &handler) {
                sycl::stream out(1024, 256, handler);

                auto global_range = sycl::nd_range<1>(nb_virtual_tiles, m_nbWorkItems);
                handler.parallel_for(global_range, [=](sycl::nd_item<1> it) {
                    auto local = the_tiles[0].deviceSharedVariables();
                    int idx = it.get_global_id(0);
                    if (idx < nb_tiles) {
                        int32_t workitem = it.get_local_id(0);
#if WITH_TIMERS == 1
                        double startT = Custom_Timer::dcclock(), endT;
                        int32_t thN = workitem;

#endif
                        auto &my_tile = the_tiles[idx];
                        my_tile.setOut(out);
                        my_tile.setBuffers(local->buffers(workitem));
                        my_tile.setTcur(tcur);
                        my_tile.setDt(dt);
                        my_tile.gatherconserv();
                        my_tile.godunov();
#if WITH_TIMERS == 1
                        endT = Custom_Timer::dcclock();
                        (m_timerLoops[thN])[LOOP_GODUNOV] += (endT - startT);
#endif
                    }
                });
            })
            .wait();

        // we have to wait here that all tiles are ready to update uold
        real_t minimum = std::numeric_limits<float>::max();

        real_t *localDt = m_localDt;
        std::fill(m_localDt, m_localDt + m_nbWorkItems, minimum);

        sycl::buffer<real_t> min_buf(minimum);
        queue
            .submit([&](sycl::handler &handler) {
                sycl::stream out(1024, 256, handler);

                auto global_range = sycl::nd_range<1>(nb_virtual_tiles, m_nbWorkItems);

                handler.parallel_for(global_range, [=](sycl::nd_item<1> it) {
                    int tile_idx = it.get_global_id(0);
                    auto local = the_tiles[0].deviceSharedVariables();
                    if (tile_idx < nb_tiles) {
                        int32_t workitem = it.get_local_id(0);

#if WITH_TIMERS == 1
                        int32_t thN = 0;
                        double startT = Custom_Timer::dcclock(), endT;
                        thN = myThread();
#endif
                        auto *my_tile = &the_tiles[tile_idx];
                        my_tile->setOut(out);
                        my_tile->setBuffers(local->buffers(workitem));
                        my_tile->updateconserv();

                        real_t local_dt = my_tile->computeDt();
                        localDt[workitem] = sycl::min(localDt[workitem], local_dt);

#if WITH_TIMERS == 1
                        endT = Custom_Timer::dcclock();
                        (m_timerLoops[thN])[LOOP_UPDATE] += (endT - startT);
#endif
                    }
                });
            })
            .wait();
        last_dt = *std::min_element(localDt, localDt + m_nbWorkItems);
        // we have to wait here that uold has been fully updated by all tiles
        double end = Custom_Timer::dcclock();
        m_mainTimer.add(ALLTILECMP, (end - start));

        if (m_prt) {
            std::cout << "After pass " << pass << " direction [" << m_scan << "]" << std::endl;
        }

        getUoldFromDevice();

        if (pass == 0)
            changeDirection();
        // std::cerr << " new dir\n";
    } // X_SCAN - Y_SCAN

    // final estimation of the time step
    dt = last_dt;
    // inquire the other MPI domains
    dt = reduceMin(dt);

    return dt;
}

void Domain::compute() {
    int32_t n = 0;
    real_t dt = 0;
    char vtkprt[64];
    double start, end;
    double startstep, endstep, elpasstep, elpasStepTot = 0.0;
    struct rusage myusage;
    double giga = 1024 * 1024 * 1024;
    double mega = 1024 * 1024;
    double totalCellPerSec = 0.0;
    double minCellPerSec = std::numeric_limits<float>::max();
    double maxCellPerSec = 0;
    double ecartCellPerSec = 0;
    long nbTotCelSec = 0;
    FakeRead *reader = 0;
    int myPe = ParallelInfo::mype();

    if (myPe == 0 && m_fakeRead) {
        fprintf(stderr, "HydroC: Creating fake paging file(s) ...\n");
    }
    start = Custom_Timer::dcclock();
    if (m_fakeRead)
        reader = new FakeRead(m_fakeReadSize, myPe);
#ifdef MPI_ON
    if (m_nProc > 1)
        MPI_Barrier(MPI_COMM_WORLD);
#endif
    end = Custom_Timer::dcclock();

    if (myPe == 0 && m_fakeRead) {
        char txt[64];
        double elaps = end - start;
        Custom_Timer::convertToHuman(txt, elaps);
        fprintf(stderr, "HydroC: Creating fake paging file(s) done in %s.\n", txt);
    }

    memset(vtkprt, 0, 64);
    if (myPe == 0 && m_stats > 0) {
    }

#ifdef MPI_ON
    if (myPe == 0 && m_stats > 0) {
        if (m_nProc == 1)
            std::cout << "Hydro: MPI is present with " << m_nProc << " rank" << std::endl;
        else
            std::cout << "Hydro: MPI is present with " << m_nProc << " ranks" << std::endl;
    }
#endif

    if (hasProtection()) {
        // restore the state of the computation
        double start, end;
        start = Custom_Timer::dcclock();
        readProtection();
        end = Custom_Timer::dcclock();
        if (myPe == 0) {
            char txt[256];
            double elaps = (end - start);
            Custom_Timer::convertToHuman(txt, elaps);
            std::cout << "Read protection in " << txt << " (" << elaps << "s)" << std::endl;
            std::cout.flush();
        }
    }
    start = Custom_Timer::dcclock();
    vtkprt[0] = '\0';

    if (m_tcur == 0) {
        sendUoldToDevice();

        computeDt();
        m_dt /= 2.0;
        assert(m_dt > 1.e-15);
        if (myPe == 0) {
            std::cout << " Initial dt " << std::setiosflags(std::ios::scientific)
                      << std::setprecision(5) << m_dt << std::endl;
            ;
        }
    }
    dt = m_dt;

    while (m_tcur < m_tend) {
        int needSync = 0;
        vtkprt[0] = '\0';
        if ((m_iter % 2) == 0) {
            m_dt = dt; // either the initial one or the one computed by the time step
        }
        if (reader)
            reader->Read(m_fakeRead);

        // - - - - - - - - - - - - - - - - - - -
        {
            startstep = Custom_Timer::dcclock();

            dt = computeTimeStep();
            endstep = Custom_Timer::dcclock();
            elpasstep = endstep - startstep;
            elpasStepTot += elpasstep;
        }
        // - - - - - - - - - - - - - - - - - - -
        m_tcur += m_dt;
        n++;      // iteration of this run
        m_iter++; // global iteration of the computation (accross runs)

        if (m_nStepMax > 0) {
            if (m_iter > m_nStepMax)
                break;
        }

        //
        if (m_iter == m_nDumpline) {
            dumpLine();
            sprintf(vtkprt, "%s{dumpline}", vtkprt);
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
            sprintf(vtkprt, "%s[%05d]", vtkprt, m_nvtk);
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
        if (myPe == 0) {
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
                double startflush = Custom_Timer::dcclock();
                sync();
                sync();
                sync();
                double endflush = Custom_Timer::dcclock();
                double elapsflush = endflush - startflush;
                sprintf(ftxt, "{f:%.4lf}", elapsflush);
            }
            if (reader)
                sprintf(ftxt, "%s r", ftxt);

            fprintf(stdout, "Iter %6d Time %-13.6lf Dt %-13.6g", m_iter, m_tcur, m_dt);
            fprintf(stdout, "(%.5lfs %.3lfMc/s %.3lfGB)", elpasstep, cellPerSec,
                    float(Matrix2<double>::getMax() / giga));
            fprintf(stdout, " %.1lf %s %s", resteAll, vtkprt, ftxt);
#ifdef MPI_ON
            // fprintf(stdout, "%.3lf MB/s", m_mainTimer.get(BOUNDINITBW));
#endif
            fprintf(stdout, "\n");
            fflush(stdout);
        }

        {
            int needToStopGlob = false;
            if (myPe == 0) {
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
                if (myPe == 0) {
                    std::cerr << " Hydro stops by time limit " << m_tr.timeRemain() << " < "
                              << m_timeGuard << std::endl;
                }
                std::cout.flush();
                std::cerr.flush();
                break;
            }
        }
        // std::cout << "suivant"<< myPe<< endl; std::cout.flush();
    } // while (m_tcur < m_tend)
    end = Custom_Timer::dcclock();
    m_nbRun++;
    m_elapsTotal += (end - start);

    // TODO: temporaire a equiper de mesure de temps
    if (m_checkPoint) {
        double start, end;
        start = Custom_Timer::dcclock();
        // m_tcur -= m_dt;
        if ((m_iter % 2) == 0) {
            m_dt = dt;
        }
        writeProtection();
        end = Custom_Timer::dcclock();
        if (myPe == 0) {
            char txt[256];
            double elaps = (end - start);
            Custom_Timer::convertToHuman(txt, elaps);
            std::cerr << "Write protection in " << txt << " (" << elaps << "s)" << std::endl;
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
        std::cerr << "error getting my resources usage" << std::endl;
        exit(1);
    }
    m_maxrss = myusage.ru_maxrss;
    m_ixrss = myusage.ru_ixrss;

    if (myPe == 0) {
        char timeHuman[256];
        long maxMemUsed = getMemUsed();
        double elaps = (end - start);
        printf("End of computations in %.3lf s (", elaps);
        Custom_Timer::convertToHuman(timeHuman, (end - start));
        printf("%s) with %d tiles", timeHuman, m_nbTiles);
#ifdef _OPENMP
        printf(" using %d threads", m_numThreads);
#endif
#ifdef MPI_ON
        printf(" and %d MPI tasks", m_nProc);
#endif
        // printf(" maxMEMproc %.3fGB", float (maxMemUsed / giga));
        printf(" maxMatrix %.3f MB", float(Matrix2<double>::getMax() / mega));
        if (ParallelInfo::nb_procs() > 1) {
            printf(" maxMatrixTot %.3f GB",
                   float(Matrix2<double>::getMax() * ParallelInfo::nb_procs() / giga));
        }
        printf("\n");
        Custom_Timer::convertToHuman(timeHuman, m_elapsTotal);
        printf("Total simulation time: %s in %d runs\n", timeHuman, m_nbRun);
        if (nbTotCelSec == 0) {
            nbTotCelSec = 1;   // avoid divide by 0
            minCellPerSec = 0; // so that everything is 0
        }
        double avgCellPerSec = totalCellPerSec / nbTotCelSec;
        printf("Average MC/s: %.3lf", avgCellPerSec);
        ecartCellPerSec = sqrt((ecartCellPerSec / nbTotCelSec) - (avgCellPerSec * avgCellPerSec));
        printf(" min %.3lf, max %.3lf, sig %.3lf\n", minCellPerSec, maxCellPerSec, ecartCellPerSec);
#if WITH_THREAD_TIMERS == 1
        // std::cout.precision(4);
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
        for (int32_t i = 0; i < m_nbWorkItems; i++) {
            // m_threadTimers[i].print();
            m_mainTimer += m_threadTimers[i];
        }
        // TODO
        // #pragma message "Bandwidth monitoring to do properly"
        m_mainTimer.set(BOUNDINITBW, 0);
        m_mainTimer.getStats(); // all processes involved
        // std::cout << std::endl;
        if (myPe == 0 && m_stats > 0) {
#ifdef MPI_ON
            m_mainTimer.printStats();
#else
            m_mainTimer.print();
#endif
        }
        if (myPe == 0 && m_stats > 0) {
            double elapsParallelOMP = m_mainTimer.get(ALLTILECMP);
            double seenParallel = 0;
            for (int32_t i = 0; i < TILEOMP; ++i) {
                m_mainTimer.div(Fname_t(i), m_nbWorkItems);
                seenParallel += m_mainTimer.get(Fname_t(i));
            }
            double efficiency = 100.0 * seenParallel / elapsParallelOMP;
            printf("TotalOMP//: %lf, SeenOMP//: %lf effOMP%%=%.2lf\n", elapsParallelOMP,
                   seenParallel, efficiency);
#ifdef MPI_ON
            double seenMPI = 0.0;
            for (int32_t i = 0; i < BANDWIDTH; ++i) {
                seenMPI += m_mainTimer.get(Fname_t(i));
            }
            efficiency = 100.0 * seenMPI / (end - start);
            printf("TotalMPI//: %lf, SeenMPI//: %lf effMPI%%=%.2lf\n", (end - start), seenMPI,
                   efficiency);
#endif
        }
    }

    if (reader)
        delete reader;
    // std::cerr << "End compute " << myPe << std::endl;
}

// EOF
