/*
 * Domain_compute_StepbyStep.cpp
 *
 *  Created on: 27 avr. 2021
 *      Author: weilljc
 */

#include "Domain.hpp"
#include "ParallelInfo.hpp"
#include "ParallelInfoOpaque.hpp"
#include "Utilities.hpp"
#include "cclock.hpp"

real_t Domain::computeTimeStepByStep() {
    real_t dt = 0, last_dt = zero;

    for (int32_t pass = 0; pass < 2; pass++) {
        double start, startT, endT;

        sycl::queue queue = ParallelInfo::extraInfos()->m_queue;

        auto the_tiles = m_tilesOnDevice;
        auto tileSize = m_tileSize + 2 * m_ExtraLayer;

#ifdef MPI_ON
        // This is modifying uold
        boundary_init();
        boundary_process();

        start = Custom_Timer::dcclock();
        startT = start;

        sendUoldToDevice(); // Since Uold is modified by the two previous routines
        endT = Custom_Timer::dcclock();
        m_mainTimer.add(SENDUOLD, endT - startT);

        startT = endT;
#else

        start = startT = Custom_Timer::dcclock();
        int32_t b_d = m_boundary_down, b_u = m_boundary_up;
        int32_t b_l = m_boundary_left, b_r = m_boundary_right;

        queue
            .submit([&](sycl::handler &handler) {
                handler.parallel_for(sycl::range<1>(m_nbTiles), [=](sycl::item<1> idx) {
                    auto &my_tile = the_tiles[idx[0]];
                    my_tile.boundary_process(b_l, b_r, b_u, b_d);
                });
            })
            .wait();

        endT = Custom_Timer::dcclock();
        m_mainTimer.add(BOUNDEXEC, endT - startT);
        startT = endT;

#endif

        // Update Dt
        auto dt = m_dt, dx = m_dx;
        queue
            .submit([&](sycl::handler &handler) {
                handler.single_task([=]() {
                    the_tiles[0].deviceSharedVariables()->m_dt = dt;
                    the_tiles[0].deviceSharedVariables()->m_dx = dx;
                });
            })
            .wait();

        // do Gather => Validated
        queue.submit([&](sycl::handler &handler) {
            handler.parallel_for(sycl::nd_range<3>(sycl::range<3>(tileSize, tileSize, m_nbTiles),
                                                   sycl::range<3>(16, 16, 1)),
                                 [=](auto ids) {
                                     the_tiles[ids.get_global_id(2)].gatherconserv(
                                         ids.get_global_id(0), ids.get_global_id(1));
                                 });
        });

        double start_wait = Custom_Timer::dcclock();
        queue.wait();

        endT = Custom_Timer::dcclock();
        m_mainTimer.add(WAITQUEUE, endT - start_wait);

        m_mainTimer.add(GATHCVAR, endT - startT);

        if (m_prt) {
            std::cout << "= = = = = = = =  = =" << std::endl;
            std::cout << "      Godunov" << std::endl;
            std::cout << "= = = = = = = =  = =" << std::endl;
            std::cout << std::endl << " scan " << (int32_t)m_scan << std::endl;
            std::cout << std::endl << " time " << m_tcur << std::endl;
            std::cout << std::endl << " dt " << m_dt << std::endl;
        }

        startT = endT;

        queue.submit([&](sycl::handler &handler) {
            handler.parallel_for(sycl::range<3>(m_nbTiles, tileSize, tileSize),
                                 [=](auto ids) { the_tiles[ids[0]].constprim(ids[1], ids[2]); });
        });
        start_wait = Custom_Timer::dcclock();
        queue.wait();

        endT = Custom_Timer::dcclock();
        m_mainTimer.add(WAITQUEUE, endT - start_wait);
        m_mainTimer.add(CONSTPRIM, endT - startT);

        startT = endT;

        auto smallp = Square(onHost.m_smallc) / onHost.m_gamma; // TODO: We
        queue.submit([&](sycl::handler &handler) {
            handler.parallel_for(sycl::range<3>(tileSize, tileSize, m_nbTiles), [=](auto ids) {
                the_tiles[ids[2]].eos(TILE_FULL, ids[0], ids[1], smallp);
            });
        });

        start_wait = Custom_Timer::dcclock();
        queue.wait();

        endT = Custom_Timer::dcclock();
        m_mainTimer.add(WAITQUEUE, endT - start_wait);
        m_mainTimer.add(EOS, endT - startT);

        if (m_iorder > 1) {
            startT = endT;
            real_t ov_slope_type = one / onHost.m_slope_type;

            queue.submit([&](sycl::handler &handler) {
                handler.parallel_for(sycl::range<3>(m_nbTiles, tileSize, tileSize), [=](auto ids) {
                    the_tiles[ids[0]].slope(ids[1], ids[2], ov_slope_type);
                });
            });

            start_wait = Custom_Timer::dcclock();
            queue.wait();
            endT = Custom_Timer::dcclock();
            m_mainTimer.add(WAITQUEUE, endT - start_wait);
            m_mainTimer.add(SLOPE, endT - startT);
            // Slope must be finished before trace
        }

        startT = endT;
        real_t zerol = zero, zeror = zero, project = zero;
        real_t dtdx = m_dt / m_dx;
        if (onHost.m_scheme == SCHEME_MUSCL) { // MUSCL-Hancock method
            zerol = -hundred / dtdx;
            zeror = hundred / dtdx;
            project = one;
        }

        if (onHost.m_scheme == SCHEME_PLMDE) { // standard PLMDE
            zerol = zero;
            zeror = zero;
            project = one;
        }

        if (onHost.m_scheme == SCHEME_COLLELA) { // Collela's method
            zerol = zero;
            zeror = zero;
            project = zero;
        }

        queue.submit([&](sycl::handler &handler) {
            handler.parallel_for(sycl::range<3>(m_nbTiles, tileSize, tileSize), [=](auto ids) {
                the_tiles[ids[0]].trace(ids[1], ids[2], zerol, zeror, project, dtdx);
            });
        });

        start_wait = Custom_Timer::dcclock();
        queue.wait(); // Trace must be finished before qleftr

        endT = Custom_Timer::dcclock();
        m_mainTimer.add(WAITQUEUE, endT - start_wait);

        m_mainTimer.add(TRACE, endT - startT);

        startT = endT;
        auto qleftr = queue.submit([&](sycl::handler &handler) {
            handler.parallel_for(sycl::range<3>(m_nbTiles, tileSize, tileSize),
                                 [=](auto ids) { the_tiles[ids[0]].qleftright(ids[1], ids[2]); });
        });

        queue.wait();
        start_wait = Custom_Timer::dcclock();
        queue.wait();

        endT = Custom_Timer::dcclock();

        m_mainTimer.add(WAITQUEUE, endT - start_wait);

        double qleftTime = endT - startT;

        m_mainTimer.add(QLEFTR, qleftTime);

        startT = endT;
        real_t gamma6 = (onHost.m_gamma + one) / (two * onHost.m_gamma);
        real_t smallpp = onHost.m_smallr * smallp;

        queue.submit([&](sycl::handler &handler) {
            handler.parallel_for(sycl::nd_range<3>(sycl::range<3>(tileSize, tileSize, m_nbTiles),
                                                   sycl::range<3>(16, 16, 1)),
                                 [=](auto ids) // [[intel::reqd_sub_group_size(8)]]
                                 {
                                     the_tiles[ids.get_global_id(2)].riemann(
                                         ids.get_global_id(0), ids.get_global_id(1), smallp, gamma6,
                                         smallpp);
                                 });
        });
        start_wait = Custom_Timer::dcclock();
        queue.wait();
        endT = Custom_Timer::dcclock();
        m_mainTimer.add(WAITQUEUE, endT - start_wait);

        m_mainTimer.add(RIEMANN, endT - startT);

        startT = endT;
        queue.submit([&](sycl::handler &handler) {
            handler.parallel_for(sycl::nd_range<3>(sycl::range<3>(tileSize, tileSize, m_nbTiles),
                                                   sycl::range<3>(16, 16, 1)),
                                 [=](auto ids) // [[intel::reqd_sub_group_size(8)]]
                                 {
                                     the_tiles[ids.get_global_id(2)].compflx(ids.get_global_id(0),
                                                                             ids.get_global_id(1));
                                 });
        });
        start_wait = Custom_Timer::dcclock();
        queue.wait();

        endT = Custom_Timer::dcclock();
        m_mainTimer.add(WAITQUEUE, endT - start_wait);

        m_mainTimer.add(COMPFLX, endT - startT);

        // we have to wait here that all tiles are ready to update uold
        startT = endT;

        queue.submit([&](sycl::handler &handler) {
            handler.parallel_for(sycl::nd_range<3>(sycl::range<3>(tileSize, tileSize, m_nbTiles),
                    sycl::range<3>(16, 16, 1)), [=](auto ids)
            		{
                	the_tiles[ids.get_global_id(2)].updateconserv( ids.get_global_id(0), ids.get_global_id(1)
                			, dtdx);
            });
        });

        start_wait = Custom_Timer::dcclock();
        queue.wait();
        endT = Custom_Timer::dcclock();
        m_mainTimer.add(WAITQUEUE, endT - start_wait);

        m_mainTimer.add(UPDCVAR, endT - startT);

        startT = endT;

        real_t *result = sycl::malloc_shared<real_t>(1, queue);
        *result = zero;

        if (m_scan == Y_SCAN) {
            queue
                .submit([&](sycl::handler &handler) {
                    handler.parallel_for(sycl::range<1>(m_nbTiles), [=](auto ids) {
                        the_tiles[ids].swapScan();
                        the_tiles[ids].swapStorageDims();
                    });
                })
                .wait();
        }

        queue
            .submit([&](sycl::handler &handler) {
                handler.parallel_for(sycl::range<3>(m_nbTiles, tileSize, tileSize), [=](auto ids) {
                    the_tiles[ids[0]].computeDt1(ids[1], ids[2]);
                });
            })
            .wait();

        queue
            .submit([&](sycl::handler &handler) {
                handler.parallel_for(sycl::range<3>(tileSize, tileSize, m_nbTiles), [=](auto ids) {
                    the_tiles[ids[2]].eos(TILE_INTERIOR, ids[0], ids[1], smallp);
                });
            })
            .wait();

        queue.submit([&](sycl::handler &handler) {
            handler.parallel_for(sycl::nd_range<3>(sycl::range<3>(tileSize, tileSize, m_nbTiles),
                                                   sycl::range<3>(16, 16, 1)),
                                 sycl::ONEAPI::reduction(result, sycl::ONEAPI::maximum<real_t>()),
                                 [=](auto ids, auto &res) {
                                     real_t courn = the_tiles[ids.get_global_id(2)].computeDt2(
                                         ids.get_global_id(0), ids.get_global_id(1));
                                     res.combine(courn);
                                 });
        });
        start_wait = Custom_Timer::dcclock();
        queue.wait();

        endT = Custom_Timer::dcclock();
        m_mainTimer.add(WAITQUEUE, endT - start_wait);

        endT = Custom_Timer::dcclock();
        m_mainTimer.add(COMPDT, endT - startT);

        if (m_scan == Y_SCAN) {
            queue
                .submit([&](sycl::handler &handler) {
                    handler.parallel_for(sycl::range<1>(m_nbTiles), [=](auto ids) {
                        the_tiles[ids].swapScan();
                        the_tiles[ids].swapStorageDims();
                    });
                })
                .wait();
        }

        real_t courno = *result;
        last_dt = onHost.m_cfl * m_dx / sycl::max(courno, onHost.m_smallc);

        sycl::free(result, queue);

        // we have to wait here that uold has been fully updated by all tiles
        double end = Custom_Timer::dcclock();
        m_mainTimer.add(ALLTILECMP, (end - start));

        if (m_prt) {
            std::cout << "After pass " << pass << " direction [" << m_scan << "]" << std::endl;
        }

#ifdef MPI_ON
        startT = Custom_Timer::dcclock();
        getUoldFromDevice();
        m_mainTimer.add(GETUOLD, Custom_Timer::dcclock() - startT);
#endif

        if (pass == 0)
            changeDirection();

    } // X_SCAN - Y_SCAN

    // final estimation of the time step
    dt = last_dt;
    // inquire the other MPI domains
    dt = reduceMin(dt);

    return dt;
}
