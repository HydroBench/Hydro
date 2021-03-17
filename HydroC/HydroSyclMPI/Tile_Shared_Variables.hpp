//
// Definition of a structure that should contains the shared memory of the different Tiles
//
// e.g. all constants + uold
//

// TODO: This is global variables, shame on me !

#ifndef TILESSHAREDVARIABLES
#define TILESSHAREDVARIABLES

#include "Device_Buffers.hpp"
#include "EnumDefs.hpp"
#include "SoaDevice.hpp"
#include "Timers.hpp"

struct TilesSharedVariables {

    real_t m_gamma, m_smallc, m_smallr, m_cfl;

    int32_t m_niter_riemann;
    int32_t m_order;
    real_t m_slope_type;
    int32_t m_scheme;

    // convenient variables

    int32_t m_prt;

    Timers *m_threadTimers; // one Timers per thread
    DeviceBuffers *m_device_buffers;

    SoaDevice<real_t> m_uold;

    void initPhys(real_t gamma, real_t smallc, real_t smallr, real_t cfl, real_t slope_type,
                  int32_t nIterRiemmann, int32_t order, int32_t scheme) {
        m_gamma = gamma;
        m_smallc = smallc;
        m_smallr = smallr;
        m_cfl = cfl;
        m_slope_type = slope_type;
        m_niter_riemann = nIterRiemmann;
        m_order = order;
        m_scheme = scheme;
    }

    SYCL_EXTERNAL
    void setTimes(Timers *pTimers) { m_threadTimers = pTimers; }

    SYCL_EXTERNAL
    DeviceBuffers *buffers(int workitem_idx) { return &m_device_buffers[workitem_idx]; }
};

extern TilesSharedVariables onHost, *onDevice;

#endif
