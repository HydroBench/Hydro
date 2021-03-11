//
// (C) Guillaume.Colin-de-Verdiere at CEA.Fr
//
#include "Options.hpp"
#include "DeviceBuffers.hpp"
#include "ParallelInfo.hpp"
#include "ParallelInfoOpaque.hpp"
#include <CL/sycl.hpp>

//

using namespace std;

inline void *memsetth(void *s, int c, size_t n) {
    char *ptr = (char *)s;
    char cval = c;
#pragma omp simd
    for (int32_t i = 0; i < n; i++) {
        ptr[i] = cval;
    }
    return s;
}

DeviceBuffers::DeviceBuffers(int32_t xmin, int32_t xmax, int32_t ymin, int32_t ymax) {
    int32_t lgx, lgy, lgmax;

    lgx = (xmax - xmin);
    lgy = (ymax - ymin);
    
    lgmax = std::max(lgx, lgy);
   
    sycl::queue & q = ParallelInfo::extraInfos()->m_queue;

    m_q =  SoaDevice<real_t>(NB_VAR, lgx, lgy,q);
    m_qxm =  SoaDevice<real_t>(NB_VAR, lgx, lgy,q);
    m_qxp = SoaDevice<real_t>(NB_VAR, lgx, lgy,q);
    m_dq =  SoaDevice<real_t>(NB_VAR, lgx, lgy,q);
    m_qleft =  SoaDevice<real_t>(NB_VAR, lgx, lgy,q);
    m_qright = SoaDevice<real_t>(NB_VAR, lgx, lgy,q);
    m_qgdnv =  SoaDevice<real_t>(NB_VAR, lgx, lgy,q);

    m_c =  Array2D<real_t>(lgx, lgy,q);
    m_e =  Array2D<real_t>(lgx, lgy,q);

    m_sgnm =  Array1D<real_t>(lgmax,q);
    m_pl =  Array1D<real_t>(lgmax,q);

}

DeviceBuffers::~DeviceBuffers() {
    // TODO: Somehow whe have to decide what to do at one point :-)
}

void DeviceBuffers::swapStorageDims() {


    m_q.swapDimOnly();
    m_qxm.swapDimOnly();
    m_qxp.swapDimOnly();
    m_dq.swapDimOnly();
    m_qleft.swapDimOnly();
    m_qright.swapDimOnly();
    m_qgdnv.swapDimOnly();

    m_c.swapDimOnly();
    m_e.swapDimOnly();
}

// EOF
