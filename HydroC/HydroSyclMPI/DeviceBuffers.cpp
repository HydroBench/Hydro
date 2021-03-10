//
// (C) Guillaume.Colin-de-Verdiere at CEA.Fr
//
#include "Options.hpp"
#include "DeviceBuffers.hpp"

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
   
    m_q = new SoaDevice<real_t>(NB_VAR, lgx, lgy);
    m_qxm = new SoaDevice<real_t>(NB_VAR, lgx, lgy);
    m_qxp = new SoaDevice<real_t>(NB_VAR, lgx, lgy);
    m_dq = new SoaDevice<real_t>(NB_VAR, lgx, lgy);

    m_qleft = new SoaDevice<real_t>(NB_VAR, lgx, lgy);
    m_qright = new SoaDevice<real_t>(NB_VAR, lgx, lgy);
    m_qgdnv = new SoaDevice<real_t>(NB_VAR, lgx, lgy);

    m_c = new Array2D<real_t>(lgx, lgy);
    m_e = new Array2D<real_t>(lgx, lgy);

    m_sgnm = new Array1D<real_t>(lgmax);
    m_pl = new Array1D<real_t>(lgmax);

}

DeviceBuffers::~DeviceBuffers() {
    delete m_q;
    delete m_qxm;
    delete m_qxp;
    delete m_dq;
    delete m_qleft;
    delete m_qright;
    delete m_qgdnv;
    delete m_c;
    delete m_e;

    delete m_sgnm;
    delete m_pl;

}

void DeviceBuffers::swapStorageDims() {

    m_q->swapDimOnly();
    m_qxm->swapDimOnly();
    m_qxp->swapDimOnly();
    m_dq->swapDimOnly();
    m_qleft->swapDimOnly();
    m_qright->swapDimOnly();
    m_qgdnv->swapDimOnly();

    m_c->swapDimOnly();
    m_e->swapDimOnly();
}

// EOF
