//
// (C) Guillaume.Colin-de-Verdiere at CEA.Fr
//

#include "precision.hpp"
#include "Device_Buffers.hpp"

#include <CL/sycl.hpp>
#include <algorithm>
#include <cstdint>
#include <iostream>

//

using namespace std;

void DeviceBuffers::init(int32_t xmin, int32_t xmax, int32_t ymin, int32_t ymax) {
    int32_t lgx, lgy, lgmax;

    lgx = (xmax - xmin);
    lgy = (ymax - ymin);

    lgmax = std::max(lgx, lgy);

    m_q = (SoaDevice<real_t>(NB_VAR, lgx, lgy));
    m_qxm = (SoaDevice<real_t>(NB_VAR, lgx, lgy));
    m_qxp = (SoaDevice<real_t>(NB_VAR, lgx, lgy));
    m_dq = (SoaDevice<real_t>(NB_VAR, lgx, lgy));
    m_qleft = (SoaDevice<real_t>(NB_VAR, lgx, lgy));
    m_qright = (SoaDevice<real_t>(NB_VAR, lgx, lgy));
    m_qgdnv = (SoaDevice<real_t>(NB_VAR, lgx, lgy));
    m_c = (Array2D<real_t>(lgx, lgy));
    m_e = (Array2D<real_t>(lgx, lgy));

    m_sgnm = (Array1D<real_t>(lgmax));
    m_pl = (Array1D<real_t>(lgmax));
    m_swapped = false;
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
    m_swapped = !m_swapped;
}

void DeviceBuffers::firstTouch() {
    // Should do something on the buffer here to force allocation
}
// EOF
