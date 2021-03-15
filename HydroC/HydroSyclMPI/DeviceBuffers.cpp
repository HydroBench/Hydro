//
// (C) Guillaume.Colin-de-Verdiere at CEA.Fr
//

#include "DeviceBuffers.hpp"

#include "precision.hpp"

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

    m_q = std::move(SoaDevice<real_t>(NB_VAR, lgx, lgy));
    m_qxm = std::move(SoaDevice<real_t>(NB_VAR, lgx, lgy));
    m_qxp = std::move(SoaDevice<real_t>(NB_VAR, lgx, lgy));
    m_dq = std::move(SoaDevice<real_t>(NB_VAR, lgx, lgy));
    m_qleft = std::move(SoaDevice<real_t>(NB_VAR, lgx, lgy));
    m_qright = std::move(SoaDevice<real_t>(NB_VAR, lgx, lgy));
    m_qgdnv = std::move(SoaDevice<real_t>(NB_VAR, lgx, lgy));
    m_c = std::move(Array2D<real_t>(lgx, lgy));
    m_e = std::move(Array2D<real_t>(lgx, lgy));

    m_sgnm = std::move(Array1D<real_t>(lgmax));
    m_pl = std::move(Array1D<real_t>(lgmax));
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
