//
//
// Some Transfers to and from Device
//

#include "Domain.hpp"

#include "ParallelInfo.hpp"
#include "ParallelInfoOpaque.hpp"

#include "Tile_Shared_Variables.hpp"

#include <CL/sycl.hpp>

void Domain::sendUoldToDevice() {

    sycl::queue queue = ParallelInfo::extraInfos()->m_queue;

    for (int var = 0; var < NB_VAR; var++) {
        auto &matrice = *(*m_uold)(var);

        RArray2D<real_t> matDevice = onHost.m_uold(var);

        assert(matrice.getH() == matDevice.getH());
        assert(matrice.getW() == matDevice.getW());

        queue.submit([&](auto &handler) {
            handler.memcpy(matrice.data(), matDevice.data(),
                           matDevice.getH() * matDevice.getW() * sizeof(real_t));
        });
    }
}

void Domain::getUoldFromDevice() {
    sycl::queue queue = ParallelInfo::extraInfos()->m_queue;

    for (int var = 0; var < NB_VAR; var++) {
        auto matrice = *(*m_uold)(var);

        RArray2D<real_t> matDevice = onHost.m_uold(var);

        assert(matrice.getH() == matDevice.getH());
        assert(matrice.getW() == matDevice.getW());

        queue.submit([&](auto &handler) {
            handler.memcpy(matDevice.data(), matrice.data(),
                           matDevice.getH() * matDevice.getW() * sizeof(real_t));
        });
    }
}
