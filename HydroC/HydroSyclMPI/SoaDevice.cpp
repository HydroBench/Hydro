//
// Implementation of Array of 2D arrays on devices
//

#include "SoaDevice.hpp"
#include "ParallelInfo.hpp"
#include "ParallelInfoOpaque.hpp"

#include <CL/sycl.hpp>

// #define ALIGNEXT 128

inline void adjust(int32_t &_w, int32_t &_h, int32_t size) {
#ifdef ALIGNEXT
    int nb_elt_per_align = ALIGNEXT / size;
    int remain = _w % nb_elt_per_align;
    if (remain)
        _w += (nb_elt_per_align - remain);
    remain = _h % nb_elt_per_align;
    if (remain)
        _h += (nb_elt_per_align - remain);
#endif
}

template <typename T>
SoaDevice<T>::SoaDevice(int variables, int32_t w, int32_t h)
    : m_w(w), m_h(h), m_nbvariables(variables) {
    adjust(m_w, m_h, sizeof(T));
    m_2Dsize = m_w * m_h;
#ifdef ALIGNEXT
    m_array = sycl::aligned_alloc_device<T>(ALIGNEXT, m_2Dsize * m_nbvariables,
                                            ParallelInfo::extraInfos()->m_queue);
#else
    m_array = sycl::malloc_device<T>(m_2Dsize * m_nbvariables, ParallelInfo::extraInfos()->m_queue);
#endif

    m_managed = true;
    m_swapped = false;

#if 0
    auto queue = ParallelInfo::extraInfos()->m_queue;
    queue.submit(
        [&](sycl::handler &h) { h.memset(m_array, 0, m_w * m_h * m_nbvariables * sizeof(T)); }).wait();
#endif
}

template <typename T> SoaDevice<T>::~SoaDevice() {

    if (m_array != nullptr && m_managed) {

        sycl::free(m_array, ParallelInfo::extraInfos()->m_queue);
    }
}

template <typename T>
Array2D<T>::Array2D(int32_t w, int32_t h) : m_w(w), m_h(h), m_managed_alloc(true) {
    adjust(m_w, m_h, sizeof(T));

#ifdef ALIGNEXT
    m_data =
        sycl::aligned_alloc_device<T>(ALIGNEXT, m_w * m_h, ParallelInfo::extraInfos()->m_queue);
#else
    m_data = sycl::malloc_device<T>(m_w * m_h, ParallelInfo::extraInfos()->m_queue);
#endif

    m_swapped = false;

#if 0
    auto queue = ParallelInfo::extraInfos()->m_queue;
    queue.submit([&](sycl::handler &h) { h.memset(m_data, 0, m_w * m_h * sizeof(T)); }).wait();
#endif
}

template <typename T> Array2D<T>::~Array2D() {

    if (m_managed_alloc)
        sycl::free(m_data, ParallelInfo::extraInfos()->m_queue);
}

template <typename T> Array1D<T>::Array1D(int32_t lgr) {
    m_data = sycl::malloc_device<T>(lgr, ParallelInfo::extraInfos()->m_queue);
    m_lgr = lgr;
}

template <typename T> Array1D<T>::~Array1D() {

    if (m_data != nullptr)
        sycl::free(m_data, ParallelInfo::extraInfos()->m_queue);
}

SYCL_EXTERNAL
const sycl::stream &operator<<(const sycl::stream &os, const Array2D<real_t> &mat) {

    int32_t srcX = mat.getW();
    int32_t srcY = mat.getH();
    os << " nx=" << srcX << " ny=" << srcY << sycl::endl;
    int width = os.get_width();
    for (int32_t j = 0; j < srcY; j++) {

        for (int32_t i = 0; i < srcX; i++) {

            os << sycl::setw(12) << sycl::scientific << sycl::setprecision(4) << mat(i, j) << " ";
        }
        os << sycl::stream_manipulator::endl;
    }
    os << sycl::setw(width) << sycl::stream_manipulator::endl << sycl::stream_manipulator::endl;
    return os;
}

SYCL_EXTERNAL
const sycl::stream &operator<<(const sycl::stream &os, const RArray2D<real_t> &mat) {
    int32_t srcX = mat.getW();
    int32_t srcY = mat.getH();
    int32_t width = os.get_width();
    os << " nx=" << srcX << " ny=" << srcY << "\n";
    for (int32_t j = 0; j < srcY; j++) {

        for (int32_t i = 0; i < srcX; i++) {
            os << sycl::setw(12) << sycl::scientific << sycl::setprecision(4) << mat(i, j) << " ";
        }
        os << sycl::stream_manipulator::endl;
    }
    os << sycl::setw(width) << sycl::stream_manipulator::endl
       << sycl::stream_manipulator::endl
       << sycl::stream_manipulator::flush;
    return os;
}

template <>
void RArray2D<real_t>::putFullCol(int32_t x, int32_t offy, real_t *theCol, int32_t lgr) {
    for (int32_t j = 0; j < lgr; j++) {
        m_data[(j + offy) * m_w + x] = theCol[j];
    }
}

template class Array1D<real_t>;
template class Array2D<real_t>;
template class SoaDevice<real_t>;
