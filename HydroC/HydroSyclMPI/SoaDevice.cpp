//
// Implementation of Array of 2D arrays on devices

#include "SoaDevice.hpp"
#include "ParallelInfo.hpp"
#include "ParallelInfoOpaque.hpp"

#include <CL/sycl.hpp>

template <typename T>
SoaDevice<T>::SoaDevice(int w, int h, int variables, sycl::queue & queue) : m_w(w), m_h(h), m_nbvariables(variables) {
    m_array =
        sycl::malloc_device<T>(m_w * m_h * m_nbvariables, queue);
    m_queue = queue;
}


template <typename T> SoaDevice<T>::~SoaDevice() {
    sycl::free(m_array, m_queue);
}

template <typename T>
Array2D<T>::Array2D(int32_t w, int32_t h, sycl::queue & queue) : m_w(w), m_h(h), m_managed_alloc(true) {
    m_data = sycl::malloc_device<T>(m_w * m_h, *m_queue);
}

template <typename T> Array2D<T>::~Array2D() {
    if (m_managed_alloc)
        sycl::free(m_data, m_queue);
}

template <typename T> Array1D<T>::Array1D(int32_t lgr, sycl::queue & queue)  {
    m_queue = queue;
    m_data = sycl::malloc_device<T>(lgr, m_queue);
}
template <typename T> Array1D<T>::~Array1D() {
    sycl::free(m_data, m_queue);
}