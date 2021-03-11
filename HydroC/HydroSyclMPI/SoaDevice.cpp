//
// Implementation of Array of 2D arrays on devices

#include "SoaDevice.hpp"
#include "ParallelInfo.hpp"
#include "ParallelInfoOpaque.hpp"

#include <CL/sycl.hpp>

template <typename T>
SoaDevice<T>::SoaDevice(int w, int h, int variables) : m_w(w), m_h(h), m_nbvariables(variables) {
    m_array =
        sycl::malloc_device<T>(m_w * m_h * m_nbvariables, ParallelInfo::extraInfos()->m_queue);
    
}


template <typename T> SoaDevice<T>::~SoaDevice() {
    sycl::free(m_array,  ParallelInfo::extraInfos()->m_queue);
}

template <typename T>
Array2D<T>::Array2D(int32_t w, int32_t h) : m_w(w), m_h(h), m_managed_alloc(true) {
    m_data = sycl::malloc_device<T>(m_w * m_h, ParallelInfo::extraInfos()->m_queue);
}

template <typename T> Array2D<T>::~Array2D() {
    if (m_managed_alloc)
        sycl::free(m_data, ParallelInfo::extraInfos()->m_queue);
}

template <typename T> Array1D<T>::Array1D(int32_t lgr)  {
    
    m_data = sycl::malloc_device<T>(lgr, ParallelInfo::extraInfos()->m_queue);
}
template <typename T> Array1D<T>::~Array1D() {
    sycl::free(m_data, ParallelInfo::extraInfos()->m_queue);
}


const sycl::stream &operator<<(const sycl::stream &os, const Array2D<double> & mat) {

    int32_t srcX = mat.getW();
    int32_t srcY = mat.getH();
    os << " nx=" << srcX << " ny=" << srcY << "\n";
    for (int32_t j = 0; j < srcY; j++) {

        for (int32_t i = 0; i < srcX; i++) {
            os << mat(i,j) << " ";
        }
        os << "\n";
    }
    os << "\n\n";
    return os;
}



const sycl::stream &operator<<(const sycl::stream &os, const RArray2D<double> & mat) {
    int32_t srcX = mat.getW();
    int32_t srcY = mat.getH();
    os << " nx=" << srcX << " ny=" << srcY << "\n";
    for (int32_t j = 0; j < srcY; j++) {

        for (int32_t i = 0; i < srcX; i++) {
            os << mat(i,j) << " ";
        }
        os << "\n";
    }
    os << "\n\n";
    return os;
}

template <typename T> 
void Array2D<T>::putFullCol(int32_t x, int32_t offy, T* theCol, int32_t lgr)
{
  for (int32_t j = 0; j < lgr; j++) {
        m_data[(j+offy)*m_w+x] = theCol[j];
    }
}

template <>
void RArray2D<double>::putFullCol(int32_t x, int32_t offy, double * theCol, int32_t lgr)
{
  for (int32_t j = 0; j < lgr; j++) {
        m_data[(j+offy)*m_w+x] = theCol[j];
    }
}

template class Array1D<double>;
template class Array2D<double>;
template class SoaDevice<double>;
