//
// Special Classes for Array of 2D Arrays on devices
//

#ifndef SOAONDEVICE_H
#define SOAONDEVICE_H

#include "precision.hpp"

#include <algorithm>
#include <cstdint>
#include <ostream>

#include <CL/sycl.hpp>

template <typename T> class Array1D;
template <typename T> class Array2D;
template <typename T> class SoaDevice;

template <typename T> std::ostream &operator<<(std::ostream &, const Array1D<T> &);
template <typename T> std::ostream &operator<<(std::ostream &, const Array2D<T> &);
template <typename T> std::ostream &operator<<(std::ostream &, const SoaDevice<T> &);

template <typename T> const sycl::stream &operator<<(const sycl::stream &, const Array2D<T> &);

template <typename S> class Array1D {
    S *m_data; // This is a device address
    sycl::queue m_queue;

  public:
    Array1D() : m_data(nullptr){};
    Array1D(int32_t lgr, sycl::queue& queue);
    ~Array1D();

    SYCL_EXTERNAL
    S *data() { return m_data; }

    S &operator()(int32_t idx) { return m_data[idx]; }
    S operator()(int32_t idx) const { return m_data[idx]; }

    friend std::ostream &operator<<(std::ostream &, const Array1D<S> &);

    SYCL_EXTERNAL
    friend const sycl::stream &operator<<(const sycl::stream &, const Array1D<S> &);
};

template <typename S> class Array2D {
    S *m_data; // This is a device address
    sycl::queue m_queue;

    int32_t m_w;
    int32_t m_h;

    bool m_managed_alloc;

  public:
    Array2D() : m_data(nullptr), m_managed_alloc(false) {}
    Array2D(int32_t w, int32_t h, sycl::queue& );

    SYCL_EXTERNAL
    Array2D(S *val, int32_t w, int32_t h, sycl::queue &queue)
        : m_data(val), m_queue(queue), m_w(w), m_h(h), m_managed_alloc(false) {}

    SYCL_EXTERNAL
    Array2D(const Array2D &org)
        : m_data(org.m_data), m_queue(org.m_queue), m_w(org.m_w), m_h(org.m_h),
          m_managed_alloc(false) {}

    SYCL_EXTERNAL
    ~Array2D();

    SYCL_EXTERNAL
    void swapDimOnly() { std::swap(m_w, m_h); }

    SYCL_EXTERNAL
    S &operator()(int32_t i, int32_t j) { return m_data[j * m_w + i]; }

    SYCL_EXTERNAL
    S operator()(int32_t i, int32_t j) const { return m_data[j * m_w + i]; }

    SYCL_EXTERNAL
    S *getRow(int row) { return &m_data[row * m_w]; }

    SYCL_EXTERNAL
    void putFullCol(int32_t x, int32_t offy, S *thecol, int32_t l);

    friend std::ostream &operator<<(std::ostream &, const Array2D<S> &);

    SYCL_EXTERNAL
    friend const sycl::stream &operator<<(const sycl::stream &, const Array2D<S> &);
};

template <typename T> class SoaDevice {

    T *m_array; // This is a device adress
    sycl::queue m_queue;
    int32_t m_w, m_h, m_nbvariables;

  public:
    SoaDevice() : m_array(nullptr) {}
    SoaDevice(int w, int h, int variables, sycl::queue &);
    SoaDevice(const SoaDevice &org)
        : m_w(org.m_w), m_h(org.m_h), m_nbvariables(org.m_nbvariables), m_array(org.m_array),
          m_queue(org.m_queue) {}

    ~SoaDevice();

    SYCL_EXTERNAL
    T &operator()(int32_t v, int32_t i, int32_t j) { return m_array[v * m_h * m_w + j * m_w + i]; }

    SYCL_EXTERNAL
    T operator()(int32_t v, int32_t i, int32_t j) const {
        return m_array[v * m_h * m_w + j * m_w + i];
    }

    void swapDimOnly() { std::swap(m_h, m_w); }
    Array2D<T> operator()(int32_t var) {
        return Array2D<T>(m_array + var * m_h * m_w, m_w, m_h, m_queue);
    }
};

#endif