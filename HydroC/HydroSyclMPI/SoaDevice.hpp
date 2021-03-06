//
// Special Classes for Array of 2D Arrays on devices
//

#ifndef SOAONDEVICE_H
#define SOAONDEVICE_H

#include "precision.hpp"

#include <algorithm>
#include <cstdint>
#include <iostream>
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
    int32_t m_lgr;

  public:
    Array1D() : m_data(nullptr), m_lgr(0){};
    Array1D(int32_t lgr);
    ~Array1D();

    Array1D &operator=(const Array1D &) = delete;

    // for move operation
    Array1D &operator=(Array1D &&other) {
        m_lgr = other.m_lgr;
        m_data = other.m_data;
        other.m_data = nullptr;
        return *this;
    }

    SYCL_EXTERNAL
    S *data() { return m_data; }

    SYCL_EXTERNAL
    friend std::ostream &operator<<(std::ostream &, const Array1D<S> &);

    SYCL_EXTERNAL
    friend const sycl::stream &operator<<(const sycl::stream &, const Array1D<S> &);
};

// This is only the view of SOADevice

template <typename S> class RArray2D {
    S *m_data; // This is a device address
    int32_t m_w;
    int32_t m_h;

  public:
    RArray2D() = delete;
    RArray2D(const RArray2D &org) : m_data(org.m_data), m_w(org.m_w), m_h(org.m_h){};

    RArray2D &operator=(const RArray2D &) = delete;

    SYCL_EXTERNAL
    ~RArray2D() {} // It  is a view not a kill (JB007)

    S *data() { return m_data; }

    SYCL_EXTERNAL
    RArray2D(S *val, int32_t w, int32_t h) : m_data(val), m_w(w), m_h(h) {}

    SYCL_EXTERNAL
    int getW() const { return m_w; }

    SYCL_EXTERNAL
    int getH() const { return m_h; }

    SYCL_EXTERNAL
    S &operator()(int32_t i, int32_t j) { return m_data[j * m_w + i]; }

    SYCL_EXTERNAL
    S operator()(int32_t i, int32_t j) const { return m_data[j * m_w + i]; }

    SYCL_EXTERNAL
    S *operator[](int row) { return &m_data[row * m_w]; }

    SYCL_EXTERNAL
    S *getRow(int row) { return &m_data[row * m_w]; }

    SYCL_EXTERNAL
    void putFullCol(int32_t x, int32_t offy, S *thecol, int32_t l);

    SYCL_EXTERNAL
    friend const sycl::stream &operator<<(const sycl::stream &, const RArray2D<S> &);
};

template <typename S> class Array2D {
    S *m_data; // This is a device address
    int32_t m_w;
    int32_t m_h;

    bool m_managed_alloc;
    bool m_swapped;

  public:
    Array2D() : m_data(nullptr), m_managed_alloc(false), m_swapped(false) { m_h = m_w = -1; }
    Array2D(int32_t w, int32_t h);

    Array2D(const Array2D &org) = delete;

    Array2D &operator=(const Array2D &) = delete;

    // for move operation
    Array2D &operator=(Array2D &&rhs) {
        m_data = rhs.m_data;
        m_w = rhs.m_w;
        m_h = rhs.m_h;
        m_managed_alloc = rhs.m_managed_alloc;
        m_swapped = rhs.m_swapped;

        rhs.m_data = nullptr;
        rhs.m_managed_alloc = false;

        return *this;
    }

    ~Array2D();

    SYCL_EXTERNAL
    int getW() const { return m_w; }

    SYCL_EXTERNAL
    int getH() const { return m_h; }

    SYCL_EXTERNAL
    void swapDimOnly() {
        int32_t t = m_w;
        m_w = m_h;
        m_h = t;
        m_swapped = !m_swapped;
    }

    SYCL_EXTERNAL
    S operator()(int32_t i, int32_t j) const { return m_data[j * m_w + i]; }

    SYCL_EXTERNAL
    S *getRow(int row) { return &m_data[row * m_w]; }

    S *data() { return m_data; }

    friend std::ostream &operator<<(std::ostream &, const Array2D<S> &);

    SYCL_EXTERNAL
    friend const sycl::stream &operator<<(const sycl::stream &, const Array2D<S> &);
};

template <typename T> class SoaDevice {

    T *m_array; // This is a device adress

    int32_t m_w, m_h, m_nbvariables;
    int32_t m_2Dsize;
    bool m_managed;
    bool m_swapped;

  public:
    SoaDevice() : m_array(nullptr), m_managed(false), m_swapped(false) {
        m_w = m_h = m_nbvariables = m_2Dsize = -1;
    }
    SoaDevice(int variables, int32_t w, int32_t h);
    SoaDevice(const SoaDevice &org) = delete;
    SoaDevice &operator=(const SoaDevice &) = delete;

    T *data() { return m_array; }
    // for move operation;
    SoaDevice &operator=(SoaDevice &&rhs) {

        m_w = rhs.m_w;
        m_h = rhs.m_h;
        m_2Dsize = m_w * m_h;
        m_nbvariables = rhs.m_nbvariables;
        m_array = rhs.m_array;
        m_managed = rhs.m_managed;
        m_swapped = rhs.m_swapped;

        rhs.m_managed = false;
        rhs.m_array = nullptr;
        return *this;
    }

    ~SoaDevice();

    SYCL_EXTERNAL
    void swapDimOnly() {
        int t = m_w;
        m_w = m_h;
        m_h = t;
        m_swapped = !m_swapped;
    }

    // Provide a view to the matrix2D for var
    SYCL_EXTERNAL
    RArray2D<T> operator()(int32_t var) { return RArray2D<T>(m_array + var * m_2Dsize, m_w, m_h); }
};

#endif
