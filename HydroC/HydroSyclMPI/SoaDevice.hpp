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

  public:
    Array1D() : m_data(nullptr){};
    Array1D(int32_t lgr);
    ~Array1D();

    // for move operation
    Array1D &operator=(Array1D &&other) {

        m_data = other.m_data;
        other.m_data = nullptr;
        return *this;
    }

    SYCL_EXTERNAL
    S *data() { return m_data; }

    S &operator()(int32_t idx) { return m_data[idx]; }
    S operator()(int32_t idx) const { return m_data[idx]; }

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
    RArray2D &operator=(const RArray2D &) = delete;

    SYCL_EXTERNAL
    RArray2D(S *val, int32_t w, int32_t h) : m_data(val), m_w(w), m_h(h) {}

    SYCL_EXTERNAL
    ~RArray2D() {}

    SYCL_EXTERNAL
    int getW() const { return m_w; }

    SYCL_EXTERNAL
    int getH() const { return m_h; }

    SYCL_EXTERNAL
    void swapDimOnly() {
        int32_t t = m_w;
        m_w = m_h;
        m_h = t;
    }

    SYCL_EXTERNAL
    S &operator()(int32_t i, int32_t j) { return m_data[j * m_w + i]; }

    SYCL_EXTERNAL
    S operator()(int32_t i, int32_t j) const { return m_data[j * m_w + i]; }

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

  public:
    Array2D() : m_data(nullptr), m_managed_alloc(false) { m_h = m_w = -1; }

    Array2D(int32_t w, int32_t h);

#if 0
	SYCL_EXTERNAL
	Array2D(S *val, int32_t w, int32_t h) :
			m_data(val), m_w(w), m_h(h), m_managed_alloc(false) {
	}
#endif

    SYCL_EXTERNAL
    Array2D(const Array2D &org) = delete;

    // for move operation
    Array2D &operator=(Array2D &&rhs) {
        m_data = rhs.m_data;
        m_w = rhs.m_w;
        m_h = rhs.m_h;
        m_managed_alloc = rhs.m_managed_alloc;
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
    }

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

    int32_t m_w, m_h, m_nbvariables;
    bool m_managed;

  public:
    SoaDevice() : m_array(nullptr), m_managed(false) { m_w = m_h = m_nbvariables = -1; }
    SoaDevice(int w, int h, int variables);
    SoaDevice(const SoaDevice &org) = delete;

    // for move operation;
    SoaDevice &operator=(SoaDevice &&rhs) {

        m_w = rhs.m_w;
        m_h = rhs.m_h;
        m_nbvariables = rhs.m_nbvariables;
        m_array = rhs.m_array;
        m_managed = rhs.m_managed;
        rhs.m_managed = false;
        rhs.m_array = nullptr;
        return *this;
    }

    ~SoaDevice();

    SYCL_EXTERNAL
    T &operator()(int32_t v, int32_t i, int32_t j) { return m_array[v * m_h * m_w + j * m_w + i]; }

    SYCL_EXTERNAL
    T operator()(int32_t v, int32_t i, int32_t j) const {
        return m_array[v * m_h * m_w + j * m_w + i];
    }

    void swapDimOnly() { std::swap(m_h, m_w); }
    RArray2D<T> operator()(int32_t var) { return RArray2D<T>(m_array + var * m_h * m_w, m_w, m_h); }
};

#endif
