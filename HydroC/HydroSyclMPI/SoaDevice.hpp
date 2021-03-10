//
// Special Classes for Array of 2D Arrays on devices
//

#ifndef SOAONDEVICE_H
#define SOAONDEVICE_H

#include "precision.hpp"

#include <algorithm>
#include <cstdint>

template <typename S> class Array1D {
    S * m_data;
public:
    Array1D() = delete;
    Array1D(int32_t lgr);
    ~Array1D();
    S * data () { return m_data;}
    
    S & operator() (int32_t idx) { return m_data[idx];}
    S operator() (int32_t idx) const  { return m_data[idx];}

};

template <typename S> class Array2D {
    S *m_data;
    
    int32_t m_w;
    int32_t m_h;

    bool m_managed_alloc;

  public:
    Array2D() = delete;
    Array2D(int32_t w, int32_t h);
    Array2D(S *val, int32_t w, int32_t h) : m_data(val), m_w(w), m_h(h), m_managed_alloc(false) {}

    ~Array2D();
    void swapDimOnly() { std::swap(m_w, m_h); }
    S &operator()(int32_t i, int32_t j) { return m_data[j * m_w + i]; }
    S operator()(int32_t i, int32_t j) const { return m_data[j * m_w + i]; }
};

template <typename T> class SoaDevice {

    int32_t m_w, m_h, m_nbvariables;
    T *m_array;

  public:
    SoaDevice() = delete;
    SoaDevice(int w, int h, int variables);
    ~SoaDevice();

    T &operator()(int32_t v, int32_t i, int32_t j) { return m_array[v * m_h * m_w + j * m_w + i]; }
    T operator()(int32_t v, int32_t i, int32_t j) const {
        return m_array[v * m_h * m_w + j * m_w + i];
    }

    void swapDimOnly() { std::swap(m_h, m_w); }
    Array2D<T> &operator()(int32_t var) { return Array2D<T>(m_array + var * m_h * m_w, m_w); }
};

#endif