//
// (C) Guillaume.Colin-de-Verdiere at CEA.Fr
//

#include "Matrix.hpp"
#include "Options.hpp"
#include "Utilities.hpp"

#include <cstdio>
#include <cstdlib>
#include <iomanip>

#include <unistd.h>

#ifdef MPI_ON
#include <mpi.h>
#endif

long Volume::_volume = 0;
long Volume::_volumeMax = 0;

template <typename T> void Matrix2<T>::allocate(void) {
    int32_t lgrow = 0;
    int32_t maxPad = 0, padh = 0;
    static int decal = 0;

    assert((_w * _h) > 0);

#if WITHHBW == 0 && WITHPOSIX == 0 && WITHNEW == 0
#define WITHPOSIX 1
#endif
    size_t lgrTab = (_w * _h) * sizeof(T);
#ifdef WITHNEW
    _arr = new T[_w * _h];
    _org = _arr;
// #pragma message "C++ NEW usage activated"
#endif

#ifdef WITHPOSIX
    // #pragma message "posix_memalign activated"
    int rc = posix_memalign((void **)&_arr, _nbloc, lgrTab + _nbloc);
    _org = _arr;
#endif
    _volume += lgrTab;
    _volumeMax = std::max(_volume, _volumeMax);

    memset(_arr, 0, lgrTab);
    assert(_arr != 0);
}

template <typename T> void Matrix2<T>::swapDimOnly() { std::swap(_w, _h); }

template <typename T> void Matrix2<T>::swapDimAndValues() {
    int32_t t = _w;
    _w = _h;
    _h = t;
    abort(); // not yet implemented
}

#ifndef ALIGNEXT
#define ALIGNEXT 128
#endif

template <typename T> Matrix2<T>::Matrix2(int32_t w, int32_t h) : _w(w), _h(h), _arr(0), _org(0) {
    // padd the array to make the next row aligned too.
    _w += (((_w * sizeof(T)) % ALIGNEXT) / sizeof(T));
    _h += (((_h * sizeof(T)) % ALIGNEXT) / sizeof(T));
    _w += (_w % 2);
    _h += (_h % 2);
    allocate();
    // std::cerr << "create " << this << std::endl;
}

template <typename T> void Matrix2<T>::fill(T v) {
    int32_t i, j;

    T *tmp = _arr;

    for (j = 0; j < _h; j++) {
        // #pragma simd
        for (i = 0; i < _w; i++) {
            tmp[Mat2Index(i, j)] = v;
        }
    }
    return;
}

template <typename T> Matrix2<T>::~Matrix2() {
    // std::cerr << "Destruction object " << this << std::endl;
    assert(_arr != 0);
    size_t lgrTab = (_w * _h) * sizeof(T);
    _volume -= lgrTab;

#ifdef WITHNEW
    delete[] _org;
#endif
#ifdef WITHHBW
    hbw_free(_org);
#endif
#if defined(WITHPOSIX)
    free(_org);
#endif
    _arr = 0;
    _org = 0;
}

template <typename T>
Matrix2<T>::Matrix2(const Matrix2<T> &m) : _w(m._w), _h(m._h), _arr(0), _org(0) {
    allocate();
    assert(_arr != 0);
    memcpy(_arr, m._arr, _w * _h * sizeof(T));
}

template <typename T> int32_t *Matrix2<T>::listMortonIdx(void) {
    // int32_t x, y;
    int32_t maxmorton = maxMorton();
    int32_t seen = 0;
    int32_t *list = new int32_t[nbElem()];

#pragma novector
    for (int32_t i = 0; i < maxmorton; ++i) {
        int32_t x, y;
        if (idxFromMorton(x, y, i)) {
            list[seen] = i;
            seen++;
        }
    }
    return list;
}

template <typename T> void Matrix2<T>::Copy(const Matrix2 &src) {
    for (int32_t j = 0; j < _h; j++) {
        // #pragma simd
        for (int32_t i = 0; i < _w; i++) {
            _arr[Mat2Index(i, j)] = src._arr[Mat2Index(i, j)];
        }
    }
}

template <typename T>
void Matrix2<T>::aux_getFullCol(int32_t x, int32_t h, T *__restrict__ theCol,
                                T *__restrict__ theArr) {
    // #pragma simd
    for (int32_t j = 0; j < h; j++) {
        theCol[j] = theArr[Mat2Index(x, j)];
    }
}

template <typename T> void Matrix2<T>::getFullCol(int32_t x, T *theCol) {
    aux_getFullCol(x, _h, theCol, _arr);
}

template <typename T>
void Matrix2<T>::aux_putFullCol(int32_t x, int32_t h, int32_t offY, T *__restrict__ theCol,
                                T *__restrict__ theArr) {
#if USEMKL == 1
    if (sizeof(real_t) == sizeof(double))
        vdUnpackI(h, (double *)theCol, (double *)&theArr[Mat2Index(x, offY)], Mat2Index(0, 1));
    if (sizeof(real_t) == sizeof(float))
        vsUnpackI(h, (float *)theCol, (float *)&theArr[Mat2Index(x, offY)], Mat2Index(0, 1));
#else
    // #pragma simd
    for (int32_t j = 0; j < h; j++) {
        theArr[Mat2Index(x, j + offY)] = theCol[j];
    }
#endif
}

template <typename T> void Matrix2<T>::putFullCol(int32_t x, int32_t offY, T *theCol, int32_t l) {
    aux_putFullCol(x, l, offY, theCol, _arr);
}

template <typename T> std::ostream &operator<<(std::ostream &os, const Matrix2<T> &mat) {
    int32_t srcX = mat.getW();
    int32_t srcY = mat.getH();
    os << " nx=" << srcX << " ny=" << srcY << std::endl;
    for (int32_t j = 0; j < srcY; j++) {
#pragma novector
        for (int32_t i = 0; i < srcX; i++) {
            os << std::setw(12) << std::setiosflags(std::ios::scientific) << std::setprecision(4)
               << mat._arr[mat.Mat2Index(i, j)] << " ";
        }
        os << std::endl;
    }
    os << std::endl << std::endl;
    return os;
}

template <typename T> long Matrix2<T>::getLengthByte() {
    int32_t srcX = getW();
    int32_t srcY = getH();
    return srcX * srcY * sizeof(T);
}

template <typename T> void Matrix2<T>::read(const int f) {
    int32_t srcX = getW();
    int32_t srcY = getH();
#pragma novector
    for (int j = 0; j < srcY; j++) {
        int l = ::read(f, &_arr[Mat2Index(0, j)], srcX * sizeof(T));
    }
}

template <typename T> void Matrix2<T>::write(const int f) {
    int32_t srcX = getW();
    int32_t srcY = getH();
#pragma novector
    for (int j = 0; j < srcY; j++) {
        int l = ::write(f, &_arr[Mat2Index(0, j)], srcX * sizeof(T));
    }
}

//
// Class instanciation: we force the compiler to generate the proper externals.
//

template class Matrix2<double>;
template class Matrix2<float>;
template class Matrix2<int32_t>;

template std::ostream &operator<<(std::ostream &, const Matrix2<double> &);
template std::ostream &operator<<(std::ostream &, const Matrix2<float> &);
// EOF
