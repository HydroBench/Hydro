//
// (C) Guillaume.Colin-de-Verdiere at CEA.Fr
//
#ifdef MPI_ON
#include <mpi.h>
#endif
#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <string.h>
#include <strings.h>
#include <unistd.h>
#include <stdlib.h>
#include <malloc.h>
#include <assert.h>
#include <limits.h>
#include <sys/time.h>
#include <float.h>
#include <math.h>

#if USEMKL == 1
#include <mkl.h>
#endif

#ifdef WITHHBW
#include <hbwmalloc.h>
#endif

using namespace std;

//
#include "Options.hpp"
#include "Utilities.hpp"
#include "Matrix.hpp"

template < typename T > void Matrix2 < T >::allocate(void)
{
	int32_t lgrow = 0;
	int32_t maxPad = 0, padh = 0;
	static int decal = 0;


#if WITHHBW==0 && WITHPOSIX == 0 && WITHNEW == 0
#define WITHPOSIX 1
#endif
	size_t lgrTab = (_w * _h) * sizeof(T);
#ifdef WITHNEW
	_arr = new T[_w * _h];
#pragma message "C++ NEW usage activated"
#endif
#ifdef WITHHBW
	int rc = hbw_posix_memalign((void **) &_arr, _nbloc, lgrTab + _nbloc);
#pragma message "HBW memory usage activated"
#endif
#ifdef WITHPOSIX
#pragma message "posix_memalign activated"
	int rc = posix_memalign((void **) &_arr, _nbloc, lgrTab + _nbloc);
#endif
#if defined(WITHPOSIX) || defined(WITHHBW)
	char *tmp = (char *) _arr;
	tmp += decal;
	decal += _ninc;
	if (decal >= _nbloc) decal = 0;
	_arr = (T*) tmp;
#endif
	memset(_arr, 0, lgrTab);
	assert(_arr != 0);
	_arr = _arr;
}

template < typename T > void Matrix2 < T >::swapDimOnly()
{
	Swap(_w, _h);
}

template < typename T > void Matrix2 < T >::swapDimAndValues()
{
	int32_t t = _w;
	_w = _h;
	_h = _w;
	abort();		// not yet implemented
}

 template < typename T > Matrix2 < T >::Matrix2(int32_t w, int32_t h):
_w(w), _h(h)
{
	allocate();
	// std::cerr << "create " << this << std::endl;
}

template < typename T > void Matrix2 < T >::fill(T v)
{
	int32_t i, j;
#ifdef _OPENMP
	int embedded = 0;	// to make it openmp proof 
#endif

#ifdef _OPENMP
	embedded = omp_in_parallel();
#endif
	T *tmp = _arr;
#ifdef _OPENMP
#pragma omp parallel for shared(tmp) private(i,j) if (!embedded) SCHEDULE
#endif
	for (j = 0; j < _h; j++) {
// #pragma simd
		for (i = 0; i < _w; i++) {
			tmp[Mat2Index(i, j)] = v;
		}
	}
	return;
}

template < typename T > Matrix2 < T >::~Matrix2()
{
	// std::cerr << "Destruction object " << this << std::endl;
	assert(_arr != 0);

#if defined(WITHPOSIX) || defined(WITHHBW)
	size_t tmp = (size_t) _arr;
	tmp = tmp >> _nshift;
	tmp = tmp << _nshift;
	_arr = (T*) tmp;
#endif
#ifdef WITHNEW
	delete[]_arr;
#endif
#ifdef WITHHBW
	hbw_free(_arr);
#endif
#if defined(WITHPOSIX)
	free(_arr);
#endif
	_arr = NULL;
}

template < typename T > Matrix2 < T >::Matrix2(const Matrix2 < T > &m)
{
	// std::cerr << "copy op " << this << std::endl;
	_w = (m._w);
	_h = (m._h);
	allocate();
	assert(_arr != 0);
	memcpy(_arr, m._arr, _w * _h * sizeof(T));
}

template < typename T > Matrix2 < T > &Matrix2 < T >::operator=(const Matrix2 < T > &rhs)
{
	// std::cerr << "= op " << this << std::endl;
	_w = (rhs._w);
	_h = (rhs._h);
	allocate();
	assert(_arr != 0);
	memcpy(_arr, rhs._arr, _w * _h * sizeof(T));
	return *this;
}

template < typename T > int32_t * Matrix2 < T >::listMortonIdx(void)
{
	int32_t x, y;
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

template < typename T > void Matrix2 < T >::Copy(const Matrix2 & src)
{
	for (int32_t j = 0; j < _h; j++) {
// #pragma simd
		for (int32_t i = 0; i < _w; i++) {
			_arr[Mat2Index(i, j)] = src._arr[Mat2Index(i, j)];
		}
	}
}

template < typename T > void Matrix2 < T >::aux_getFullCol(int32_t x, int32_t h, T * __restrict__ theCol, T * __restrict__ theArr)
{
// #pragma simd
	for (int32_t j = 0; j < h; j++) {
		theCol[j] = theArr[Mat2Index(x, j)];
	}
}

template < typename T > void Matrix2 < T >::getFullCol(int32_t x, T * theCol)
{
	aux_getFullCol(x, _h, theCol, _arr);
}

template < typename T > void Matrix2 < T >::aux_putFullCol(int32_t x, int32_t h, int32_t offY, T * __restrict__ theCol, T * __restrict__ theArr)
{
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

template < typename T > void Matrix2 < T >::putFullCol(int32_t x, int32_t offY, T * theCol, int32_t l)
{
	aux_putFullCol(x, l, offY, theCol, _arr);
}

template < typename T > void Matrix2 < T >::InsertMatrix(const Matrix2 & src, int32_t x0, int32_t y0)
{
	int32_t srcX = src.getW();
	int32_t srcY = src.getH();
	for (int32_t j = 0; j < srcY; j++) {
// #pragma simd
		for (int32_t i = 0; i < srcX; i++) {
			_arr[Mat2Index(i + x0, j + y0)] = src._arr[Mat2Index(i, j)];
		}
	}
}

template < typename T > void Matrix2 < T >::printFormatted(const char *txt)
{
	int32_t srcX = getW();
	int32_t srcY = getH();
	cout << txt << " nx=" << srcX << " ny=" << srcY << endl;
	for (int32_t j = 0; j < srcY; j++) {
#pragma novector
		for (int32_t i = 0; i < srcX; i++) {
			cout << setw(12) << setiosflags(ios::scientific) << setprecision(4) << _arr[Mat2Index(i, j)] << " ";
		}
		cout << endl;
	}
	cout << endl << endl;
}

template < typename T > long Matrix2 < T >::getLengthByte()
{
	int32_t srcX = getW();
	int32_t srcY = getH();
	return srcX * srcY * sizeof(T);
}

template < typename T > void Matrix2 < T >::read(const int f)
{
	int32_t srcX = getW();
	int32_t srcY = getH();
#pragma novector
	for (int j = 0; j < srcY; j++) {
		int l =::read(f, &_arr[Mat2Index(0, j)], srcX * sizeof(T));
	}
}

template < typename T > void Matrix2 < T >::write(const int f)
{
	int32_t srcX = getW();
	int32_t srcY = getH();
#pragma novector
	for (int j = 0; j < srcY; j++) {
		int l =::write(f, &_arr[Mat2Index(0, j)], srcX * sizeof(T));
	}
}

//
// Class instanciation: we force the compiler to generate the proper externals.
//

template class Matrix2 < double >;
template class Matrix2 < float >;
// template class Matrix2 < int >;
template class Matrix2 < int32_t >;

//EOF
