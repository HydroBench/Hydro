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

template < typename T > int32_t Matrix2 < T >::cmpPad(int32_t x)
{
	int32_t pad;
	size_t lgrow = 0;
	lgrow = x * sizeof(T);
	lgrow += (_align_value - lgrow % _align_value) % _align_value;
	pad = lgrow / sizeof(T);
	return pad;
}

static const int nbloc = 1024;
static const int nshift = 10;
static const int ninc = 128;

template < typename T > void Matrix2 < T >::allocate(void)
{
	int32_t lgrow = 0;
	int32_t maxPad = 0, padh = 0;
	static int decal = 0;

	// make sure that rows are aligned thru padding 
	_padw = cmpPad(_w);
	padh = cmpPad(_h);

	if ((_padw * padh) == 0) {
		cerr << _w << endl;
		cerr << _h << endl;
		cerr << _padw << endl;
		cerr << padh << endl;
#ifdef MPI_ON
		MPI_Abort(MPI_COMM_WORLD, 1);
#else
		abort();
#endif
	}

#if WITHHBW==0 && WITHPOSIX == 0 && WITHNEW == 0
#define WITHPOSIX 1
#endif
	size_t lgrTab = (_padw * padh + _align_value) * sizeof(T);
#ifdef WITHNEW
	_arr_alloc = new T[_padw * padh + _align_value];
#pragma message "C++ NEW usage activated"
#endif
#ifdef WITHHBW
	int rc = hbw_posix_memalign((void **) &_arr_alloc, nbloc, lgrTab + nbloc);
#pragma message "HBW memory usage activated"
#endif
#ifdef WITHPOSIX
	// _arr_alloc = (T *) malloc(lgrTab);
#pragma message "posix_memalign activated"
	int rc = posix_memalign((void **) &_arr_alloc, nbloc, lgrTab + nbloc);
#endif
#if defined(WITHPOSIX) || defined(WITHHBW)
	char *tmp = (char *) _arr_alloc;
	tmp += decal;
	decal += ninc;
	if (decal >= nbloc) decal = 0;
	_arr_alloc = (T*) tmp;
#endif
	memset(_arr_alloc, 0, lgrTab);
	assert(_arr_alloc != 0);
	_arr = _arr_alloc;

	// make sure that the working array is properly aligned
	size_t offset = (_align_value - ((size_t) (_arr_alloc)) % _align_value) % _align_value;
	_arr = reinterpret_cast < T * >(static_cast < char *>(static_cast < void *>(_arr_alloc))+offset * _align_flag);
}

template < typename T > void Matrix2 < T >::swapDimOnly()
{
	Swap(_w, _h);
	_padw = cmpPad(_w);
}

template < typename T > void Matrix2 < T >::swapDimAndValues()
{
	int32_t t = _w;
	_w = _h;
	_h = _w;
	_padw = cmpPad(_w);
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
	assert(_arr_alloc != 0);

#if defined(WITHPOSIX) || defined(WITHHBW)
	size_t tmp = (size_t) _arr_alloc;
	tmp = tmp >> nshift;
	tmp = tmp << nshift;
	_arr_alloc = (T*) tmp;
#endif
#ifdef WITHNEW
	delete[]_arr_alloc;
#endif
#ifdef WITHHBW
	hbw_free(_arr_alloc);
#else
	free(_arr_alloc);
#endif
	_arr_alloc = 0;
	_arr = NULL;
}

template < typename T > Matrix2 < T >::Matrix2(const Matrix2 < T > &m)
{
	// std::cerr << "copy op " << this << std::endl;
	_w = (m._w);
	_h = (m._h);
	allocate();
	assert(_arr != 0);
	memcpy(_arr, m._arr, _padw * _h * sizeof(T));
}

template < typename T > Matrix2 < T > &Matrix2 < T >::operator=(const Matrix2 < T > &rhs)
{
	// std::cerr << "= op " << this << std::endl;
	_w = (rhs._w);
	_h = (rhs._h);
	allocate();
	assert(_arr != 0);
	memcpy(_arr, rhs._arr, _padw * _h * sizeof(T));
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

// =================================================================

 template < typename T > Matrix3 < T >::Matrix3(int32_t w, int32_t h, int32_t d):
_w(w), _h(h), _d(d)
{
	allocate();
	// std::cerr << "create " << this << std::endl;
}

template < typename T > void Matrix3 < T >::fill(T v)
{
	int32_t i, j, k;
#ifdef _OPENMP
	int embedded = 0;	// to make it openmp proof 
#endif

#ifdef _OPENMP
	embedded = omp_in_parallel();
#endif
	T *tmp = _arr;
#ifdef _OPENMP
#pragma omp parallel for shared(tmp) private(i,j,k) collapse(2) if (!embedded) SCHEDULE
#endif
	for (k = 0; k < _d; k++) {
		for (j = 0; j < _h; j++) {
// #pragma simd
			for (i = 0; i < _w; i++) {
				tmp[index(i, j, k)] = v;
			}
		}
	}
}

template < typename T > void Matrix3 < T >::allocate(void)
{
	int32_t lgrow;

	// make sure that rows are aligned thru padding 
	lgrow = _w * sizeof(T);
	lgrow += (_align_value - lgrow % _align_value) % _align_value;
	_padw = lgrow / sizeof(T);

	assert((_padw * _h * _d) != 0);

	_arr_alloc = new T[_padw * _h * _d + _align_value];
	assert(_arr_alloc != 0);
	_arr = _arr_alloc;

	// make sure that the working array is properly aligned
	size_t offset = (_align_value - ((size_t) (_arr_alloc)) % _align_value) % _align_value;
	_arr = reinterpret_cast < T * >(static_cast < char *>(static_cast < void *>(_arr_alloc))+offset * _align_flag);
}

template < typename T > Matrix3 < T >::~Matrix3()
{
	// std::cerr << "Destruction object " << this << std::endl;
	assert(_arr_alloc != 0);
	delete[]_arr_alloc;
	_arr_alloc = 0;
	_arr = NULL;
}

template < typename T > Matrix3 < T >::Matrix3(const Matrix3 < T > &m)
{
	// std::cerr << "copy op " << this << std::endl;
	_w = (m._w);
	_h = (m._h);
	_d = (m._d);
	allocate();
	assert(_arr != 0);
	memcpy(_arr, m._arr, _padw * _h * _d * sizeof(T));
}

template < typename T > Matrix3 < T > &Matrix3 < T >::operator=(const Matrix3 < T > &rhs)
{
	// std::cerr << "= op " << this << std::endl;
	_w = (rhs._w);
	_h = (rhs._h);
	_d = (rhs._d);
	allocate();
	assert(_arr != 0);
	memcpy(_arr, rhs._arr, _padw * _h * _d * sizeof(T));
	return *this;
}

//
// Class instanciation: we force the compiler to generate the proper externals.
//

template class Matrix2 < double >;
template class Matrix2 < float >;
// template class Matrix2 < int >;
template class Matrix2 < int32_t >;

template class Matrix3 < int >;
template class Matrix3 < double >;
template class Matrix3 < float >;

//EOF
