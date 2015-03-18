//
// (C) Guillaume.Colin-de-Verdiere at CEA.Fr
//
#ifndef MATRIX_H
#define MATRIX_H
#include <cstring>
#include <cstdlib>
#include <cassert>
//#include <cstdint>
#include <stdint.h>		// for the definition of uint
#ifdef _OPENMP
#include <omp.h>
#endif
//
#include "Morton.hpp"

// #define Mat2Index(i,j) ((i) + (j) * _padw)

template < typename T > class Matrix2 {
 private:
	static const int32_t _align_value = 256;
	static const int32_t _align_flag = 1;
	int32_t _w, _h;
	int32_t _padw;
	T *_arr_alloc;
	T *_arr;		// aligned array

	int32_t cmpPad(int32_t x);
	void allocate(void);

	// // index in the array
	size_t Mat2Index(int32_t x, int32_t y) const {
		size_t r = x + y * _padw;
		 return r;
	};
	void aux_getFullCol(int32_t x, int32_t h, T * __restrict__ theCol, T * __restrict__ theArr);
	void aux_putFullCol(int32_t x, int32_t h, int32_t offY, T * __restrict__ theCol, T * __restrict__ theArr);

 public:
	// basic constructor
	Matrix2(void) {
		_w = _h = _padw = 0;
		_arr_alloc = 0;
	};

	// prefered constructor
	Matrix2(int32_t w, int32_t h);
	//  destructor
	~Matrix2();

	// copy operator
	Matrix2(const Matrix2 & m);

	// assignment operator
	Matrix2 & operator=(const Matrix2 & rhs);

	// access through ()
	// lhs 
	T & operator()(int32_t x, int32_t y) {
		return _arr[Mat2Index(x, y)];
	};

	// rhs
	T operator() (int32_t x, int32_t y) const {
		return _arr[Mat2Index(x, y)];
	};

	// accessors
	T *getRow(int32_t y) {
		return &_arr[Mat2Index(0, y)];
	};
	void getFullCol(int32_t x, T * theCol);
	void putFullCol(int32_t x, int32_t offY, T * theCol, int32_t l);

	int32_t getW(void) const {	// width
		return _w;
	};
	int32_t getH(void)const {	// heigth
		return _h;
	};
	int32_t nbElem(void)const {	// heigth
		return _w * _h;
	};
	void clear(void) {
		memset(_arr, 0, _padw * _h * sizeof(T));
	};
	void fill(T v);
	void swapDimOnly();
	void swapDimAndValues();
	int32_t maxMorton(void) {
		return morton2(getW(), getH());
	};
	bool idxFromMorton(int32_t & x, int32_t & y, int32_t m) {
		umorton2(&x, &y, m);
		if ((x < _w) && (y < _h))
			return true;
		return false;
	};
	int32_t *listMortonIdx(void);
	void Copy(const Matrix2 & src);
	void InsertMatrix(const Matrix2 & src, int32_t x0, int32_t y0);
	void printFormatted(const char *txt);
	long getLengthByte();
	void read(const int f);
	void write(const int f);
};

template < typename T > class Matrix3 {
 private:
	static const int32_t _align_value = 256;
	static const int32_t _align_flag = 1;
	int32_t _w, _h, _d;
	int32_t _padw;
	T *_arr_alloc;
	T *_arr;		// aligned array

	void allocate(void);

	// index in the array
	size_t index(int32_t x, int32_t y, int32_t z) const {
		size_t r = (z * _h + y) * _padw + x;
		 return r;
	};

 public:
	// basic constructor
	Matrix3(void) {
		_w = _h = _d = _padw = 0;
		_arr_alloc = 0;
	};

	// prefered constructor
	Matrix3(int32_t w, int32_t h = 1, int32_t d = 1);
	//  destructor
	~Matrix3();

	// copy operator
	Matrix3(const Matrix3 & m);

	// assignment operator
	Matrix3 & operator=(const Matrix3 & rhs);

	// access through ()
	// lhs 
	T & operator()(int32_t x, int32_t y, int32_t z) {
		return _arr[index(x, y, z)];
	};

	// rhs
	T operator() (int32_t x, int32_t y, int32_t z) const {
		return _arr[index(x, y, z)];
	};

	// accessors
	int32_t getW(void)const {	// width
		return _w;
	};
	int32_t getH(void)const {	// heigth
		return _h;
	};
	int32_t getD(void)const {	// depth
		return _d;
	};
	void clear(void) {
		memset(_arr, 0, _padw * _h * _d * sizeof(T));
	};

	void fill(T v);
};

#endif
//EOF
