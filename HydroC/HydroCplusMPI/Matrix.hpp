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
	static const uint32_t _align_value = 256;
	static const uint32_t _align_flag = 1;
	uint32_t _w, _h;
	uint32_t _padw;
	T *_arr_alloc;
	T *_arr;		// aligned array

	uint32_t cmpPad(uint32_t x);
	void allocate(void);

	// // index in the array
	size_t Mat2Index(uint32_t x, uint32_t y) const { size_t r = x + y * _padw; return r; };
	void aux_getFullCol(uint32_t x, uint32_t h, T * __restrict__ theCol,
			    T * __restrict__ theArr);
	void aux_putFullCol(uint32_t x, uint32_t h, uint32_t offY,
			    T * __restrict__ theCol, T * __restrict__ theArr);

 public:
	// basic constructor
	Matrix2(void) {
		_w = _h = _padw = 0;
		_arr_alloc = 0;
	};

	// prefered constructor
	Matrix2(uint32_t w, uint32_t h);
	//  destructor
	~Matrix2();

	// copy operator
	Matrix2(const Matrix2 & m);

	// assignment operator
	Matrix2 & operator=(const Matrix2 & rhs);

	// access through ()
	// lhs 
	T & operator()(uint32_t x, uint32_t y) {
		return _arr[Mat2Index(x, y)];
	};

	// rhs
	T operator() (uint32_t x, uint32_t y) const {
		return _arr[Mat2Index(x, y)];
	};

	// accessors
	T *getRow(uint32_t y) {
		return &_arr[Mat2Index(0, y)];
	};
	void getFullCol(uint32_t x, T * theCol);
	void putFullCol(uint32_t x, uint32_t offY, T * theCol, uint32_t l);

	uint32_t getW(void) const {	// width
		return _w;
	};
	uint32_t getH(void)const {	// heigth
		return _h;
	};
	uint32_t nbElem(void)const {	// heigth
		return _w * _h;
	};
	void clear(void) {
		memset(_arr, 0, _padw * _h * sizeof(T));
	};
	void fill(T v);
	void swapDimOnly();
	void swapDimAndValues();
	uint32_t maxMorton(void) {
		return morton2(getW(), getH());
	};
	bool idxFromMorton(uint32_t & x, uint32_t & y, uint32_t m) {
		umorton2(&x, &y, m);
		if ((x < _w) && (y < _h))
			return true;
		return false;
	};
	uint32_t *listMortonIdx(void);
	void Copy(const Matrix2 & src);
	void InsertMatrix(const Matrix2 & src, uint32_t x0, uint32_t y0);
  void printFormatted(const char *txt);
  long getLengthByte();
  void read(const int f);
  void write(const int f);
};

template < typename T > class Matrix3 {
 private:
	static const uint32_t _align_value = 256;
	static const uint32_t _align_flag = 1;
	uint32_t _w, _h, _d;
	uint32_t _padw;
	T *_arr_alloc;
	T *_arr;		// aligned array

	void allocate(void);

	// index in the array
	size_t index(uint32_t x, uint32_t y, uint32_t z) const {
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
	Matrix3(uint32_t w, uint32_t h = 1, uint32_t d = 1);
	//  destructor
	~Matrix3();

	// copy operator
	Matrix3(const Matrix3 & m);

	// assignment operator
	Matrix3 & operator=(const Matrix3 & rhs);

	// access through ()
	// lhs 
	T & operator()(uint32_t x, uint32_t y, uint32_t z) {
		return _arr[index(x, y, z)];
	};

	// rhs
	T operator() (uint32_t x, uint32_t y, uint32_t z) const {
		return _arr[index(x, y, z)];
	};

	// accessors
	uint32_t getW(void)const {	// width
		return _w;
	};
	uint32_t getH(void)const {	// heigth
		return _h;
	};
	uint32_t getD(void)const {	// depth
		return _d;
	};
	void clear(void) {
		memset(_arr, 0, _padw * _h * _d * sizeof(T));
	};

	void fill(T v);
};

#endif
//EOF
