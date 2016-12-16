//
// (C) Guillaume.Colin-de-Verdiere at CEA.Fr
//
#ifndef FAKEREAD_H
#define FAKEREAD_H
//
#include <iostream>
#include <cstring>
#include <cstdlib>
#include <cassert>
#include <cerrno>
#include <stdint.h>		// for the definition of uint32_t

// template <typename T>
class FakeRead {
 private:
	size_t m_fSize;
	int m_bSize;
	FILE *m_fd;
	double *m_buf;
	int m_rank;
 protected:
 public:
	// basic constructor
	// FakeRead(void); // default constructor. make it private if needed.
	 FakeRead(long fSize, int rank);
	// destructor
	~FakeRead();
	int Read(int64_t lg);
	// copy operator
	// FakeRead(const FakeRead & obj);
	// assignment operator
	// FakeRead & operator=(const FakeRead & rhs);
	// access through ()
	// FakeRead & operator() (uint32_t i) ; // lhs
	// FakeRead & operator() (uint32_t i) const ; // rhs
};
#endif
//EOF
