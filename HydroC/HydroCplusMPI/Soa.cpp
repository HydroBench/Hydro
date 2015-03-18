//
// (C) Guillaume.Colin-de-Verdiere at CEA.Fr
//
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <cstdio>
#include <climits>
#include <cerrno>

#include <strings.h>
#include <unistd.h>
#include <malloc.h>
#include <sys/time.h>
#include <float.h>

//
#include "Soa.hpp"

// template <typename T> 
// Soa::Soa(void) { }

// template <typename T> 
 Soa::Soa(int32_t nb, int32_t w, int32_t h):m_nbElem(nb)
{
	m_tab = new Matrix2 < double >*[m_nbElem];
#pragma novector
	for (int i = 0; i < m_nbElem; ++i) {
		m_tab[i] = new Matrix2 < double >(w, h);
	}
}

// template <typename T> 
Soa::~Soa()
{
#pragma novector
	for (int i = 0; i < m_nbElem; ++i) {
		delete m_tab[i];
	}
	delete[]m_tab;
}

long Soa::getLengthByte()
{
	long lgr = 0;
#pragma novector
	for (int i = 0; i < m_nbElem; ++i) {
		lgr += m_tab[i]->getLengthByte();
	}
	return lgr;
}

void Soa::read(const int f)
{
#pragma novector
	for (int i = 0; i < m_nbElem; ++i) {
		m_tab[i]->read(f);
	}
}

void Soa::write(const int f)
{
#pragma novector
	for (int i = 0; i < m_nbElem; ++i) {
		m_tab[i]->write(f);
	}
}

// template <typename T> 
// Soa::Soa(const Soa & obj) { }

// template <typename T> 
// Soa & Soa::operator=(const Soa & rhs) { }

// template <typename T> 
// Soa & Soa::operator() (int32_t i) { }

// template <typename T> 
// Soa & Soa::operator() (int32_t i) const { }

// Instantion of the needed types for the linker
// template class Soa<the_type>;
//EOF
