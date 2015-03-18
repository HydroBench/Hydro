//
// (C) Guillaume.Colin-de-Verdiere at CEA.Fr
//
#ifndef THREADBUFFERS_H
#define THREADBUFFERS_H
//
#include <iostream>
#include <cstring>
#include <cstdlib>
#include <cassert>
#include <cerrno>
#include <stdint.h>		// for the definition of int32_t

#include "Soa.hpp"
#include "EnumDefs.hpp"
#include "Utilities.hpp"

class ThreadBuffers {
 private:
	Soa * m_q, *m_qxm, *m_qxp, *m_dq;	// NXT, NYT
	Soa *m_qleft, *m_qright, *m_qgdnv;	// NX + 1, NY + 1
	 Matrix2 < real_t > *m_c, *m_e;	// NXT, NYT

	// work arrays for a single row/column
	real_t *m_sgnm;		//
	//
	real_t *m_pstar;
	real_t *m_rl;
	real_t *m_ul;
	real_t *m_pl;
	real_t *m_ur;
	real_t *m_pr;
	real_t *m_cl;
	real_t *m_cr;
	real_t *m_rr;
	long *m_goon;		// 
 protected:
 public:
	// basic constructor
	 ThreadBuffers(int32_t xmin, int32_t xmax, int32_t ymin, int32_t ymax);
	// destructor
	~ThreadBuffers();
	void swapStorageDims();

	Soa *getQ() {
		return m_q;
	};
	Soa *getQXM() {
		return m_qxm;
	};
	Soa *getQXP() {
		return m_qxp;
	};
	Soa *getDQ() {
		return m_dq;
	};
	Soa *getQLEFT() {
		return m_qleft;
	};
	Soa *getQRIGHT() {
		return m_qright;
	};
	Soa *getQGDNV() {
		return m_qgdnv;
	};
	Matrix2 < real_t > *getC() {
		return m_c;
	};
	Matrix2 < real_t > *getE() {
		return m_e;
	};

	// work arrays for a single row/column
	real_t *getPSTAR() {
		return m_pstar;
	};
	real_t *getRL() {
		return m_rl;
	};
	real_t *getUL() {
		return m_ul;
	};
	real_t *getPL() {
		return m_pl;
	};
	real_t *getUR() {
		return m_ur;
	};
	real_t *getPR() {
		return m_pr;
	};
	real_t *getCL() {
		return m_cl;
	};
	real_t *getCR() {
		return m_cr;
	};
	real_t *getRR() {
		return m_rr;
	};
	long *getGOON() {
		return m_goon;
	};
	real_t *getSGNM() {
		return m_sgnm;
	};

};
#endif
//EOF
