//
// (C) Guillaume.Colin-de-Verdiere at CEA.Fr
//
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <float.h>
#include <malloc.h>
#include <strings.h>
#include <sys/time.h>
#include <unistd.h>

//
#include "Options.hpp"
#include "ThreadBuffers.hpp"

using namespace std;

inline void *memsetth(void *s, int c, size_t n)
{
    char *ptr = (char *)s;
    char cval = c;
#pragma omp simd
    for (int32_t i = 0; i < n; i++) {
	ptr[i] = cval;
    }
    return s;
}

ThreadBuffers::ThreadBuffers(int32_t xmin, int32_t xmax, int32_t ymin,
			     int32_t ymax)
{
    int32_t lgx, lgy, lgmax;
    lgx = (xmax - xmin);
    lgy = (ymax - ymin);
    lgmax = lgx;
    if (lgmax < lgy)
	lgmax = lgy;

    m_q = new Soa(NB_VAR, lgx, lgy);
    m_qxm = new Soa(NB_VAR, lgx, lgy);
    m_qxp = new Soa(NB_VAR, lgx, lgy);
    m_dq = new Soa(NB_VAR, lgx, lgy);
    m_qleft = new Soa(NB_VAR, lgx, lgy);
    m_qright = new Soa(NB_VAR, lgx, lgy);
    m_qgdnv = new Soa(NB_VAR, lgx, lgy);

    m_c = new Matrix2 < real_t > (lgx, lgy);
    m_e = new Matrix2 < real_t > (lgx, lgy);

#if RIEMANNINREGS == 0
    m_pstar = AlignedAllocReal(lgmax);
    m_rl = AlignedAllocReal(lgmax);
    m_ul = AlignedAllocReal(lgmax);
    m_ur = AlignedAllocReal(lgmax);
    m_pr = AlignedAllocReal(lgmax);
    m_cl = AlignedAllocReal(lgmax);
    m_cr = AlignedAllocReal(lgmax);
    m_rr = AlignedAllocReal(lgmax);
    m_goon = AlignedAllocLong(lgmax);
#endif
    m_pl = AlignedAllocReal(lgmax);
    m_sgnm = AlignedAllocReal(lgmax);

    // remplit la mémoire pour forcer l'allocation
    memsetth(m_sgnm, 0, lgmax * sizeof(real_t));
    memsetth(m_pl, 0, lgmax * sizeof(real_t));
#if RIEMANNINREGS == 0
    memsetth(m_pstar, 0, lgmax * sizeof(real_t));
    memsetth(m_rl, 0, lgmax * sizeof(real_t));
    memsetth(m_ul, 0, lgmax * sizeof(real_t));
    memsetth(m_ur, 0, lgmax * sizeof(real_t));
    memsetth(m_pr, 0, lgmax * sizeof(real_t));
    memsetth(m_cl, 0, lgmax * sizeof(real_t));
    memsetth(m_cr, 0, lgmax * sizeof(real_t));
    memsetth(m_rr, 0, lgmax * sizeof(real_t));
    memsetth(m_goon, 0, lgmax * sizeof(long));
#endif
}

ThreadBuffers::~ThreadBuffers()
{
    delete m_q;
    delete m_qxm;
    delete m_qxp;
    delete m_dq;
    delete m_qleft;
    delete m_qright;
    delete m_qgdnv;
    delete m_c;
    delete m_e;

    free(m_sgnm);
    free(m_pl);

#if RIEMANNINREGS == 0
    free(m_pstar);
    free(m_rl);
    free(m_ul);
    free(m_ur);
    free(m_pr);
    free(m_cl);
    free(m_cr);
    free(m_rr);
    free(m_goon);
#endif
}

void ThreadBuffers::swapStorageDims()
{
#pragma novector
    for (int32_t i = 0; i < NB_VAR; i++) {
	Matrix2 < real_t > *m;
	m = (*m_q) (i);
	m->swapDimOnly();
	m = (*m_qxm) (i);
	m->swapDimOnly();
	m = (*m_qxp) (i);
	m->swapDimOnly();
	m = (*m_dq) (i);
	m->swapDimOnly();
	m = (*m_qleft) (i);
	m->swapDimOnly();
	m = (*m_qright) (i);
	m->swapDimOnly();
	m = (*m_qgdnv) (i);
	m->swapDimOnly();
    }
    m_c->swapDimOnly();
    m_e->swapDimOnly();
}

// EOF
