//
// (C) Guillaume.Colin-de-Verdiere at CEA.Fr
//
#ifndef TIMERS_H
#define TIMERS_H
//
#include <cassert>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdint.h>		// for the definition of uint32_t

// template <typename T>
typedef enum {
    SLOPE = 0,
    TRACE,
    QLEFTR,
    COMPFLX,
    UPDCVAR,
    GATHCVAR,
    EOS,
    COMPDT,
    CONSTPRIM,
    RIEMANN,
    TILEOMP,			// maker to end tile measurements
    BOUNDEXEC,
    BOUNDINIT,
    REDUCEMIN,
    REDUCEMAX,
    BANDWIDTH,			// marker to separate performance of routine from bandwitdh
    ALLTILECMP,
    BOUNDINITBW,
    LASTENTRY
} Fname_t;

class Timers {
 private:
    double *m_elaps;
    double *m_vmin, *m_vmax, *m_vavg;

 protected:
 public:
    // basic constructor
     Timers(void);		// default constructor. make it private if needed.
    // Timers();
    // destructor
    ~Timers();
    void set(const Fname_t f, const double d) {
	m_elaps[f] = d;
    };
    void add(const Fname_t f, const double d) {
	m_elaps[f] += d;
    };
    void div(const Fname_t f, const double d) {
	m_elaps[f] /= d;
    };
    double get(const Fname_t f) const {
	return m_elaps[f];
    };

    void getStats();
    void print(void);
    void printStats(void);
    const char *name(Fname_t f);
    // copy operator
    // Timers(const Timers & obj);
    // assignment operator
    Timers & operator=(const Timers & rhs);
    // access through ()
    // Timers & operator() (uint32_t i) ; // lhs
    // Timers & operator() (uint32_t i) const ; // rhs
    Timers & operator+=(const Timers & rhs);
};
#endif
// EOF
