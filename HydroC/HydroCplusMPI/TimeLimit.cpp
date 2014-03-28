//
// (C) Guillaume.Colin-de-Verdiere at CEA.Fr
//
#ifdef MPI_ON
#include <mpi.h>
#endif
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
#include "cclock.h"
#include "TimeLimit.hpp"

using namespace std;

// template <typename T> 
TimeLimit::TimeLimit(void) { 
	char *p = getenv("BRIDGE_MPRUN_MAXTIME") ;
	m_allotedTime = 30 * 60; // 30mn by default
	m_orgTime = dcclock();
	m_curTime = 0;
	if (p != 0) {
		m_allotedTime = strtod(p, NULL);
	}
	// cerr << "Tremain " << m_allotedTime << endl;
}

// template <typename T> 
// TimeLimit::TimeLimit() { }

// template <typename T> 
TimeLimit::~TimeLimit() { 
}

double TimeLimit::timeRemain() {
	double remain;
	m_curTime = dcclock() - m_orgTime;
	remain = m_allotedTime - m_curTime;
	return remain;
}

double TimeLimit::timeRemainAll() {
	double remain;
#ifndef MPI_ON
	return timeRemain();
#else
	int mype = -1;
	int msiz = -1;

	MPI_Comm_rank(MPI_COMM_WORLD, &mype);
	MPI_Comm_size(MPI_COMM_WORLD, &msiz);
	
	if (mype == 0) {
		remain = timeRemain();
	}
	if (msiz > 1) MPI_Bcast(&remain, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	return remain;
#endif
}

// template <typename T> 
// TimeLimit::TimeLimit(const TimeLimit & obj) { }

// template <typename T> 
// TimeLimit & TimeLimit::operator=(const TimeLimit & rhs) { }

// template <typename T> 
// TimeLimit & TimeLimit::operator() (uint32_t i) { }

// template <typename T> 
// TimeLimit & TimeLimit::operator() (uint32_t i) const { }


// Instantion of the needed types for the linker
// template class TimeLimit<the_type>;

#ifdef WITH_MAIN
int main(int argc, char **argv) 
{
	TimeLimit tr;
#ifdef MPI_ON
	MPI_Init(&argc, &argv);
#endif
	cout << tr.timeRemainAll() << endl;
	sleep(10);
	cout << tr.timeRemainAll() << endl;
#ifdef MPI_ON
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();
#endif
	return 0;
}
#endif
//EOF
