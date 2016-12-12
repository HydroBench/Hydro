//
// (C) Guillaume.Colin-de-Verdiere at CEA.Fr
//
#ifndef DOMAIN_H
#define DOMAIN_H
//
#include <iostream>
#include <cstring>
#include <cstdlib>
#include <cassert>
#include <cerrno>
#include <stdint.h>		// for the definition of int32_t
#ifdef _OPENMP
#include <omp.h>
#endif
#if WITHPNG > 0
#include <png.h>
#endif

#include "EnumDefs.hpp"
#include "Soa.hpp"
#include "Tile.hpp"
#include "ThreadBuffers.hpp"
#include "TimeLimit.hpp"

// template <typename T>
class Domain {
 private:
	// variables to protect between runs
	int32_t m_globNx, m_globNy;	// global size of the simulation //=
	Soa *m_uold;		// on the full domain
	godunovDir_t m_scan;	//=
	int m_nvtk;
	int m_npng;
	real_t m_tcur, m_dt;	//=
	int32_t m_iter;		// current iteration
	int32_t m_nbRun;
	double m_elapsTotal;

	// those variables are not protected to allow for modifications
	TimeLimit m_tr;
	int m_prt;
	Tile **m_tiles;		//=
	int32_t m_nbtiles;	//=
	int32_t m_tileSize;	//=
	int32_t m_numThreads;	// nb of threads available
	int32_t m_withMorton;
	int32_t *m_mortonIdx;
	 Matrix2 < int32_t > *m_morton;
	ThreadBuffers **m_buffers;	// shared buffers for all threads

	long m_maxrss, m_ixrss;	// memory usage;

	int32_t m_nx, m_ny;	// size of the local domain
	int m_box[MAXBOX_D];	// size of our current domain in global coordinate
	real_t m_dx;		//=
	int32_t m_boundary_left, m_boundary_right, m_boundary_down, m_boundary_up;	//=
	int32_t m_ExtraLayer;
	real_t m_gamma, m_smallc, m_smallr, m_slope_type;	//=
	int32_t m_testcase;	//=
	int32_t m_iorder;	//=
	real_t m_cfl;		//=
	int32_t m_nIterRiemann;	//=

	int32_t m_nStepMax;	//=
	int32_t m_nOutput;	//=
	int32_t m_checkPoint;
	int32_t m_forceStop;    //=

	real_t *m_localDt;	//=
	real_t m_tend;		//=
	real_t m_dtOutput;	//=
	real_t m_nextOutput;
	real_t m_dtImage;	//=
	int32_t m_nImage;       //=
	real_t m_nextImage;
	godunovScheme_t m_scheme;	//=

	// 
	int32_t m_myPe, m_nProc;	//=

	// working arrays
	real_t *m_recvbufru;	// receive right or up  //=
	real_t *m_recvbufld;	// receive left or down //=
	real_t *m_sendbufru;	// send right or up     //=
	real_t *m_sendbufld;	// send left or down    //=

	// misc.
	char *m_inputFile;

	// PNG output
	int32_t m_withPng;
	FILE *m_fp;
#if WITHPNG > 0
	png_structp m_png_ptr;
	png_infop m_info_ptr;
	png_bytep *m_row_pointers;
	png_byte *m_buffer;
#else
	uint8_t *m_buffer;
#endif
	int32_t m_shrink;
	int32_t m_shrinkSize;
	double m_timeGuard;
	int32_t m_numa;		// try to cope with numa effects
	int32_t m_forceSync;
	// timing of functions
	double **m_timerLoops;

	// member functions

	void vtkfile(int step);
	void vtkOutput(int step);
	void parseParams(int argc, char **argv);
	void readInput();
	void domainDecompose();
	void setTiles();
	void printSummary();
	void initMPI();
	void endMPI();
	void boundary_init();
	void boundary_process();
	real_t computeTimeStep();
	void computeDt();
	void createTestCase();
	void changeDirection();
	real_t reduceMin(real_t val);
	real_t reduceMaxAndBcast(real_t val);

	void getExtends(tileSpan_t span, int32_t & xmin, int32_t & xmax, int32_t & ymin, int32_t & ymax) {
		// returns the dimension of the tile with ghost cells.
		if (span == TILE_INTERIOR) {
			xmin = m_ExtraLayer;
			xmax = m_ExtraLayer + m_nx;
			ymin = m_ExtraLayer;
			ymax = m_ExtraLayer + m_ny;
		} else {	// TILE_FULL
			xmin = 0;
			xmax = 2 * m_ExtraLayer + m_nx;
			ymin = 0;
			ymax = 2 * m_ExtraLayer + m_ny;
		}
	};
	// pack/unpack array for ghost cells exchange. Works for MPI
	int32_t pack_arrayv(int32_t xoffset, real_t * buffer);
	int32_t unpack_arrayv(int32_t xoffset, real_t * buffer);
	int32_t pack_arrayh(int32_t yoffset, real_t * buffer);
	int32_t unpack_arrayh(int32_t yoffset, real_t * buffer);

	int32_t nbTile(int32_t tileSize) {
		int nbtC = 0;
		int nbtx = (m_nx + tileSize - 1) / tileSize;
		int nbty = (m_ny + tileSize - 1) / tileSize;
		nbtC = nbtx * nbty;
		return nbtC;
	}

	Domain(void) {
	};			// default constructor.
	void swapScan() {
		if (m_scan == X_SCAN)
			m_scan = Y_SCAN;
		else
			m_scan = X_SCAN;
	}
	int32_t tileFromMorton(int32_t t);
	void pngWriteFile(char *name);
	void pngProcess(void);
	void pngCloseFile(void);
	void getMaxVarValues(real_t * mxP, real_t * mxD, real_t * mxUV);
	void pngFillGap(int curx, int cury, int nx, int ny, int Iptr[4], int &cpt, int32_t imgSizeX, int32_t imgSizeY);

	// Checkpoint helper routines
	long protectScalars(const protectionMode_t mode, const int f);
	long protectArrays(const protectionMode_t mode, const int f);
	void writeProtectionVars(const int f);
	void readProtectionVars(const int f);
	void writeProtectionHeader(const int f);
	void readProtectionHeader(const int f);

	bool hasProtection();

	void writeProtection();
	void readProtection();

	bool StopComputation();

 protected:
 public:
	// basic constructor
	Domain(int argc, char **argv);
	// destructor
	~Domain();
	bool isStopped();
	int getMype() {
		return m_myPe;
	};
	int getNbpe() {
		return m_nProc;
	};
	void compute();
	int32_t myThread() {
		int32_t r = 0;
#ifdef _OPENMP
		r = omp_get_thread_num();
#endif
		return r;
	}
};
#endif
//EOF
