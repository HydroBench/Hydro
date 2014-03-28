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
#include <stdint.h>		// for the definition of uint32_t
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
	uint32_t m_globNx, m_globNy;	// global size of the simulation //=
	Soa *m_uold;		// on the full domain
	godunovDir_t m_scan;	//=
	int m_nvtk;
	int m_npng;
	real_t m_tcur, m_dt;	//=
	uint32_t m_iter; // current iteration
	uint32_t m_nbRun;
	double m_elapsTotal;

  // those variables are not protected to allow for modifications
	TimeLimit m_tr;
	int m_prt;
	Tile ** m_tiles;	//=
	uint32_t m_nbtiles;	//=
	uint32_t m_tileSize;	//=
	uint32_t m_numThreads;	// nb of threads available
	uint32_t m_withMorton;
	uint32_t *m_mortonIdx;
	Matrix2 <uint32_t> *m_morton;
	ThreadBuffers **m_buffers;	// shared buffers for all threads

	long m_maxrss, m_ixrss;	// memory usage;
	
  
	uint32_t m_nx, m_ny;	// size of the local domain
	int m_box[MAXBOX_D];	// size of our current domain in global coordinate
	real_t m_dx;		//=
	uint32_t m_boundary_left, m_boundary_right, m_boundary_down, m_boundary_up;	//=
	uint32_t m_ExtraLayer;
	real_t m_gamma, m_smallc, m_smallr, m_slope_type;	//=
	uint32_t m_testcase;	//=
	uint32_t m_iorder;	//=
	real_t m_cfl;		//=
	uint32_t m_nIterRiemann;	//=

	uint32_t m_nStepMax;	//=
	uint32_t m_nOutput;	//=
	uint32_t m_checkPoint;

	real_t *m_localDt;	//=
	real_t m_tend;		//=
	real_t m_dtOutput;	//=
	real_t m_nextOutput;
	real_t m_dtImage;	//=
	real_t m_nextImage;
	godunovScheme_t m_scheme;	//=

	// 
	uint32_t m_myPe, m_nProc;	//=

	// working arrays
	real_t *m_recvbufru;	// receive right or up  //=
	real_t *m_recvbufld;	// receive left or down //=
	real_t *m_sendbufru;	// send right or up     //=
	real_t *m_sendbufld;	// send left or down    //=

	// misc.
	char *m_inputFile;

	// PNG output
	uint32_t m_withPng;
	FILE *m_fp;
#if WITHPNG > 0
	png_structp m_png_ptr;
	png_infop m_info_ptr;
	png_bytep* m_row_pointers;
	png_byte* m_buffer;
#else
	uint8_t * m_buffer;
#endif
	uint32_t m_shrink;
	uint32_t m_shrinkSize;
	double m_timeGuard;

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

	void getExtends(tileSpan_t span, uint32_t & xmin, uint32_t & xmax,
			uint32_t & ymin, uint32_t & ymax) {
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
	uint32_t pack_arrayv(uint32_t xoffset, real_t * buffer);
	uint32_t unpack_arrayv(uint32_t xoffset, real_t * buffer);
	uint32_t pack_arrayh(uint32_t yoffset, real_t * buffer);
	uint32_t unpack_arrayh(uint32_t yoffset, real_t * buffer);

	Domain(void) {
	};			// default constructor.
	void swapScan() {
		if (m_scan == X_SCAN)
			m_scan = Y_SCAN;
		else
			m_scan = X_SCAN;
	}
	uint32_t tileFromMorton(uint32_t t);
	void pngWriteFile(char *name);
	void pngProcess(void);
	void pngCloseFile(void);
	void getMaxVarValues(real_t *mxP, real_t *mxD, real_t *mxUV);
	void pngFillGap(int curx, int cury, int nx, int ny, int Iptr[4], int &cpt, uint32_t imgSizeX, uint32_t imgSizeY);
  
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
	int getMype() { return m_myPe; };
	void compute();
	uint32_t myThread() {
		uint32_t r = 0;
#ifdef _OPENMP
		r = omp_get_thread_num();
#endif
		return r;
	}
};
#endif
//EOF
