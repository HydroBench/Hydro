//
// (C) Guillaume.Colin-de-Verdiere at CEA.Fr
//
#ifndef TILE_H
#define TILE_H
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

#include "Options.hpp"
#include "Soa.hpp"
#include "EnumDefs.hpp"
#include "Utilities.hpp"
#include "ThreadBuffers.hpp"

// template <typename T>
class Tile {
 private:
	Tile * m_voisin[4];
	int32_t m_hasBeenProcessed;
#ifdef _OPENMP
	omp_lock_t m_lock;
#endif

	ThreadBuffers *m_myBuffers;	// the link to our ThreadBuffers
	int m_prt;

	// dimensions
	int32_t m_nx, m_ny;	// internal box size
	int32_t m_offx, m_offy;	// offset of the tile in the domain
	int32_t m_gnx, m_gny;	// total domain size
	int32_t m_ExtraLayer;
	// convenient variables
	godunovDir_t m_scan;	// direction of the scan

	// Godunov variables and arrays
	real_t m_tcur, m_dt, m_dx;
	Soa *m_uold;
	// work arrays private to a tile
	Soa *m_u;		// NXT, NYT
	Soa *m_flux;		// NX + 1, NY + 1

	// work arrays for a tile which can be shared across threads
	Soa *m_q, *m_qxm, *m_qxp, *m_dq;	// NXT, NYT
	Soa *m_qleft, *m_qright, *m_qgdnv;	// NX + 1, NY + 1
	 Matrix2 < real_t > *m_c, *m_e;	// NXT, NYT

	// work arrays for a single row/column
	real_t *m_sgnm;		//
	real_t *m_pl;
#if RIEMANNINREGS == 0
	Preal_t m_pstar;
	Preal_t m_rl;
	Preal_t m_ul;
	Preal_t m_ur;
	Preal_t m_pr;
	Preal_t m_cl;
	Preal_t m_cr;
	Preal_t m_rr;
	long *m_goon;		// 
#endif

	real_t m_gamma, m_smallc, m_smallr, m_cfl;

	int32_t m_niter_riemann;
	int32_t m_order;
	real_t m_slope_type;
	int32_t m_scheme;
	int32_t m_boundary_right, m_boundary_left, m_boundary_down, m_boundary_up;

	// mpi
	int32_t m_nproc, m_mype;
	// working arrays
	Preal_t m_recvbufru;	// receive right or up
	Preal_t m_recvbufld;	// receive left or down
	Preal_t m_sendbufru;	// send right or up
	Preal_t m_sendbufld;	// send left or down

	// 

	// compute routines
	void slopeOnRow(int32_t xmin, int32_t xmax, Preal_t qS, Preal_t dqS);	// fait
	void slope();		// fait
	void traceonRow(int32_t xmin, int32_t xmax, real_t dtdx, real_t zeror, real_t zerol, real_t project, Preal_t cS, Preal_t qIDS, Preal_t qIUS, Preal_t qIVS, Preal_t qIPS, Preal_t dqIDS, Preal_t dqIUS, Preal_t dqIVS, Preal_t dqIPS, Preal_t pqxpIDS, Preal_t pqxpIUS, Preal_t pqxpIVS, Preal_t pqxpIPS, Preal_t pqxmIDS, Preal_t pqxmIUS, Preal_t pqxmIVS, Preal_t pqxmIPS);	// fait
	void trace();		// fait
	void qleftrOnRow(int32_t xmin, int32_t xmax, Preal_t pqleftS, Preal_t pqrightS, Preal_t pqxmS, Preal_t pqxpS);	// fait
	void qleftr();		// fait
	void compflxOnRow(int32_t xmin, int32_t xmax, real_t entho, Preal_t qgdnvIDS, Preal_t qgdnvIUS, Preal_t qgdnvIVS, Preal_t qgdnvIPS, Preal_t fluxIVS, Preal_t fluxIUS, Preal_t fluxIPS, Preal_t fluxIDS);	// fait
	void compflx();		// fait
	void eosOnRow(int32_t xmin, int32_t xmax, real_t smallp, Preal_t qIDS, Preal_t eS, Preal_t qIPS, Preal_t cS);	// fait
	void eos(tileSpan_t span);	// fait
	void constprimOnRow(int32_t xmin, int32_t xmax, Preal_t qIDS, Preal_t qIPS, Preal_t qIVS, Preal_t qIUS, Preal_t uIDS, Preal_t uIPS, Preal_t uIVS, Preal_t uIUS, Preal_t eS);
	void constprim();	// fait
	void riemannOnRow(int32_t xmin, int32_t xmax, real_t smallp, real_t gamma6, real_t smallpp, Preal_t qgdnvIDS, Preal_t qgdnvIUS, Preal_t qgdnvIPS, Preal_t qgdnvIVS, Preal_t qleftIDS, Preal_t qleftIUS, Preal_t qleftIPS, Preal_t qleftIVS, Preal_t qrightIDS, Preal_t qrightIUS, Preal_t qrightIPS, Preal_t qrightIVS, long *__restrict__ goon, Preal_t sgnm, Preal_t pstar, Preal_t rl, Preal_t ul, Preal_t pl, Preal_t rr, Preal_t ur, Preal_t pr, Preal_t cl, Preal_t cr);	// fait
	void
	 riemannOnRowInRegs(int32_t xmin, int32_t xmax, real_t smallp, real_t gamma6,
			    real_t smallpp, Preal_t qgdnvIDS,
			    Preal_t qgdnvIUS,
			    Preal_t qgdnvIPS,
			    Preal_t qgdnvIVS,
			    Preal_t qleftIDS,
			    Preal_t qleftIUS, Preal_t qleftIPS, Preal_t qleftIVS, Preal_t qrightIDS, Preal_t qrightIUS, Preal_t qrightIPS, Preal_t qrightIVS, Preal_t sgnm);
	void riemann();		// fait
	void compute_dt_loop2OnRow(real_t & tmp1, real_t & tmp2, int32_t xmin, int32_t xmax, Preal_t cS, Preal_t qIUS, Preal_t qIVS);
	void compute_dt_loop1OnRow(int32_t xmin, int32_t xmax,
				   Preal_t qIDS, Preal_t qIPS, Preal_t qIUS, Preal_t qIVS, Preal_t uoldIDS, Preal_t uoldIUS, Preal_t uoldIVS, Preal_t uoldIPS, Preal_t eS);
	real_t compute_dt();
	void gatherconservXscan(int32_t xmin, int32_t xmax,
				Preal_t uIDS, Preal_t uIUS, Preal_t uIVS, Preal_t uIPS, Preal_t uoldIDS, Preal_t uoldIUS, Preal_t uoldIVS, Preal_t uoldIPS);
	void gatherconservYscan();
	void updateconservXscan(int32_t xmin, int32_t xmax, real_t dtdx,
				Preal_t uIDS,
				Preal_t uIUS,
				Preal_t uIVS,
				Preal_t uIPS,
				Preal_t uoldIDS, Preal_t uoldIUS, Preal_t uoldIVS, Preal_t uoldIPS, Preal_t fluxIDS, Preal_t fluxIVS, Preal_t fluxIUS, Preal_t fluxIPS);
	void updateconservYscan(int32_t s, int32_t xmin, int32_t xmax,
				int32_t ymin, int32_t ymax, real_t dtdx,
				Matrix2 < real_t > &uoldID,
				Matrix2 < real_t > &uoldIP,
				Matrix2 < real_t > &uoldIV,
				Matrix2 < real_t > &uoldIU,
				Preal_t fluxIVS, Preal_t fluxIUS, Preal_t fluxIPS, Preal_t fluxIDS, Preal_t uIDS, Preal_t uIPS, Preal_t uIVS, Preal_t uIUS, Preal_t pl);

	// utilities on tile
	void getExtends(tileSpan_t span, int32_t & xmin, int32_t & xmax, int32_t & ymin, int32_t & ymax) {
		// returns the dimension of the tile with or without ghost cells.
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
		if (m_scan == Y_SCAN) {
			Swap(xmin, ymin);
			Swap(xmax, ymax);
		}
	};

	// pack/unpack array for ghost cells exchange. Works either for OpenMP
	int32_t pack_arrayv(int32_t xoffset, Preal_t buffer);
	int32_t unpack_arrayv(int32_t xoffset, Preal_t buffer);
	int32_t pack_arrayh(int32_t yoffset, Preal_t buffer);
	int32_t unpack_arrayh(int32_t yoffset, Preal_t buffer);

	// 

 protected:
 public:
	// basic constructor
	Tile(void);		// default constructor
	// destructor
	~Tile();

	void setNeighbourTile(tileNeighbour_t type, Tile * tile);
	Tile *getNeighbourTile(tileNeighbour_t type) {
		return m_voisin[type];
	}
	void initTile(Soa * uold);
	void initPhys(real_t gamma, real_t smallc, real_t smallr, real_t cfl, real_t slope_type, int32_t niter_riemann, int32_t order, int32_t scheme);
	void setMpi(int32_t nproc, int32_t mype);

	void boundary_init();	// fait
	void boundary_process();	// fait
	void swapStorageDims();
	void swapScan() {
		if (m_scan == X_SCAN)
			m_scan = Y_SCAN;
		else
			m_scan = X_SCAN;
	}
	void godunov();		// 
	void gatherconserv();	// fait
	void updateconserv();	// fait
	real_t computeDt();	// returns local time step

	// set/get
	void setScan(godunovDir_t s) {
		m_scan = s;
	};
	void setTcur(real_t t) {
		m_tcur = t;
	};
	void setDt(real_t t) {
		m_dt = t;
	};
	void setPrt(int prt) {
		m_prt = prt;
	};
	void setExtend(int32_t nx, int32_t ny, int32_t gnx, int32_t gny, int32_t offx, int32_t offy, real_t dx);
	void setVoisins(Tile * left, Tile * right, Tile * up, Tile * down);
	void setBuffers(ThreadBuffers * buf);
	void notProcessed() {
		m_hasBeenProcessed = 0;
	};
	void doneProcessed(int step) {
		m_hasBeenProcessed = step;
	};
	int32_t isProcessed(int step) {
		return m_hasBeenProcessed == step;
	};
	void waitVoisin(Tile * voisin, int step);
};
#endif
//EOF
