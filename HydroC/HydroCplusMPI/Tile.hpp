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

#include "Options.hpp"
#include "Soa.hpp"
#include "EnumDefs.hpp"
#include "Utilities.hpp"
#include "ThreadBuffers.hpp"

// template <typename T>
class Tile {
 private:
	Tile * m_voisin[4];
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
	real_t *m_pstar;
	real_t *m_rl;
	real_t *m_ul;
	real_t *m_ur;
	real_t *m_pr;
	real_t *m_cl;
	real_t *m_cr;
	real_t *m_rr;
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
	real_t *m_recvbufru;	// receive right or up
	real_t *m_recvbufld;	// receive left or down
	real_t *m_sendbufru;	// send right or up
	real_t *m_sendbufld;	// send left or down

	// 

	// compute routines
	void slopeOnRow(int32_t xmin, int32_t xmax, real_t * __restrict__ qS, real_t * __restrict__ dqS);	// fait
	void slope();		// fait
	void traceonRow(int32_t xmin, int32_t xmax, real_t dtdx, real_t zeror, real_t zerol, real_t project, real_t * __restrict__ cS, real_t * __restrict__ qIDS, real_t * __restrict__ qIUS, real_t * __restrict__ qIVS, real_t * __restrict__ qIPS, real_t * __restrict__ dqIDS, real_t * __restrict__ dqIUS, real_t * __restrict__ dqIVS, real_t * __restrict__ dqIPS, real_t * __restrict__ pqxpIDS, real_t * __restrict__ pqxpIUS, real_t * __restrict__ pqxpIVS, real_t * __restrict__ pqxpIPS, real_t * __restrict__ pqxmIDS, real_t * __restrict__ pqxmIUS, real_t * __restrict__ pqxmIVS, real_t * __restrict__ pqxmIPS);	// fait
	void trace();		// fait
	void qleftrOnRow(int32_t xmin, int32_t xmax, real_t * __restrict__ pqleftS, real_t * __restrict__ pqrightS, real_t * __restrict__ pqxmS, real_t * __restrict__ pqxpS);	// fait
	void qleftr();		// fait
	void compflxOnRow(int32_t xmin, int32_t xmax, real_t entho, real_t * __restrict__ qgdnvIDS, real_t * __restrict__ qgdnvIUS, real_t * __restrict__ qgdnvIVS, real_t * __restrict__ qgdnvIPS, real_t * __restrict__ fluxIVS, real_t * __restrict__ fluxIUS, real_t * __restrict__ fluxIPS, real_t * __restrict__ fluxIDS);	// fait
	void compflx();		// fait
	void eosOnRow(int32_t xmin, int32_t xmax, real_t smallp, real_t * __restrict__ qIDS, real_t * __restrict__ eS, real_t * __restrict__ qIPS, real_t * __restrict__ cS);	// fait
	void eos(tileSpan_t span);	// fait
	void constprimOnRow(int32_t xmin, int32_t xmax,
			    real_t * __restrict__ qIDS,
			    real_t * __restrict__ qIPS,
			    real_t * __restrict__ qIVS,
			    real_t * __restrict__ qIUS,
			    real_t * __restrict__ uIDS, real_t * __restrict__ uIPS, real_t * __restrict__ uIVS, real_t * __restrict__ uIUS, real_t * __restrict__ eS);
	void constprim();	// fait
	void riemannOnRow(int32_t xmin, int32_t xmax, real_t smallp, real_t gamma6, real_t smallpp, real_t * __restrict__ qgdnvIDS, real_t * __restrict__ qgdnvIUS, real_t * __restrict__ qgdnvIPS, real_t * __restrict__ qgdnvIVS, real_t * __restrict__ qleftIDS, real_t * __restrict__ qleftIUS, real_t * __restrict__ qleftIPS, real_t * __restrict__ qleftIVS, real_t * __restrict__ qrightIDS, real_t * __restrict__ qrightIUS, real_t * __restrict__ qrightIPS, real_t * __restrict__ qrightIVS, long *__restrict__ goon, real_t * __restrict__ sgnm, real_t * __restrict__ pstar, real_t * __restrict__ rl, real_t * __restrict__ ul, real_t * __restrict__ pl, real_t * __restrict__ rr, real_t * __restrict__ ur, real_t * __restrict__ pr, real_t * __restrict__ cl, real_t * __restrict__ cr);	// fait
	void
	 riemannOnRowInRegs(int32_t xmin, int32_t xmax, real_t smallp, real_t gamma6,
			    real_t smallpp, real_t * __restrict__ qgdnvIDS,
			    real_t * __restrict__ qgdnvIUS,
			    real_t * __restrict__ qgdnvIPS,
			    real_t * __restrict__ qgdnvIVS,
			    real_t * __restrict__ qleftIDS,
			    real_t * __restrict__ qleftIUS,
			    real_t * __restrict__ qleftIPS,
			    real_t * __restrict__ qleftIVS,
			    real_t * __restrict__ qrightIDS,
			    real_t * __restrict__ qrightIUS, real_t * __restrict__ qrightIPS, real_t * __restrict__ qrightIVS, real_t * __restrict__ sgnm);
	void riemann();		// fait
	void compute_dt_loop2OnRow(real_t & tmp1, real_t & tmp2, int32_t xmin, int32_t xmax, real_t * __restrict__ cS, real_t * __restrict__ qIUS, real_t * __restrict__ qIVS);
	void compute_dt_loop1OnRow(int32_t xmin, int32_t xmax,
				   real_t * __restrict__ qIDS,
				   real_t * __restrict__ qIPS,
				   real_t * __restrict__ qIUS,
				   real_t * __restrict__ qIVS,
				   real_t * __restrict__ uoldIDS,
				   real_t * __restrict__ uoldIUS, real_t * __restrict__ uoldIVS, real_t * __restrict__ uoldIPS, real_t * __restrict__ eS);
	real_t compute_dt();
	void gatherconservXscan(int32_t xmin, int32_t xmax,
				real_t * __restrict__ uIDS,
				real_t * __restrict__ uIUS,
				real_t * __restrict__ uIVS,
				real_t * __restrict__ uIPS,
				real_t * __restrict__ uoldIDS, real_t * __restrict__ uoldIUS, real_t * __restrict__ uoldIVS, real_t * __restrict__ uoldIPS);
	void gatherconservYscan();
	void updateconservXscan(int32_t xmin, int32_t xmax, real_t dtdx,
				real_t * __restrict__ uIDS,
				real_t * __restrict__ uIUS,
				real_t * __restrict__ uIVS,
				real_t * __restrict__ uIPS,
				real_t * __restrict__ uoldIDS,
				real_t * __restrict__ uoldIUS,
				real_t * __restrict__ uoldIVS,
				real_t * __restrict__ uoldIPS,
				real_t * __restrict__ fluxIDS, real_t * __restrict__ fluxIVS, real_t * __restrict__ fluxIUS, real_t * __restrict__ fluxIPS);
	void updateconservYscan(int32_t s, int32_t xmin, int32_t xmax,
				int32_t ymin, int32_t ymax, real_t dtdx,
				Matrix2 < real_t > &uoldID,
				Matrix2 < real_t > &uoldIP,
				Matrix2 < real_t > &uoldIV,
				Matrix2 < real_t > &uoldIU,
				real_t * __restrict__ fluxIVS,
				real_t * __restrict__ fluxIUS,
				real_t * __restrict__ fluxIPS,
				real_t * __restrict__ fluxIDS,
				real_t * __restrict__ uIDS, real_t * __restrict__ uIPS, real_t * __restrict__ uIVS, real_t * __restrict__ uIUS, real_t * __restrict__ pl);

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
	int32_t pack_arrayv(int32_t xoffset, real_t * buffer);
	int32_t unpack_arrayv(int32_t xoffset, real_t * buffer);
	int32_t pack_arrayh(int32_t yoffset, real_t * buffer);
	int32_t unpack_arrayh(int32_t yoffset, real_t * buffer);

	// 

 protected:
 public:
	// basic constructor
	Tile(void);		// default constructor
	// destructor
	~Tile();

	void setNeighbourTile(tileNeighbour_t type, Tile * tile);

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
	}
	void setExtend(int32_t nx, int32_t ny, int32_t gnx, int32_t gny, int32_t offx, int32_t offy, real_t dx);
	void setVoisins(Tile * left, Tile * right, Tile * up, Tile * down);
	void setBuffers(ThreadBuffers * buf);
};
#endif
//EOF
