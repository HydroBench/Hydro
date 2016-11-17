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
#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;

//

#include "Options.hpp"
#define LAMBDAFUNC 1
#if LAMBDAFUNC == 1
#pragma message "Activation des lambdas function"
#endif

#if  USEINTRINSICS != 0
#include "arch.hpp"
#endif
#include "Tile.hpp"

// template <typename T> 
// Tile::Tile(void) { }

// template <typename T> 
Tile::Tile()
{
	for (int32_t i = 0; i < NEIGHBOUR_TILE; i++) {
		m_voisin[i] = 0;
	}
	m_ExtraLayer = 2;
	m_scan = X_SCAN;
	m_uold = 0;
#ifdef _OPENMP
	omp_init_lock(&m_lock);
#endif
}

// template <typename T> 
Tile::~Tile()
{
#ifdef _OPENMP
	omp_destroy_lock(&m_lock);
#endif
	delete m_u;
	delete m_flux;
}

void
 Tile::setNeighbourTile(tileNeighbour_t type, Tile * tile)
{
	m_voisin[type] = tile;
}

void Tile::initTile(Soa * uold)
{
	int32_t xmin, xmax, ymin, ymax;
	int32_t lgx, lgy, lgmax;

	getExtends(TILE_FULL, xmin, xmax, ymin, ymax);

	lgx = (xmax - xmin);
	lgy = (ymax - ymin);
	lgmax = lgx;
	if (lgmax < lgy)
		lgmax = lgy;

	//
	m_uold = uold;
	//
	m_u = new Soa(NB_VAR, lgx, lgy);
	m_flux = new Soa(NB_VAR, lgx, lgy);

}

void Tile::swapStorageDims()
{
#pragma novector
	for (int32_t i = 0; i < NB_VAR; i++) {
		Matrix2 < real_t > *m;
		m = (*m_u) (i);
		m->swapDimOnly();
		m = (*m_flux) (i);
		m->swapDimOnly();
	}
}

void Tile::setMpi(int32_t nproc, int32_t mype)
{
	m_nproc = nproc;
	m_mype = mype;
}

void Tile::initPhys(real_t gamma, real_t smallc, real_t smallr, real_t cfl, real_t slope_type, int32_t niter_riemann, int32_t order, int32_t scheme)
{
	m_gamma = gamma;
	m_smallc = smallc;
	m_smallr = smallr;

	m_cfl = cfl;
	m_slope_type = slope_type;

	m_niter_riemann = niter_riemann;
	m_order = order;
	m_scheme = scheme;
}

void Tile::setExtend(int32_t nx, int32_t ny, int32_t gnx, int32_t gny, int32_t offx, int32_t offy, real_t dx)
{
	m_nx = nx;
	m_ny = ny;
	m_gnx = gnx;
	m_gny = gny;
	m_offx = offx;
	m_offy = offy;
	m_dx = dx;
}

// Compute part
void Tile::slopeOnRow(int32_t xmin, int32_t xmax, Preal_t qS, Preal_t dqS)
{

	// #pragma vector aligned  // impossible !
#if TILEUSER == 0
#pragma loop_count min=TILEMIN, avg=TILESIZ
#endif
	for (int32_t i = xmin + 1; i < xmax - 1; i++) {
		real_t dlft, drgt, dcen, dsgn, slop, dlim;
		real_t llftrgt = 0;
		real_t t1;
		dlft = m_slope_type * (qS[i] - qS[i - 1]);
		drgt = m_slope_type * (qS[i + 1] - qS[i]);
		dcen = half * (dlft + drgt) / m_slope_type;
		dsgn = (dcen > 0) ? one : -one;	// sign(one, dcen);
#ifndef NOTDEF
		llftrgt = ((dlft * drgt) <= zero);
		t1 = Min(Fabs(dlft), Fabs(drgt));
		dqS[i] = dsgn * Min((one - llftrgt) * t1, Fabs(dcen));
#else
		slop = Min(Fabs(dlft), Fabs(drgt));
		dlim = slop;
		if ((dlft * drgt) <= zero) {
			dlim = zero;;
		}
		dqS[i] = dsgn * Min(dlim, Fabs(dcen));
#endif
	}
}

void Tile::slope()
{
	int32_t xmin, xmax, ymin, ymax;

	for (int32_t nbv = 0; nbv < NB_VAR; nbv++) {
		Matrix2 < real_t > &q = *(*m_q) (nbv);
		Matrix2 < real_t > &dq = *(*m_dq) (nbv);

		getExtends(TILE_FULL, xmin, xmax, ymin, ymax);

		for (int32_t s = ymin; s < ymax; s++) {
			Preal_t qS = q.getRow(s);
			Preal_t dqS = dq.getRow(s);
			slopeOnRow(xmin, xmax, qS, dqS);
		}
	}
	Matrix2 < real_t > &q = *(*m_q) (IP_VAR);
	if (m_prt)
		q.printFormatted("Tile q slope");
	Matrix2 < real_t > &dq = *(*m_dq) (IP_VAR);
	if (m_prt)
		dq.printFormatted("Tile dq slope");
}

void Tile::trace()
{
	int32_t xmin, xmax, ymin, ymax;
	Matrix2 < real_t > &qID = *(*m_q) (ID_VAR);
	Matrix2 < real_t > &qIV = *(*m_q) (IV_VAR);
	Matrix2 < real_t > &qIU = *(*m_q) (IU_VAR);
	Matrix2 < real_t > &qIP = *(*m_q) (IP_VAR);
	Matrix2 < real_t > &dqID = *(*m_dq) (ID_VAR);
	Matrix2 < real_t > &dqIV = *(*m_dq) (IV_VAR);
	Matrix2 < real_t > &dqIU = *(*m_dq) (IU_VAR);
	Matrix2 < real_t > &dqIP = *(*m_dq) (IP_VAR);
	Matrix2 < real_t > &pqxmID = *(*m_qxm) (ID_VAR);
	Matrix2 < real_t > &pqxmIP = *(*m_qxm) (IP_VAR);
	Matrix2 < real_t > &pqxmIV = *(*m_qxm) (IV_VAR);
	Matrix2 < real_t > &pqxmIU = *(*m_qxm) (IU_VAR);
	Matrix2 < real_t > &pqxpID = *(*m_qxp) (ID_VAR);
	Matrix2 < real_t > &pqxpIP = *(*m_qxp) (IP_VAR);
	Matrix2 < real_t > &pqxpIV = *(*m_qxp) (IV_VAR);
	Matrix2 < real_t > &pqxpIU = *(*m_qxp) (IU_VAR);

	real_t zerol = 0.0, zeror = 0.0, project = 0.;
	real_t dtdx = m_dt / m_dx;

	if (m_scheme == SCHEME_MUSCL) {	// MUSCL-Hancock method
		zerol = -hundred / dtdx;
		zeror = hundred / dtdx;
		project = one;
	}
	// if (strcmp(Hscheme, "plmde") == 0) {       // standard PLMDE
	if (m_scheme == SCHEME_PLMDE) {	// standard PLMDE
		zerol = zero;
		zeror = zero;
		project = one;
	}
	// if (strcmp(Hscheme, "collela") == 0) {     // Collela's method
	if (m_scheme == SCHEME_COLLELA) {	// Collela's method
		zerol = zero;
		zeror = zero;
		project = zero;
	}

	getExtends(TILE_FULL, xmin, xmax, ymin, ymax);

	for (int32_t s = ymin; s < ymax; s++) {
		Preal_t cS = (*m_c).getRow(s);
		Preal_t qIDS = qID.getRow(s);
		Preal_t qIUS = qIU.getRow(s);
		Preal_t qIVS = qIV.getRow(s);
		Preal_t qIPS = qIP.getRow(s);
		Preal_t dqIDS = dqID.getRow(s);
		Preal_t dqIUS = dqIU.getRow(s);
		Preal_t dqIVS = dqIV.getRow(s);
		Preal_t dqIPS = dqIP.getRow(s);
		Preal_t pqxpIDS = pqxpID.getRow(s);
		Preal_t pqxpIUS = pqxpIU.getRow(s);
		Preal_t pqxpIVS = pqxpIV.getRow(s);
		Preal_t pqxpIPS = pqxpIP.getRow(s);
		Preal_t pqxmIDS = pqxmID.getRow(s);
		Preal_t pqxmIUS = pqxmIU.getRow(s);
		Preal_t pqxmIVS = pqxmIV.getRow(s);
		Preal_t pqxmIPS = pqxmIP.getRow(s);
		traceonRow(xmin, xmax, dtdx, zeror, zerol, project,
			   cS, qIDS, qIUS, qIVS, qIPS, dqIDS, dqIUS, dqIVS, dqIPS, pqxpIDS, pqxpIUS, pqxpIVS, pqxpIPS, pqxmIDS, pqxmIUS, pqxmIVS, pqxmIPS);
	}
	if (m_prt)
		pqxmIP.printFormatted("Tile pqxmIP trace");
	if (m_prt)
		pqxpIP.printFormatted("Tile pqxpIP trace");
}

void Tile::traceonRow(int32_t xmin,
		      int32_t xmax,
		      real_t dtdx,
		      real_t zeror,
		      real_t zerol,
		      real_t project,
		      Preal_t cS,
		      Preal_t qIDS,
		      Preal_t qIUS,
		      Preal_t qIVS,
		      Preal_t qIPS,
		      Preal_t dqIDS,
		      Preal_t dqIUS,
		      Preal_t dqIVS,
		      Preal_t dqIPS, Preal_t pqxpIDS, Preal_t pqxpIUS, Preal_t pqxpIVS, Preal_t pqxpIPS, Preal_t pqxmIDS, Preal_t pqxmIUS, Preal_t pqxmIVS, Preal_t pqxmIPS)
{
#if ALIGNED > 0
#pragma vector aligned
#if TILEUSER == 0
#pragma loop_count min=TILEMIN, avg=TILEAVG
#endif
#endif
	for (int32_t i = xmin; i < xmax; i++) {
		real_t cc, csq, r, u, v, p;
		real_t dr, du, dv, dp;
		real_t alpham, alphap, alpha0r, alpha0v;
		real_t spminus, spplus, spzero;
		real_t apright, amright, azrright, azv1right;
		real_t apleft, amleft, azrleft, azv1left;
		real_t upcc, umcc, upccx, umccx, ux;
		real_t rOcc, OrOcc, dprcc;

		cc = cS[i];
		csq = Square(cc);
		r = qIDS[i];
		u = qIUS[i];
		v = qIVS[i];
		p = qIPS[i];
		dr = dqIDS[i];
		du = dqIUS[i];
		dv = dqIVS[i];
		dp = dqIPS[i];
		rOcc = r / cc;
		OrOcc = cc / r;
		dprcc = dp / (r * cc);
		alpham = half * (dprcc - du) * rOcc;
		alphap = half * (dprcc + du) * rOcc;
		alpha0r = dr - dp / csq;
		alpha0v = dv;
		upcc = u + cc;
		umcc = u - cc;
		upccx = upcc * dtdx;
		umccx = umcc * dtdx;
		ux = u * dtdx;

		// Right state
		spminus = (umcc >= zeror) ? (project) : umccx + one;
		spplus = (upcc >= zeror) ? (project) : upccx + one;
		spzero = (u >= zeror) ? (project) : ux + one;
		apright = -half * spplus * alphap;
		amright = -half * spminus * alpham;
		azrright = -half * spzero * alpha0r;
		azv1right = -half * spzero * alpha0v;
		pqxpIDS[i] = r + (apright + amright + azrright);
		pqxpIUS[i] = u + (apright - amright) * OrOcc;
		pqxpIVS[i] = v + (azv1right);
		pqxpIPS[i] = p + (apright + amright) * csq;

		// Left state
		spminus = (umcc <= zerol) ? (-project) : umccx - one;
		spplus = (upcc <= zerol) ? (-project) : upccx - one;
		spzero = (u <= zerol) ? (-project) : ux - one;
		apleft = -half * spplus * alphap;
		amleft = -half * spminus * alpham;
		azrleft = -half * spzero * alpha0r;
		azv1left = -half * spzero * alpha0v;
		pqxmIDS[i] = r + (apleft + amleft + azrleft);
		pqxmIUS[i] = u + (apleft - amleft) * OrOcc;
		pqxmIVS[i] = v + (azv1left);
		pqxmIPS[i] = p + (apleft + amleft) * csq;
	}
}

void Tile::qleftrOnRow(int32_t xmin, int32_t xmax, Preal_t pqleftS, Preal_t pqrightS, Preal_t pqxmS, Preal_t pqxpS)
{

	// #pragma vector aligned // impossible !
#if TILEUSER == 0
#pragma loop_count min=TILEMIN, avg=TILESIZ
#endif
	for (int32_t i = xmin; i < xmax; i++) {
		pqleftS[i] = pqxmS[i + 1];
		pqrightS[i] = pqxpS[i + 2];
	}
}

void Tile::qleftr()
{
	int32_t xmin, xmax, ymin, ymax;
	getExtends(TILE_FULL, xmin, xmax, ymin, ymax);

	for (int32_t v = 0; v < NB_VAR; v++) {
		Matrix2 < real_t > &pqleft = *(*m_qleft) (v);
		Matrix2 < real_t > &pqright = *(*m_qright) (v);
		Matrix2 < real_t > &pqxm = *(*m_qxm) (v);
		Matrix2 < real_t > &pqxp = *(*m_qxp) (v);
		for (int32_t s = ymin; s < ymax; s++) {
			Preal_t pqleftS = pqleft.getRow(s);
			Preal_t pqrightS = pqright.getRow(s);
			Preal_t pqxmS = pqxm.getRow(s);
			Preal_t pqxpS = pqxp.getRow(s);
			qleftrOnRow(xmin, xmax, pqleftS, pqrightS, pqxmS, pqxpS);
		}
	}
	Matrix2 < real_t > &pqleft = *(*m_qleft) (IP_VAR);
	if (m_prt)
		pqleft.printFormatted("Tile qleft qleftr");
	Matrix2 < real_t > &pqright = *(*m_qright) (IP_VAR);
	if (m_prt)
		pqright.printFormatted("Tile qright qleftr");
}

void Tile::compflxOnRow(int32_t xmin,
			int32_t xmax,
			real_t entho, Preal_t qgdnvIDS, Preal_t qgdnvIUS, Preal_t qgdnvIVS, Preal_t qgdnvIPS, Preal_t fluxIVS, Preal_t fluxIUS, Preal_t fluxIPS, Preal_t fluxIDS)
{
	for (int32_t i = xmin; i < xmax; i++) {
		// Mass density
		real_t massDensity = qgdnvIDS[i] * qgdnvIUS[i];
		fluxIDS[i] = massDensity;
		// Normal momentum
		fluxIUS[i] = massDensity * qgdnvIUS[i] + qgdnvIPS[i];
		// Transverse momentum 1
		fluxIVS[i] = massDensity * qgdnvIVS[i];
		// Total energy
		real_t ekin = half * qgdnvIDS[i] * (Square(qgdnvIUS[i]) + Square(qgdnvIVS[i]));
		real_t etot = qgdnvIPS[i] * entho + ekin;
		fluxIPS[i] = qgdnvIUS[i] * (etot + qgdnvIPS[i]);
	}
}

void Tile::compflx()
{
	int32_t xmin, xmax, ymin, ymax;
	Matrix2 < real_t > &qgdnvID = *(*m_qgdnv) (ID_VAR);
	Matrix2 < real_t > &qgdnvIU = *(*m_qgdnv) (IU_VAR);
	Matrix2 < real_t > &qgdnvIP = *(*m_qgdnv) (IP_VAR);
	Matrix2 < real_t > &qgdnvIV = *(*m_qgdnv) (IV_VAR);
	Matrix2 < real_t > &fluxIV = *(*m_flux) (IV_VAR);
	Matrix2 < real_t > &fluxIU = *(*m_flux) (IU_VAR);
	Matrix2 < real_t > &fluxIP = *(*m_flux) (IP_VAR);
	Matrix2 < real_t > &fluxID = *(*m_flux) (ID_VAR);

	real_t entho = 1.0 / (m_gamma - 1.0);

	getExtends(TILE_FULL, xmin, xmax, ymin, ymax);

	for (int32_t s = ymin; s < ymax; s++) {
		Preal_t qgdnvIDS = qgdnvID.getRow(s);
		Preal_t qgdnvIUS = qgdnvIU.getRow(s);
		Preal_t qgdnvIPS = qgdnvIP.getRow(s);
		Preal_t qgdnvIVS = qgdnvIV.getRow(s);
		Preal_t fluxIVS = fluxIV.getRow(s);
		Preal_t fluxIUS = fluxIU.getRow(s);
		Preal_t fluxIPS = fluxIP.getRow(s);
		Preal_t fluxIDS = fluxID.getRow(s);

		compflxOnRow(xmin, xmax, entho, qgdnvIDS, qgdnvIUS, qgdnvIVS, qgdnvIPS, fluxIVS, fluxIUS, fluxIPS, fluxIDS);
	}
	if (m_prt)
		fluxIP.printFormatted("Tile fluxIP compflx");
}

template < typename LOOP_BODY > void forall(int begin, int end, LOOP_BODY body)
{
#pragma omp simd
	for (int i = begin; i < end; ++i)
		body(i);
}

void Tile::updateconservXscan(int32_t xmin, int32_t xmax, real_t dtdx,
			      Preal_t uIDS,
			      Preal_t uIUS,
			      Preal_t uIVS,
			      Preal_t uIPS, Preal_t uoldIDS, Preal_t uoldIUS, Preal_t uoldIVS, Preal_t uoldIPS, Preal_t fluxIDS, Preal_t fluxIVS, Preal_t fluxIUS, Preal_t fluxIPS)
{
#if TILEUSER == 0
#pragma loop_count min=TILEMIN, avg=TILESIZ
#endif

#ifndef LAMBDAFUNC
#pragma omp simd
	for (int32_t i = xmin; i < xmax; i++) {
		uoldIDS[i + m_offx] = uIDS[i] + (fluxIDS[i - 2] - fluxIDS[i - 1]) * dtdx;
		uoldIVS[i + m_offx] = uIVS[i] + (fluxIVS[i - 2] - fluxIVS[i - 1]) * dtdx;
		uoldIUS[i + m_offx] = uIUS[i] + (fluxIUS[i - 2] - fluxIUS[i - 1]) * dtdx;
		uoldIPS[i + m_offx] = uIPS[i] + (fluxIPS[i - 2] - fluxIPS[i - 1]) * dtdx;
	}
#else
	forall(xmin, xmax,[&, dtdx] (int i) {
	       int im = i + m_offx, i2 = i - 2, i1 = i - 1;
	       uoldIDS[im] = uIDS[i] + (fluxIDS[i2] - fluxIDS[i1]) * dtdx;
	       uoldIVS[im] = uIVS[i] + (fluxIVS[i2] - fluxIVS[i1]) * dtdx;
	       uoldIUS[im] = uIUS[i] + (fluxIUS[i2] - fluxIUS[i1]) * dtdx; 
	       uoldIPS[im] = uIPS[i] + (fluxIPS[i2] - fluxIPS[i1]) * dtdx;
		}
	);
#endif
}

void Tile::updateconservYscan(int32_t s, int32_t xmin, int32_t xmax,
			      int32_t ymin, int32_t ymax, real_t dtdx,
			      Matrix2 < real_t > &uoldID,
			      Matrix2 < real_t > &uoldIP,
			      Matrix2 < real_t > &uoldIV,
			      Matrix2 < real_t > &uoldIU,
			      Preal_t fluxIVS, Preal_t fluxIUS, Preal_t fluxIPS, Preal_t fluxIDS, Preal_t uIDS, Preal_t uIPS, Preal_t uIVS, Preal_t uIUS, Preal_t pl)
{
#if TILEUSER == 0
#pragma loop_count min=TILEMIN, avg=TILESIZ
#endif
#pragma omp simd
	for (int32_t j = xmin; j < xmax; j++) {
		pl[j] = uIDS[j] + (fluxIDS[j - 2] - fluxIDS[j - 1]) * dtdx;
	}
	uoldID.putFullCol(s + m_offx, xmin + m_offy, &pl[xmin], (xmax - xmin));

#if TILEUSER == 0
#pragma loop_count min=TILEMIN, avg=TILESIZ
#endif
#pragma omp simd
	for (int32_t j = xmin; j < xmax; j++) {
		pl[j] = uIUS[j] + (fluxIUS[j - 2] - fluxIUS[j - 1]) * dtdx;
	}
	uoldIV.putFullCol(s + m_offx, xmin + m_offy, &pl[xmin], (xmax - xmin));

#if TILEUSER == 0
#pragma loop_count min=TILEMIN, avg=TILESIZ
#endif
#pragma omp simd
	for (int32_t j = xmin; j < xmax; j++) {
		pl[j] = uIVS[j] + (fluxIVS[j - 2] - fluxIVS[j - 1]) * dtdx;
	}
	uoldIU.putFullCol(s + m_offx, xmin + m_offy, &pl[xmin], (xmax - xmin));

#if TILEUSER == 0
#pragma loop_count min=TILEMIN, avg=TILESIZ
#endif
#pragma omp simd
	for (int32_t j = xmin; j < xmax; j++) {
		pl[j] = uIPS[j] + (fluxIPS[j - 2] - fluxIPS[j - 1]) * dtdx;
	}
	uoldIP.putFullCol(s + m_offx, xmin + m_offy, &pl[xmin], (xmax - xmin));
}

void Tile::updateconserv()
{
	int32_t xmin, xmax, ymin, ymax;
	Matrix2 < real_t > &uoldID = *(*m_uold) (ID_VAR);
	Matrix2 < real_t > &uoldIP = *(*m_uold) (IP_VAR);
	Matrix2 < real_t > &uoldIV = *(*m_uold) (IV_VAR);
	Matrix2 < real_t > &uoldIU = *(*m_uold) (IU_VAR);
	Matrix2 < real_t > &fluxIV = *(*m_flux) (IV_VAR);
	Matrix2 < real_t > &fluxIU = *(*m_flux) (IU_VAR);
	Matrix2 < real_t > &fluxIP = *(*m_flux) (IP_VAR);
	Matrix2 < real_t > &fluxID = *(*m_flux) (ID_VAR);
	Matrix2 < real_t > &uID = *(*m_u) (ID_VAR);
	Matrix2 < real_t > &uIP = *(*m_u) (IP_VAR);
	Matrix2 < real_t > &uIV = *(*m_u) (IV_VAR);
	Matrix2 < real_t > &uIU = *(*m_u) (IU_VAR);
	real_t dtdx = m_dt / m_dx;
	if (m_prt)
		cout << "dtdx " << dtdx << endl;

	getExtends(TILE_INTERIOR, xmin, xmax, ymin, ymax);
	if (m_prt)
		cout << "scan " << m_scan << endl;
	if (m_prt)
		uoldIP.printFormatted("Tile uoldIP input updateconserv");
	if (m_prt)
		fluxID.printFormatted("Tile fluxID input updateconserv");
	if (m_prt)
		fluxIU.printFormatted("Tile fluxIU input updateconserv");
	if (m_prt)
		fluxIV.printFormatted("Tile fluxIV input updateconserv");
	if (m_prt)
		fluxIP.printFormatted("Tile fluxIP input updateconserv");
	if (m_prt)
		uID.printFormatted("Tile uID updateconserv");
	if (m_prt)
		uIU.printFormatted("Tile uIU updateconserv");
	if (m_prt)
		uIV.printFormatted("Tile uIPV updateconserv");
	if (m_prt)
		uIP.printFormatted("Tile uIP updateconserv");

	if (m_scan == X_SCAN) {
		for (int32_t s = ymin; s < ymax; s++) {
			Preal_t uoldIDS = uoldID.getRow(s + m_offy);
			Preal_t uoldIPS = uoldIP.getRow(s + m_offy);
			Preal_t uoldIVS = uoldIV.getRow(s + m_offy);
			Preal_t uoldIUS = uoldIU.getRow(s + m_offy);
			Preal_t uIDS = uID.getRow(s);
			Preal_t uIPS = uIP.getRow(s);
			Preal_t uIVS = uIV.getRow(s);
			Preal_t uIUS = uIU.getRow(s);
			Preal_t fluxIDS = fluxID.getRow(s);
			Preal_t fluxIVS = fluxIV.getRow(s);
			Preal_t fluxIUS = fluxIU.getRow(s);
			Preal_t fluxIPS = fluxIP.getRow(s);
			updateconservXscan(xmin, xmax, dtdx, uIDS, uIUS, uIVS, uIPS, uoldIDS, uoldIUS, uoldIVS, uoldIPS, fluxIDS, fluxIVS, fluxIUS, fluxIPS);
		}
	} else {
		for (int32_t s = ymin; s < ymax; s++) {
			Preal_t fluxIVS = fluxIV.getRow(s);
			Preal_t fluxIUS = fluxIU.getRow(s);
			Preal_t fluxIPS = fluxIP.getRow(s);
			Preal_t fluxIDS = fluxID.getRow(s);
			Preal_t uIDS = uID.getRow(s);
			Preal_t uIPS = uIP.getRow(s);
			Preal_t uIVS = uIV.getRow(s);
			Preal_t uIUS = uIU.getRow(s);
			Preal_t pl = m_pl;

			updateconservYscan(s, xmin, xmax, ymin, ymax, dtdx, uoldID, uoldIP, uoldIV, uoldIU, fluxIVS, fluxIUS, fluxIPS, fluxIDS, uIDS, uIPS, uIVS, uIUS, pl);

			// for (int32_t j = xmin; j < xmax; j++) {
			// uoldID(s + m_offx, j + m_offy) = uID(j, s) + (fluxID(j - 2, s) - fluxID(j - 1, s)) * dtdx;
			// uoldIV(s + m_offx, j + m_offy) = uIU(j, s) + (fluxIU(j - 2, s) - fluxIU(j - 1, s)) * dtdx;
			// uoldIU(s + m_offx, j + m_offy) = uIV(j, s) + (fluxIV(j - 2, s) - fluxIV(j - 1, s)) * dtdx;
			// uoldIP(s + m_offx, j + m_offy) = uIP(j, s) + (fluxIP(j - 2, s) - fluxIP(j - 1, s)) * dtdx;
			// }
		}
	}
	if (m_prt)
		uoldID.printFormatted("Tile uoldID updateconserv");
	if (m_prt)
		uoldIU.printFormatted("Tile uoldIU updateconserv");
	if (m_prt)
		uoldIV.printFormatted("Tile uoldIV updateconserv");
	if (m_prt)
		uoldIP.printFormatted("Tile uoldIP updateconserv");
}

void Tile::gatherconservXscan(int32_t xmin, int32_t xmax,
			      Preal_t uIDS, Preal_t uIUS, Preal_t uIVS, Preal_t uIPS, Preal_t uoldIDS, Preal_t uoldIUS, Preal_t uoldIVS, Preal_t uoldIPS)
{
#if ALIGNED > 0
	// #pragma vector aligned // impossible !
#endif
#if TILEUSER == 0
#pragma loop_count min=TILEMIN, avg=TILEAVG
#endif
#pragma omp simd
	for (int32_t i = xmin; i < xmax; i++) {
		uIDS[i] = uoldIDS[i + m_offx];
	}
#pragma omp simd
	for (int32_t i = xmin; i < xmax; i++) {
		uIUS[i] = uoldIUS[i + m_offx];
	}
#pragma omp simd
	for (int32_t i = xmin; i < xmax; i++) {
		uIVS[i] = uoldIVS[i + m_offx];
	}
#pragma omp simd
	for (int32_t i = xmin; i < xmax; i++) {
		uIPS[i] = uoldIPS[i + m_offx];
	}
}

void Tile::gatherconservYscan()
{
}

void Tile::gatherconserv()
{
	int32_t xmin, xmax, ymin, ymax;
	Matrix2 < real_t > &uID = *(*m_u) (ID_VAR);
	Matrix2 < real_t > &uIP = *(*m_u) (IP_VAR);
	Matrix2 < real_t > &uIV = *(*m_u) (IV_VAR);
	Matrix2 < real_t > &uIU = *(*m_u) (IU_VAR);
	Matrix2 < real_t > &uoldID = *(*m_uold) (ID_VAR);
	Matrix2 < real_t > &uoldIP = *(*m_uold) (IP_VAR);
	Matrix2 < real_t > &uoldIV = *(*m_uold) (IV_VAR);
	Matrix2 < real_t > &uoldIU = *(*m_uold) (IU_VAR);

	getExtends(TILE_FULL, xmin, xmax, ymin, ymax);

	if (m_prt)
		uoldID.printFormatted("Tile uoldID gatherconserv");
	if (m_prt)
		uoldIU.printFormatted("Tile uoldIU gatherconserv");
	if (m_prt)
		uoldIV.printFormatted("Tile uoldIV gatherconserv");
	if (m_prt)
		uoldIP.printFormatted("Tile uoldIP gatherconserv");

	if (m_scan == X_SCAN) {
		for (int32_t s = ymin; s < ymax; s++) {
			real_t *uoldIDS = uoldID.getRow(s + m_offy);
			real_t *uoldIPS = uoldIP.getRow(s + m_offy);
			real_t *uoldIVS = uoldIV.getRow(s + m_offy);
			real_t *uoldIUS = uoldIU.getRow(s + m_offy);
			real_t *uIDS = uID.getRow(s);
			real_t *uIPS = uIP.getRow(s);
			real_t *uIVS = uIV.getRow(s);
			real_t *uIUS = uIU.getRow(s);
			gatherconservXscan(xmin, xmax, uIDS, uIUS, uIVS, uIPS, uoldIDS, uoldIUS, uoldIVS, uoldIPS);
		}
	} else {
		for (int32_t j = xmin; j < xmax; j++) {
			uID.putFullCol(j, 0, uoldID.getRow(j + m_offy) + m_offx, (ymax - ymin));
			uIU.putFullCol(j, 0, uoldIV.getRow(j + m_offy) + m_offx, (ymax - ymin));
			uIV.putFullCol(j, 0, uoldIU.getRow(j + m_offy) + m_offx, (ymax - ymin));
			uIP.putFullCol(j, 0, uoldIP.getRow(j + m_offy) + m_offx, (ymax - ymin));

			// for (int32_t s = ymin; s < ymax; s++) {
			//        uID(j, s) = uoldID(s + m_offx, j + m_offy);
			//        uIU(j, s) = uoldIV(s + m_offx, j + m_offy);
			//        uIV(j, s) = uoldIU(s + m_offx, j + m_offy);
			//        uIP(j, s) = uoldIP(s + m_offx, j + m_offy);
			// }
		}
	}
	if (m_prt)
		uID.printFormatted("Tile uID gatherconserv");
	if (m_prt)
		uIU.printFormatted("Tile uIU gatherconserv");
	if (m_prt)
		uIV.printFormatted("Tile uIV gatherconserv");
	if (m_prt)
		uIP.printFormatted("Tile uIP gatherconserv");
}

void Tile::eosOnRow(int32_t xmin, int32_t xmax, real_t smallp, Preal_t qIDS, Preal_t eS, Preal_t qIPS, Preal_t cS)
{
	if (xmin > 0) {
#pragma omp simd
		for (int32_t k = xmin; k < xmax; k++) {
			real_t rho = qIDS[k];
			real_t rrho = 1. / rho;
			real_t base = (m_gamma - one) * rho * eS[k];;
			base = Max(base, (real_t) (rho * smallp));
			qIPS[k] = base;
			cS[k] = sqrt(m_gamma * base * rrho);
		}
	} else {
#if ALIGNED > 0
#pragma vector aligned
#if TILEUSER == 0
#pragma loop_count min=TILEMIN, avg=TILEAVG
#endif
#endif
#pragma omp simd
		for (int32_t k = xmin; k < xmax; k++) {
			real_t rho = qIDS[k];
			real_t rrho = 1. / rho;
			real_t base = (m_gamma - one) * rho * eS[k];;
			base = Max(base, (real_t) (rho * smallp));
			qIPS[k] = base;
			cS[k] = sqrt(m_gamma * base * rrho);
		}
	}
}

void Tile::eos(tileSpan_t span)
{
	int32_t xmin, xmax, ymin, ymax;

	Matrix2 < real_t > &qID = *(*m_q) (ID_VAR);
	Matrix2 < real_t > &qIP = *(*m_q) (IP_VAR);

	real_t smallp = Square(m_smallc) / m_gamma;

	getExtends(span, xmin, xmax, ymin, ymax);

	for (int32_t s = ymin; s < ymax; s++) {
		real_t *qIDS = qID.getRow(s);
		real_t *eS = (*m_e).getRow(s);
		real_t *qIPS = qIP.getRow(s);
		real_t *cS = (*m_c).getRow(s);
		eosOnRow(xmin, xmax, smallp, qIDS, eS, qIPS, cS);
	}
	if (m_prt)
		qIP.printFormatted("Tile qIP eos");
	if (m_prt)
		m_c->printFormatted("Tile c eos");
}

void Tile::compute_dt_loop1OnRow(int32_t xmin, int32_t xmax,
				 Preal_t qIDS, Preal_t qIPS, Preal_t qIUS, Preal_t qIVS, Preal_t uoldIDS, Preal_t uoldIUS, Preal_t uoldIVS, Preal_t uoldIPS, Preal_t eS)
{
	for (int32_t i = xmin; i < xmax; i++) {
		real_t eken, tmp;
		qIDS[i] = uoldIDS[i + m_offx];
		qIDS[i] = Max(qIDS[i], m_smallr);
		qIUS[i] = uoldIUS[i + m_offx] / qIDS[i];
		qIVS[i] = uoldIVS[i + m_offx] / qIDS[i];
		eken = half * (Square(qIUS[i]) + Square(qIVS[i]));
		tmp = uoldIPS[i + m_offx] / qIDS[i] - eken;
		qIPS[i] = tmp;
		eS[i] = tmp;
	}
}

void Tile::compute_dt_loop2OnRow(real_t & tmp1, real_t & tmp2, int32_t xmin, int32_t xmax, Preal_t cS, Preal_t qIUS, Preal_t qIVS)
{
	for (int32_t i = xmin; i < xmax; i++) {
		tmp1 = Max(tmp1, cS[i] + Fabs(qIUS[i]));
	}
	for (int32_t i = xmin; i < xmax; i++) {
		tmp2 = Max(tmp2, cS[i] + Fabs(qIVS[i]));
	}
}

real_t Tile::compute_dt()
{
	int32_t xmin, xmax, ymin, ymax;
	real_t dt = 0, cournox, cournoy, tmp1 = 0, tmp2 = 0;
	Matrix2 < real_t > &uoldID = *(*m_uold) (ID_VAR);
	Matrix2 < real_t > &uoldIP = *(*m_uold) (IP_VAR);
	Matrix2 < real_t > &uoldIV = *(*m_uold) (IV_VAR);
	Matrix2 < real_t > &uoldIU = *(*m_uold) (IU_VAR);
	Matrix2 < real_t > &qID = *(*m_q) (ID_VAR);
	Matrix2 < real_t > &qIP = *(*m_q) (IP_VAR);
	Matrix2 < real_t > &qIV = *(*m_q) (IV_VAR);
	Matrix2 < real_t > &qIU = *(*m_q) (IU_VAR);
	godunovDir_t oldScan = m_scan;

	if (m_scan == Y_SCAN) {
		swapScan();
		swapStorageDims();
	}

	getExtends(TILE_INTERIOR, xmin, xmax, ymin, ymax);

	cournox = zero;
	cournoy = zero;

	for (int32_t s = ymin; s < ymax; s++) {
		real_t *uoldIDS;
		real_t *uoldIPS;
		real_t *uoldIVS;
		real_t *uoldIUS;
		uoldIDS = uoldID.getRow(s + m_offy);
		uoldIPS = uoldIP.getRow(s + m_offy);
		uoldIVS = uoldIV.getRow(s + m_offy);
		uoldIUS = uoldIU.getRow(s + m_offy);

		real_t *qIDS = qID.getRow(s);
		real_t *qIPS = qIP.getRow(s);
		real_t *qIVS = qIV.getRow(s);
		real_t *qIUS = qIU.getRow(s);
		real_t *eS = (*m_e).getRow(s);
		compute_dt_loop1OnRow(xmin, xmax, qIDS, qIDS, qIUS, qIVS, uoldIDS, uoldIUS, uoldIVS, uoldIPS, eS);
	}

	eos(TILE_INTERIOR);	// needs    qID, e    returns    c, qIP

	for (int32_t s = ymin; s < ymax; s++) {
		real_t *qIVS = qIV.getRow(s);
		real_t *qIUS = qIU.getRow(s);
		real_t *cS = (*m_c).getRow(s);
		compute_dt_loop2OnRow(tmp1, tmp2, xmin, xmax, cS, qIUS, qIVS);
	}
	cournox = Max(cournox, tmp1);
	cournoy = Max(cournoy, tmp2);

	dt = m_cfl * m_dx / Max(cournox, Max(cournoy, m_smallc));

	if (m_scan != oldScan) {
		swapScan();
		swapStorageDims();
	}

	if (m_prt)
		cerr << "tile dt " << dt << endl;
	return dt;
}

void Tile::constprimOnRow(int32_t xmin, int32_t xmax, Preal_t qIDS, Preal_t qIPS, Preal_t qIVS, Preal_t qIUS, Preal_t uIDS, Preal_t uIPS, Preal_t uIVS, Preal_t uIUS, Preal_t eS)
{

#if ALIGNED > 0
#pragma message "constprimOnRow aligned"
#pragma vector aligned
#if TILEUSER == 0
#pragma loop_count min=TILEMIN, avg=TILEAVG
#endif
#endif
	for (int32_t i = xmin; i < xmax; i++) {
		register real_t eken, qid, qiu, qiv, qip;
		qid = uIDS[i];
		qid = Max(qid, m_smallr);
		// if (qid < m_smallr) qid = m_smallr;
		qiu = uIUS[i] / qid;
		qiv = uIVS[i] / qid;

		eken = half * (Square(qiu) + Square(qiv));

		qip = uIPS[i] / qid - eken;
		qIUS[i] = qiu;
		qIVS[i] = qiv;
		qIDS[i] = qid;
		qIPS[i] = qip;
		eS[i] = qip;
	}
}

void Tile::constprim()
{
	int32_t xmin, xmax, ymin, ymax;
	Matrix2 < real_t > &qID = *(*m_q) (ID_VAR);
	Matrix2 < real_t > &qIP = *(*m_q) (IP_VAR);
	Matrix2 < real_t > &qIV = *(*m_q) (IV_VAR);
	Matrix2 < real_t > &qIU = *(*m_q) (IU_VAR);

	Matrix2 < real_t > &uID = *(*m_u) (ID_VAR);
	Matrix2 < real_t > &uIP = *(*m_u) (IP_VAR);
	Matrix2 < real_t > &uIV = *(*m_u) (IV_VAR);
	Matrix2 < real_t > &uIU = *(*m_u) (IU_VAR);

	getExtends(TILE_FULL, xmin, xmax, ymin, ymax);

	for (int32_t s = ymin; s < ymax; s++) {
		real_t *eS = (*m_e).getRow(s);
		real_t *qIDS = qID.getRow(s);
		real_t *qIPS = qIP.getRow(s);
		real_t *qIVS = qIV.getRow(s);
		real_t *qIUS = qIU.getRow(s);
		real_t *uIDS = uID.getRow(s);
		real_t *uIPS = uIP.getRow(s);
		real_t *uIVS = uIV.getRow(s);
		real_t *uIUS = uIU.getRow(s);
		constprimOnRow(xmin, xmax, qIDS, qIPS, qIVS, qIUS, uIDS, uIPS, uIVS, uIUS, eS);
	}
	if (m_prt)
		qIP.printFormatted("Tile qIP constprim");
	if (m_prt)
		(*m_e).printFormatted("Tile e constprim");
}

void Tile::riemannOnRow(int32_t xmin, int32_t xmax, real_t smallp,
			real_t gamma6, real_t smallpp,
			Preal_t qgdnvIDS,
			Preal_t qgdnvIUS,
			Preal_t qgdnvIPS,
			Preal_t qgdnvIVS,
			Preal_t qleftIDS,
			Preal_t qleftIUS,
			Preal_t qleftIPS,
			Preal_t qleftIVS,
			Preal_t qrightIDS,
			Preal_t qrightIUS,
			Preal_t qrightIPS,
			Preal_t qrightIVS,
			long *__restrict__ goon, Preal_t sgnm, Preal_t pstar, Preal_t rl, Preal_t ul, Preal_t pl, Preal_t rr, Preal_t ur, Preal_t pr, Preal_t cl, Preal_t cr)
{
#pragma message "riemannOnRow actif"
#if ALIGNED > 0
#pragma vector aligned
#if TILEUSER == 0
#pragma loop_count min=TILEMIN, avg=TILEAVG
#endif
#endif
	for (int32_t i = xmin; i < xmax; i++) {
		goon[i] = 1;
	}

	// Precompute values for this slice
	// #pragma ivdep
#if ALIGNED > 0
#pragma vector aligned
#if TILEUSER == 0
#pragma loop_count min=TILEMIN, avg=TILEAVG
#endif
#endif
	for (int32_t i = xmin; i < xmax; i++) {
		real_t wl_i, wr_i;
		rl[i] = Max(qleftIDS[i], m_smallr);
		ul[i] = qleftIUS[i];
		pl[i] = Max(qleftIPS[i], rl[i] * smallp);
		rr[i] = Max(qrightIDS[i], m_smallr);
		ur[i] = qrightIUS[i];
		pr[i] = Max(qrightIPS[i], rr[i] * smallp);

		// Lagrangian sound speed
		cl[i] = m_gamma * pl[i] * rl[i];
		cr[i] = m_gamma * pr[i] * rr[i];

		// First guess
		wl_i = sqrt(cl[i]);
		wr_i = sqrt(cr[i]);
		pstar[i] = Max(((wr_i * pl[i] + wl_i * pr[i]) + wl_i * wr_i * (ul[i] - ur[i])) / (wl_i + wr_i), 0.0);
	}

	// solve the riemann problem on the interfaces of this slice
	// for (int32_t iter = 0; iter < m_niter_riemann; iter++) {
#pragma loop_count min=1, max=20, avg=10
// #pragma unroll(5)
	for (int32_t iter = 0; iter < m_niter_riemann; iter++) {
#if ALIGNED > 0
#pragma vector aligned
#if TILEUSER == 0
#pragma loop_count min=TILEMIN, avg=TILEAVG
#endif
#endif
		for (int32_t i = xmin; i < xmax; i++) {
			if (goon[i] > 0) {
				real_t pst = pstar[i];
				// Newton-Raphson iterations to find pstar at the required accuracy
				real_t wwl = sqrt(cl[i] * (one + gamma6 * (pst - pl[i]) / pl[i]));
				real_t wwr = sqrt(cr[i] * (one + gamma6 * (pst - pr[i]) / pr[i]));
				real_t swwl = Square(wwl);
				real_t ql = two * wwl * swwl / (swwl + cl[i]);
				real_t qr = two * wwr * Square(wwr) / (Square(wwr) + cr[i]);
				real_t usl = ul[i] - (pst - pl[i]) / wwl;
				real_t usr = ur[i] + (pst - pr[i]) / wwr;
				real_t tmp = (qr * ql / (qr + ql) * (usl - usr));
				real_t delp_i = Max(tmp, (-pst));
				// pstar[i] = pstar[i] + delp_i;
				pst += delp_i;
				// Convergence indicator
				real_t tmp2 = delp_i / (pst + smallpp);
				real_t uo_i = Fabs(tmp2);
				goon[i] = uo_i > PRECISION;
				// FLOPS(29, 10, 2, 0);
				pstar[i] = pst;
			}
		}
	}			// iter_riemann
#if ALIGNED > 0
#pragma vector aligned
#if TILEUSER == 0
#pragma loop_count min=TILEMIN, avg=TILEAVG
#endif
#endif
	for (int32_t i = xmin; i < xmax; i++) {
		real_t wl_i = sqrt(cl[i]);
		real_t wr_i = sqrt(cr[i]);

		wr_i = sqrt(cr[i] * (one + gamma6 * (pstar[i] - pr[i]) / pr[i]));
		wl_i = sqrt(cl[i] * (one + gamma6 * (pstar[i] - pl[i]) / pl[i]));

		real_t ustar_i = half * (ul[i] + (pl[i] - pstar[i]) / wl_i + ur[i] - (pr[i] - pstar[i]) / wr_i);

		real_t left = ustar_i > 0;

		real_t ro_i, uo_i, po_i, wo_i;

		// if (left) {
		//   sgnm[i] = 1;
		//   ro_i = rl[i];
		//   uo_i = ul[i];
		//   po_i = pl[i];
		//   wo_i = wl_i;
		// } else {
		//   sgnm[i] = -1;
		//   ro_i = rr[i];
		//   uo_i = ur[i];
		//   po_i = pr[i];
		//   wo_i = wr_i;
		// }
		sgnm[i] = 1 * left + (-1 + left);
		ro_i = left * rl[i] + (1 - left) * rr[i];
		uo_i = left * ul[i] + (1 - left) * ur[i];
		po_i = left * pl[i] + (1 - left) * pr[i];
		wo_i = left * wl_i + (1 - left) * wr_i;

		real_t co_i = sqrt(Fabs(m_gamma * po_i / ro_i));
		co_i = Max(m_smallc, co_i);

		real_t rstar_i = ro_i / (one + ro_i * (po_i - pstar[i]) / Square(wo_i));
		rstar_i = Max(rstar_i, m_smallr);

		real_t cstar_i = sqrt(Fabs(m_gamma * pstar[i] / rstar_i));
		cstar_i = Max(m_smallc, cstar_i);

		real_t spout_i = co_i - sgnm[i] * uo_i;
		real_t spin_i = cstar_i - sgnm[i] * ustar_i;
		real_t ushock_i = wo_i / ro_i - sgnm[i] * uo_i;

		if (pstar[i] >= po_i) {
			spin_i = ushock_i;
			spout_i = ushock_i;
		}

		real_t scr_i = Max((real_t) (spout_i - spin_i),
				   (real_t) (m_smallc + Fabs(spout_i + spin_i)));

		real_t frac_i = (one + (spout_i + spin_i) / scr_i) * half;
		frac_i = Max(zero, (real_t) (Min(one, frac_i)));

		int addSpout = spout_i < zero;
		int addSpin = spin_i > zero;
		// real_t originalQgdnv = !addSpout & !addSpin;
		real_t qgdnv_ID, qgdnv_IU, qgdnv_IP;

		if (addSpout) {
			qgdnv_ID = ro_i;
			qgdnv_IU = uo_i;
			qgdnv_IP = po_i;
		} else if (addSpin) {
			qgdnv_ID = rstar_i;
			qgdnv_IU = ustar_i;
			qgdnv_IP = pstar[i];
		} else {
			qgdnv_ID = (frac_i * rstar_i + (one - frac_i) * ro_i);
			qgdnv_IU = (frac_i * ustar_i + (one - frac_i) * uo_i);
			qgdnv_IP = (frac_i * pstar[i] + (one - frac_i) * po_i);
		}

		qgdnvIDS[i] = qgdnv_ID;
		qgdnvIUS[i] = qgdnv_IU;
		qgdnvIPS[i] = qgdnv_IP;

		// transverse velocity
		// if (left) {
		//   qgdnvIVS[i] = qleftIVS[i];
		// } else {
		//   qgdnvIVS[i] = qrightIVS[i];
		// }
		qgdnvIVS[i] = left * qleftIVS[i] + (1.0 - left) * qrightIVS[i];
	}
}

void Tile::riemannOnRowInRegs(int32_t xmin, int32_t xmax, real_t smallp,
			      real_t gamma6, real_t smallpp,
			      Preal_t qgdnvIDS,
			      Preal_t qgdnvIUS,
			      Preal_t qgdnvIPS,
			      Preal_t qgdnvIVS,
			      Preal_t qleftIDS,
			      Preal_t qleftIUS, Preal_t qleftIPS, Preal_t qleftIVS, Preal_t qrightIDS, Preal_t qrightIUS, Preal_t qrightIPS, Preal_t qrightIVS, Preal_t sgnm)
{
#pragma message "riemannOnRowInRegs actif"
#if ALIGNED > 0
#pragma vector aligned
#if TILEUSER == 0
#pragma loop_count min=TILEMIN, avg=TILEAVG
#endif
#endif

#if OMPOVERLOAD == 1
#pragma omp parallel for private(i) shared(qgdnvIDS, qgdnvIUS, qgdnvIPS, qgdnvIVS, sgnm)
#endif
#pragma omp simd
	for (int32_t i = xmin; i < xmax; i++) {
		int goonI = 0;
		real_t pstarI;
		real_t rlI;
		real_t ulI;
		real_t plI;
		real_t rrI;
		real_t urI;
		real_t prI;
		real_t clI;
		real_t crI;

		goonI = 1;

		// Precompute values for this slice
		real_t wl_i, wr_i;
		rlI = Max(qleftIDS[i], m_smallr);
		ulI = qleftIUS[i];
		plI = Max(qleftIPS[i], rlI * smallp);
		rrI = Max(qrightIDS[i], m_smallr);
		urI = qrightIUS[i];
		prI = Max(qrightIPS[i], rrI * smallp);

		// Lagrangian sound speed
		clI = m_gamma * plI * rlI;
		crI = m_gamma * prI * rrI;

		// First guess
		wl_i = sqrt(clI);
		wr_i = sqrt(crI);
		pstarI = Max(((wr_i * plI + wl_i * prI) + wl_i * wr_i * (ulI - urI)) / (wl_i + wr_i), 0.0);
//  #pragma ivdep
		for (int32_t iter = 0; iter < 10; iter++) {
			if (goonI > 0) {
				real_t pst = pstarI;
				// Newton-Raphson iterations to find pstar at the required accuracy
				real_t wwl = sqrt(clI * (one + gamma6 * (pst - plI) / plI));
				real_t wwr = sqrt(crI * (one + gamma6 * (pst - prI) / prI));
				real_t swwl = Square(wwl);
				real_t ql = two * wwl * swwl / (swwl + clI);
				real_t qr = two * wwr * Square(wwr) / (Square(wwr) + crI);
				real_t usl = ulI - (pst - plI) / wwl;
				real_t usr = urI + (pst - prI) / wwr;
				real_t tmp = (qr * ql / (qr + ql) * (usl - usr));
				real_t delp_i = Max(tmp, (-pst));
				// pstarI = pstarI + delp_i;
				pst += delp_i;
				// Convergence indicator
				real_t tmp2 = delp_i / (pst + smallpp);
				real_t uo_i = Fabs(tmp2);
				goonI = uo_i > PRECISION;
				// FLOPS(29, 10, 2, 0);
				pstarI = pst;
			}
			// // Optimization MIC 
			// // Delete the if condition to optimize the vectorisation
			// //if (goonI > 0) {
			// real_t pst = pstarI;
			// // Newton-Raphson iterations to find pstar at the required accuracy
			// real_t wwl = sqrt(clI * (one + gamma6 * (pst - plI) / plI));
			// real_t wwr = sqrt(crI * (one + gamma6 * (pst - prI) / prI));
			// real_t swwl = Square(wwl);
			// real_t ql = two * wwl * swwl / (swwl + clI);
			// real_t qr = two * wwr * Square(wwr) / (Square(wwr) + crI);
			// real_t usl = ulI - (pst - plI) / wwl;
			// real_t usr = urI + (pst - prI) / wwr;
			// real_t tmp = (qr * ql / (qr + ql) * (usl - usr));
			// real_t delp_i = Max(tmp, (-pst));
			// // pstarI = pstarI + delp_i;
			// pst += delp_i;
			// // Convergence indicator
			// real_t tmp2 = delp_i / (pst + smallpp);
			// real_t uo_i = Fabs(tmp2);
			// // Optimization MIC 
			// // We calculate the goonI and the pstarI with the goonI variable condition
			// //goonI = uo_i > PRECISION;
			// real_t rgoonI = (real_t) goonI;
			// pstarI = (pst * rgoonI) + (pstarI * (1.0L - rgoonI));
			// goonI = (uo_i > PRECISION) * goonI;
			// // FLOPS(29, 10, 2, 0);
			// //pstarI = pst;
			// //}
		}
		wr_i = sqrt(crI * (one + gamma6 * (pstarI - prI) / prI));
		wl_i = sqrt(clI * (one + gamma6 * (pstarI - plI) / plI));

		real_t ustar_i = half * (ulI + (plI - pstarI) / wl_i + urI - (prI - pstarI) / wr_i);

		int left = ustar_i > 0;

		real_t ro_i, uo_i, po_i, wo_i;

		sgnm[i] = 1 * left + (-1 + left);
		ro_i = left * rlI + (1 - left) * rrI;
		uo_i = left * ulI + (1 - left) * urI;
		po_i = left * plI + (1 - left) * prI;
		wo_i = left * wl_i + (1 - left) * wr_i;

		real_t co_i = sqrt(Fabs(m_gamma * po_i / ro_i));
		co_i = Max(m_smallc, co_i);

		real_t rstar_i = ro_i / (one + ro_i * (po_i - pstarI) / Square(wo_i));
		rstar_i = Max(rstar_i, m_smallr);

		real_t cstar_i = sqrt(Fabs(m_gamma * pstarI / rstar_i));
		cstar_i = Max(m_smallc, cstar_i);

		real_t spout_i = co_i - sgnm[i] * uo_i;
		real_t spin_i = cstar_i - sgnm[i] * ustar_i;
		real_t ushock_i = wo_i / ro_i - sgnm[i] * uo_i;

		if (pstarI >= po_i) {
			spin_i = ushock_i;
			spout_i = ushock_i;
		}

		real_t scr_i = Max((real_t) (spout_i - spin_i),
				   (real_t) (m_smallc + Fabs(spout_i + spin_i)));

		real_t frac_i = (one + (spout_i + spin_i) / scr_i) * half;
		frac_i = Max(zero, (real_t) (Min(one, frac_i)));

		int addSpout = spout_i < zero;
		int addSpin = spin_i > zero;
		// real_t originalQgdnv = !addSpout & !addSpin;
		real_t qgdnv_ID, qgdnv_IU, qgdnv_IP;

		if (addSpout) {
			qgdnv_ID = ro_i;
			qgdnv_IU = uo_i;
			qgdnv_IP = po_i;
		} else if (addSpin) {
			qgdnv_ID = rstar_i;
			qgdnv_IU = ustar_i;
			qgdnv_IP = pstarI;
		} else {
			qgdnv_ID = (frac_i * rstar_i + (one - frac_i) * ro_i);
			qgdnv_IU = (frac_i * ustar_i + (one - frac_i) * uo_i);
			qgdnv_IP = (frac_i * pstarI + (one - frac_i) * po_i);
		}

		qgdnvIDS[i] = qgdnv_ID;
		qgdnvIUS[i] = qgdnv_IU;
		qgdnvIPS[i] = qgdnv_IP;

		// transverse velocity
		// if (left) {
		//   qgdnvIVS[i] = qleftIVS[i];
		// } else {
		//   qgdnvIVS[i] = qrightIVS[i];
		// }
		qgdnvIVS[i] = left * qleftIVS[i] + (1.0 - left) * qrightIVS[i];
	}
}

void Tile::riemann()
{
	int32_t xmin, xmax, ymin, ymax;
	Matrix2 < real_t > &qgdnvID = *(*m_qgdnv) (ID_VAR);
	Matrix2 < real_t > &qgdnvIU = *(*m_qgdnv) (IU_VAR);
	Matrix2 < real_t > &qgdnvIP = *(*m_qgdnv) (IP_VAR);
	Matrix2 < real_t > &qgdnvIV = *(*m_qgdnv) (IV_VAR);
	Matrix2 < real_t > &qleftID = *(*m_qleft) (ID_VAR);
	Matrix2 < real_t > &qleftIU = *(*m_qleft) (IU_VAR);
	Matrix2 < real_t > &qleftIP = *(*m_qleft) (IP_VAR);
	Matrix2 < real_t > &qleftIV = *(*m_qleft) (IV_VAR);
	Matrix2 < real_t > &qrightID = *(*m_qright) (ID_VAR);
	Matrix2 < real_t > &qrightIU = *(*m_qright) (IU_VAR);
	Matrix2 < real_t > &qrightIP = *(*m_qright) (IP_VAR);
	Matrix2 < real_t > &qrightIV = *(*m_qright) (IV_VAR);

	real_t smallp = Square(m_smallc) / m_gamma;
	real_t gamma6 = (m_gamma + one) / (two * m_gamma);
	real_t smallpp = m_smallr * smallp;

	getExtends(TILE_FULL, xmin, xmax, ymin, ymax);

	for (int32_t s = ymin; s < ymax; s++) {
		real_t *qgdnvIDS = qgdnvID.getRow(s);
		real_t *qgdnvIUS = qgdnvIU.getRow(s);
		real_t *qgdnvIPS = qgdnvIP.getRow(s);
		real_t *qgdnvIVS = qgdnvIV.getRow(s);
		real_t *qleftIDS = qleftID.getRow(s);
		real_t *qleftIUS = qleftIU.getRow(s);
		real_t *qleftIPS = qleftIP.getRow(s);
		real_t *qleftIVS = qleftIV.getRow(s);
		real_t *qrightIDS = qrightID.getRow(s);
		real_t *qrightIUS = qrightIU.getRow(s);
		real_t *qrightIPS = qrightIP.getRow(s);
		real_t *qrightIVS = qrightIV.getRow(s);
#if RIEMANNINREGS == 0
#pragma message "==> riemannOnRow selectionne"
		riemannOnRow(xmin, xmax, smallp, gamma6, smallpp,
			     qgdnvIDS, qgdnvIUS, qgdnvIPS, qgdnvIVS, qleftIDS,
			     qleftIUS, qleftIPS, qleftIVS, qrightIDS, qrightIUS, qrightIPS, qrightIVS, m_goon, m_sgnm, m_pstar, m_rl, m_ul, m_pl, m_rr, m_ur, m_pr, m_cl, m_cr);
#else
#pragma message "==> riemannOnRowInRegs selectionne"
		riemannOnRowInRegs(xmin, xmax, smallp,
				   gamma6, smallpp,
				   qgdnvIDS, qgdnvIUS, qgdnvIPS, qgdnvIVS, qleftIDS, qleftIUS, qleftIPS, qleftIVS, qrightIDS, qrightIUS, qrightIPS, qrightIVS, m_sgnm);
#endif
	}
	if (m_prt)
		qgdnvID.printFormatted("tile qgdnvID riemann");
	if (m_prt)
		qgdnvIU.printFormatted("tile qgdnvIU riemann");
	if (m_prt)
		qgdnvIV.printFormatted("tile qgdnvIV riemann");
	if (m_prt)
		qgdnvIP.printFormatted("tile qgdnvIP riemann");
}

void Tile::boundary_init()
{
	int32_t size, ivar, i, j, i0, j0;

	if (m_scan == X_SCAN) {
		size = pack_arrayv(m_ExtraLayer, m_sendbufld);
		size = pack_arrayv(m_nx, m_sendbufru);
	}			// X_SCAN

	if (m_scan == Y_SCAN) {
		size = pack_arrayh(m_ExtraLayer, m_sendbufld);
		size = pack_arrayh(m_ny, m_sendbufru);
	}			// Y_SCAN
}

void Tile::boundary_process()
{
	int32_t size, ivar, i, j, i0, j0;

	if (m_scan == X_SCAN) {
		if (m_voisin[RIGHT_TILE] != 0) {
			size = unpack_arrayv(m_nx + m_ExtraLayer, m_voisin[RIGHT_TILE]->m_sendbufld);
		}
		if (m_voisin[LEFT_TILE] != 0) {
			size = unpack_arrayv(0, m_voisin[LEFT_TILE]->m_sendbufru);
		}
	}			// X_SCAN

	if (m_scan == Y_SCAN) {
		if (m_voisin[DOWN_TILE] != 0) {
			unpack_arrayh(0, m_voisin[DOWN_TILE]->m_sendbufld);
		}
		if (m_voisin[UP_TILE] != 0) {
			unpack_arrayh(m_ny + m_ExtraLayer, m_voisin[UP_TILE]->m_sendbufru);
		}
	}			// Y_SCAN
	Matrix2 < real_t > &uold = *(*m_uold) (IP_VAR);
	if (m_prt)
		uold.printFormatted("tile uold boundary_process");
}

int32_t Tile::pack_arrayv(int32_t xoffset, real_t * buffer)
{
	int32_t xmin, xmax, ymin, ymax;
	int32_t ivar, i, j, p = 0;
	getExtends(TILE_FULL, xmin, xmax, ymin, ymax);

	for (ivar = 0; ivar < NB_VAR; ivar++) {
		Matrix2 < real_t > &uold = *(*m_uold) (ivar);
		for (j = ymin; j < ymax; j++) {
// #pragma ivdep
			for (i = xoffset; i < xoffset + m_ExtraLayer; i++) {
				buffer[p++] = uold(i + m_offx, j + m_offy);
			}
		}
	}
	return p;
}

int32_t Tile::unpack_arrayv(int32_t xoffset, real_t * buffer)
{
	int32_t xmin, xmax, ymin, ymax;
	int32_t ivar, i, j, p = 0;
	getExtends(TILE_FULL, xmin, xmax, ymin, ymax);

	for (ivar = 0; ivar < NB_VAR; ivar++) {
		Matrix2 < real_t > &uold = *(*m_uold) (ivar);
		for (j = ymin; j < ymax; j++) {
// #pragma ivdep
			for (i = xoffset; i < xoffset + m_ExtraLayer; i++) {
				uold(i + m_offx, j + m_offy) = buffer[p++];
			}
		}
	}
	return p;
}

int32_t Tile::pack_arrayh(int32_t yoffset, real_t * buffer)
{
	int32_t xmin, xmax, ymin, ymax;
	int32_t ivar, i, j, p = 0;
	getExtends(TILE_FULL, xmin, xmax, ymin, ymax);

	for (ivar = 0; ivar < NB_VAR; ivar++) {
		Matrix2 < real_t > &uold = *(*m_uold) (ivar);
		for (j = yoffset; j < yoffset + m_ExtraLayer; j++) {
// #pragma ivdep
			for (i = xmin; i < xmax; i++) {
				buffer[p++] = uold(i + m_offx, j + m_offy);
			}
		}
	}
	return p;
}

int32_t Tile::unpack_arrayh(int32_t yoffset, real_t * buffer)
{
	int32_t xmin, xmax, ymin, ymax;
	int32_t ivar, i, j, p = 0;
	getExtends(TILE_FULL, xmin, xmax, ymin, ymax);

	for (ivar = 0; ivar < NB_VAR; ivar++) {
		Matrix2 < real_t > &uold = *(*m_uold) (ivar);
		for (j = yoffset; j < yoffset + m_ExtraLayer; j++) {
// #pragma ivdep
			for (i = xmin; i < xmax; i++) {
				uold(i + m_offx, j + m_offy) = buffer[p++];
			}
		}
	}
	return p;
}

void Tile::godunov()
{
	Matrix2 < real_t > &uold = *(*m_uold) (IP_VAR);
	Matrix2 < real_t > &uIP = *(*m_u) (IP_VAR);
	Matrix2 < real_t > &qIP = *(*m_q) (IP_VAR);

	if (m_prt)
		cout << "= = = = = = = =  = =" << endl;
	if (m_prt)
		cout << "      Godunov" << endl;
	if (m_prt)
		cout << "= = = = = = = =  = =" << endl;
	if (m_prt)
		cout << endl << " scan " << m_scan << endl;
	if (m_prt)
		cout << endl << " time " << m_tcur << endl;
	if (m_prt)
		cout << endl << " dt " << m_dt << endl;

	constprim();
	eos(TILE_FULL);

	if (m_order > 1) {
		slope();
	}
	trace();
	qleftr();
	riemann();
	compflx();
	if (m_prt)
		uold.printFormatted("Tile uold godunov apres compflx");
}

real_t Tile::computeDt()
{
	real_t dt = 0;
	// a sync on the tiles is required before entering here
	dt = compute_dt();
	return dt;
}

void Tile::setVoisins(Tile * left, Tile * right, Tile * up, Tile * down)
{
	m_voisin[UP_TILE] = up;
	m_voisin[DOWN_TILE] = down;
	m_voisin[LEFT_TILE] = left;
	m_voisin[RIGHT_TILE] = right;
}

void Tile::setBuffers(ThreadBuffers * buf)
{
	assert(buf != 0);
	m_myBuffers = buf;

	m_q = m_myBuffers->getQ();	// NXT, NYT
	m_qxm = m_myBuffers->getQXM();	// NXT, NYT
	m_qxp = m_myBuffers->getQXP();	// NXT, NYT
	m_dq = m_myBuffers->getDQ();	// NXT, NYT
	m_qleft = m_myBuffers->getQLEFT();	// NX + 1, NY + 1
	m_qright = m_myBuffers->getQRIGHT();	// NX + 1, NY + 1
	m_qgdnv = m_myBuffers->getQGDNV();	// NX + 1, NY + 1
	m_c = m_myBuffers->getC();	// NXT, NYT
	m_e = m_myBuffers->getE();	// NXT, NYT

	// work arrays for a single row/column
	m_sgnm = m_myBuffers->getSGNM();
	m_pl = m_myBuffers->getPL();

#if RIEMANNINREGS == 0
	m_pstar = m_myBuffers->getPSTAR();
	m_rl = m_myBuffers->getRL();
	m_ul = m_myBuffers->getUL();
	m_ur = m_myBuffers->getUR();
	m_pr = m_myBuffers->getPR();
	m_cl = m_myBuffers->getCL();
	m_cr = m_myBuffers->getCR();
	m_rr = m_myBuffers->getRR();
	m_goon = m_myBuffers->getGOON();
#endif
}

void Tile::waitVoisin(Tile * voisin, int step)
{
	int okToGO = 0;
	if (voisin == 0)
		return;
	while (!okToGO) {
#ifdef _OPENMP
#pragma omp critical
#endif
		{
			okToGO = voisin->isProcessed(step);
		}
		if (!okToGO) {
			sched_yield();
		}
	}
	return;
}

//EOF
