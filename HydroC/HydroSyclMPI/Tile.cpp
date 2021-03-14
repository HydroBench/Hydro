//
// (C) Guillaume.Colin-de-Verdiere at CEA.Fr
//

#include "Options.hpp"

#include "Tile.hpp"

#include "Tile_Shared_Variables.hpp"

#include "Timers.hpp"
#include "cclock.hpp"

#include <CL/sycl.hpp> // pour sycl::sqrt

#include <cassert>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

//

Tile::Tile() {
    for (int32_t i = 0; i < NEIGHBOUR_TILE; i++) {
        m_voisin[i] = -1;
    }
    m_ExtraLayer = 2;
}

Tile::~Tile() {}

void Tile::infos() {
    *m_cout << " " << (uint64_t)m_u.data() << " " << (uint64_t)m_flux.data() << "\n";
}

// This is on Host, since we allocate the device space here
void Tile::initTile() {
    int32_t xmin, xmax, ymin, ymax;
    int32_t lgx, lgy;

    getExtendsHost(TILE_FULL, xmin, xmax, ymin, ymax);

    lgx = (xmax - xmin);
    lgy = (ymax - ymin);

    // I am on the Host, I can call a global variable !

    m_u = std::move(SoaDevice<real_t>(NB_VAR, lgx, lgy));
    m_flux = std::move(SoaDevice<real_t>(NB_VAR, lgx, lgy));
}

void Tile::swapStorageDims() {
    m_u.swapDimOnly();
    m_flux.swapDimOnly();
}

void Tile::setExtend(int32_t nx, int32_t ny, int32_t gnx, int32_t gny, int32_t offx, int32_t offy,
                     real_t dx) {
    m_nx = nx;
    m_ny = ny;
    m_gnx = gnx;
    m_gny = gny;
    m_offx = offx;
    m_offy = offy;
    m_dx = dx;
}

// Compute part so deviceSharedVariables()
void Tile::slopeOnRow(int32_t xmin, int32_t xmax, Preal_t qS, Preal_t dqS) {
    double ov_slope_type = one / deviceSharedVariables()->m_slope_type;
    // #pragma vector aligned  // impossible !

    for (int32_t i = xmin + 1; i < xmax - 1; i++) {
        real_t dlft, drgt, dcen, dsgn, slop, dlim;
        real_t llftrgt = zero;
        real_t t1;
        dlft = deviceSharedVariables()->m_slope_type * (qS[i] - qS[i - 1]);
        drgt = deviceSharedVariables()->m_slope_type * (qS[i + 1] - qS[i]);
        dcen = my_half * (dlft + drgt) * ov_slope_type;
        dsgn = (dcen > 0) ? one : -one; // sign(one, dcen);

        llftrgt = ((dlft * drgt) <= zero);
        t1 = sycl::min(sycl::abs(dlft), sycl::abs(drgt));
        dqS[i] = dsgn * sycl::min((one - llftrgt) * t1, sycl::abs(dcen));
    }
}

void Tile::slope() {
    int32_t xmin, xmax, ymin, ymax;
    double start, end;

#if 0   
    start = Custom_Timer::dcclock();
#endif

    for (int32_t nbv = 0; nbv < NB_VAR; nbv++) {
        auto q = m_work->getQ()(nbv);
        auto dq = m_work->getDQ()(nbv);

        getExtendsDevice(TILE_FULL, xmin, xmax, ymin, ymax);

        for (int32_t s = ymin; s < ymax; s++) {
            Preal_t qS = q.getRow(s);
            Preal_t dqS = dq.getRow(s);
            slopeOnRow(xmin, xmax, qS, dqS);
        }
    }
    auto q = m_work->getQ()(IP_VAR);
    if (deviceSharedVariables()->m_prt)
        cout() << "Tile q slope" << q;

    auto dq = m_work->getDQ()(IP_VAR);
    if (deviceSharedVariables()->m_prt)
        cout() << "Tile dq slope" << dq;
#if 0
    double elaps = Custom_Timer::dcclock() - start;
    m_threadTimers[myThread()].add(SLOPE, elaps);
#endif
} // slope

void Tile::trace() {
    int32_t xmin, xmax, ymin, ymax;

#if 0   
    double start, end;

    start = Custom_Timer::dcclock();
#endif

    auto qID = m_work->getQ()(ID_VAR);
    auto qIV = m_work->getQ()(IV_VAR);
    auto qIU = m_work->getQ()(IU_VAR);
    auto qIP = m_work->getQ()(IP_VAR);
    auto dqID = m_work->getDQ()(ID_VAR);
    auto dqIV = m_work->getDQ()(IV_VAR);
    auto dqIU = m_work->getDQ()(IU_VAR);
    auto dqIP = m_work->getDQ()(IP_VAR);
    auto pqxmID = m_work->getQXM()(ID_VAR);
    auto pqxmIP = m_work->getQXM()(IP_VAR);
    auto pqxmIV = m_work->getQXM()(IV_VAR);
    auto pqxmIU = m_work->getQXM()(IU_VAR);
    auto pqxpID = m_work->getQXP()(ID_VAR);
    auto pqxpIP = m_work->getQXP()(IP_VAR);
    auto pqxpIV = m_work->getQXP()(IV_VAR);
    auto pqxpIU = m_work->getQXP()(IU_VAR);

    real_t zerol = zero, zeror = zero, project = zero;
    real_t dtdx = m_dt / m_dx;

    if (deviceSharedVariables()->m_scheme == SCHEME_MUSCL) { // MUSCL-Hancock method
        zerol = -hundred / dtdx;
        zeror = hundred / dtdx;
        project = one;
    }
    // if (strcmp(Hscheme, "plmde") == 0) {       // standard PLMDE
    if (deviceSharedVariables()->m_scheme == SCHEME_PLMDE) { // standard PLMDE
        zerol = zero;
        zeror = zero;
        project = one;
    }
    // if (strcmp(Hscheme, "collela") == 0) {     // Collela's method
    if (deviceSharedVariables()->m_scheme == SCHEME_COLLELA) { // Collela's method
        zerol = zero;
        zeror = zero;
        project = zero;
    }

    getExtendsDevice(TILE_FULL, xmin, xmax, ymin, ymax);

    for (int32_t s = ymin; s < ymax; s++) {
        Preal_t cS = m_work->getC().getRow(s);
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
        traceonRow(xmin, xmax, dtdx, zeror, zerol, project, cS, qIDS, qIUS, qIVS, qIPS, dqIDS,
                   dqIUS, dqIVS, dqIPS, pqxpIDS, pqxpIUS, pqxpIVS, pqxpIPS, pqxmIDS, pqxmIUS,
                   pqxmIVS, pqxmIPS);
    }
    if (deviceSharedVariables()->m_prt)
        cout() << "Tile pqxmIP trace" << pqxmIP;

    if (deviceSharedVariables()->m_prt)
        cout() << "Tile pqxpIP Trace" << pqxpIP;
#if 0
    double elaps = Custom_Timer::dcclock() - start;
    m_threadTimers[myThread()].add(TRACE, elaps);
#endif

} // trace

void Tile::traceonRow(int32_t xmin, int32_t xmax, real_t dtdx, real_t zeror, real_t zerol,
                      real_t project, Preal_t cS, Preal_t qIDS, Preal_t qIUS, Preal_t qIVS,
                      Preal_t qIPS, Preal_t dqIDS, Preal_t dqIUS, Preal_t dqIVS, Preal_t dqIPS,
                      Preal_t pqxpIDS, Preal_t pqxpIUS, Preal_t pqxpIVS, Preal_t pqxpIPS,
                      Preal_t pqxmIDS, Preal_t pqxmIUS, Preal_t pqxmIVS, Preal_t pqxmIPS) {

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
        alpham = my_half * (dprcc - du) * rOcc;
        alphap = my_half * (dprcc + du) * rOcc;
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
        apright = -my_half * spplus * alphap;
        amright = -my_half * spminus * alpham;
        azrright = -my_half * spzero * alpha0r;
        azv1right = -my_half * spzero * alpha0v;

        pqxpIDS[i] = r + (apright + amright + azrright);
        pqxpIUS[i] = u + (apright - amright) * OrOcc;
        pqxpIVS[i] = v + (azv1right);
        pqxpIPS[i] = p + (apright + amright) * csq;

        // Left state
        spminus = (umcc <= zerol) ? (-project) : umccx - one;
        spplus = (upcc <= zerol) ? (-project) : upccx - one;
        spzero = (u <= zerol) ? (-project) : ux - one;
        apleft = -my_half * spplus * alphap;
        amleft = -my_half * spminus * alpham;
        azrleft = -my_half * spzero * alpha0r;
        azv1left = -my_half * spzero * alpha0v;

        pqxmIDS[i] = r + (apleft + amleft + azrleft);
        pqxmIUS[i] = u + (apleft - amleft) * OrOcc;
        pqxmIVS[i] = v + (azv1left);
        pqxmIPS[i] = p + (apleft + amleft) * csq;
    }
}

void Tile::qleftrOnRow(int32_t xmin, int32_t xmax, Preal_t pqleftS, Preal_t pqrightS, Preal_t pqxmS,
                       Preal_t pqxpS) {
    // #pragma vector aligned // impossible !

    for (int32_t i = xmin; i < xmax; i++) {
        pqleftS[i] = pqxmS[i + 1];
        pqrightS[i] = pqxpS[i + 2];
    }
}

void Tile::qleftr() {
    int32_t xmin, xmax, ymin, ymax;

#if 0    
    double start, end;
    start = Custom_Timer::dcclock();
#endif

    getExtendsDevice(TILE_FULL, xmin, xmax, ymin, ymax);

    for (int32_t v = 0; v < NB_VAR; v++) {
        auto pqleft = m_work->getQLEFT()(v);
        auto pqright = m_work->getQRIGHT()(v);
        auto pqxm = m_work->getQXM()(v);
        auto pqxp = m_work->getQXP()(v);
        for (int32_t s = ymin; s < ymax; s++) {
            Preal_t pqleftS = pqleft.getRow(s);
            Preal_t pqrightS = pqright.getRow(s);
            Preal_t pqxmS = pqxm.getRow(s);
            Preal_t pqxpS = pqxp.getRow(s);
            qleftrOnRow(xmin, xmax, pqleftS, pqrightS, pqxmS, pqxpS);
        }
    }
    auto pqleft = m_work->getQLEFT()(IP_VAR);
    if (deviceSharedVariables()->m_prt)
        cout() << "Tile qleft qleftr" << pqleft;

    auto pqright = m_work->getQRIGHT()(IP_VAR);
    if (deviceSharedVariables()->m_prt)
        cout() << "Tile qright qleftr" << pqright;
#if 0
    double elaps = Custom_Timer::dcclock() - start;
    m_threadTimers[myThread()].add(QLEFTR, elaps);
#endif
}

void Tile::compflxOnRow(int32_t xmin, int32_t xmax, real_t entho, Preal_t qgdnvIDS,
                        Preal_t qgdnvIUS, Preal_t qgdnvIVS, Preal_t qgdnvIPS, Preal_t fluxIVS,
                        Preal_t fluxIUS, Preal_t fluxIPS, Preal_t fluxIDS) {
    for (int32_t i = xmin; i < xmax; i++) {
        // Mass density
        real_t massDensity = qgdnvIDS[i] * qgdnvIUS[i];
        fluxIDS[i] = massDensity;
        // Normal momentum
        fluxIUS[i] = massDensity * qgdnvIUS[i] + qgdnvIPS[i];
        // Transverse momentum 1
        fluxIVS[i] = massDensity * qgdnvIVS[i];
        // Total energy
        real_t ekin = my_half * qgdnvIDS[i] * (Square(qgdnvIUS[i]) + Square(qgdnvIVS[i]));
        real_t etot = qgdnvIPS[i] * entho + ekin;
        fluxIPS[i] = qgdnvIUS[i] * (etot + qgdnvIPS[i]);
    }
}

void Tile::compflx() {
    int32_t xmin, xmax, ymin, ymax;
#if 0   
    double start, end;
    start = Custom_Timer::dcclock();
#endif

    auto qgdnvID = m_work->getQGDNV()(ID_VAR);
    auto qgdnvIU = m_work->getQGDNV()(IU_VAR);
    auto qgdnvIP = m_work->getQGDNV()(IP_VAR);
    auto qgdnvIV = m_work->getQGDNV()(IV_VAR);

    auto fluxIV = (m_flux)(IV_VAR);
    auto fluxIU = (m_flux)(IU_VAR);
    auto fluxIP = (m_flux)(IP_VAR);
    auto fluxID = (m_flux)(ID_VAR);

    real_t entho = 1.0 / (deviceSharedVariables()->m_gamma - one);

    getExtendsDevice(TILE_FULL, xmin, xmax, ymin, ymax);

    for (int32_t s = ymin; s < ymax; s++) {
        Preal_t qgdnvIDS = qgdnvID.getRow(s);
        Preal_t qgdnvIUS = qgdnvIU.getRow(s);
        Preal_t qgdnvIPS = qgdnvIP.getRow(s);
        Preal_t qgdnvIVS = qgdnvIV.getRow(s);
        Preal_t fluxIVS = fluxIV.getRow(s);
        Preal_t fluxIUS = fluxIU.getRow(s);
        Preal_t fluxIPS = fluxIP.getRow(s);
        Preal_t fluxIDS = fluxID.getRow(s);

        compflxOnRow(xmin, xmax, entho, qgdnvIDS, qgdnvIUS, qgdnvIVS, qgdnvIPS, fluxIVS, fluxIUS,
                     fluxIPS, fluxIDS);
    }
    if (deviceSharedVariables()->m_prt)
        cout() << "Tile fluxIP compflx" << fluxIP;
#if 0
    double elaps = Custom_Timer::dcclock() - start;
    m_threadTimers[myThread()].add(COMPFLX, elaps);
#endif

} // compflx

void Tile::updateconservXscan(int32_t xmin, int32_t xmax, real_t dtdx, Preal_t uIDS, Preal_t uIUS,
                              Preal_t uIVS, Preal_t uIPS, Preal_t uoldIDS, Preal_t uoldIUS,
                              Preal_t uoldIVS, Preal_t uoldIPS, Preal_t fluxIDS, Preal_t fluxIVS,
                              Preal_t fluxIUS, Preal_t fluxIPS) {

#pragma omp simd
    for (int32_t i = xmin; i < xmax; i++) {
        uoldIDS[i + m_offx] = uIDS[i] + (fluxIDS[i - 2] - fluxIDS[i - 1]) * dtdx;
        uoldIVS[i + m_offx] = uIVS[i] + (fluxIVS[i - 2] - fluxIVS[i - 1]) * dtdx;
        uoldIUS[i + m_offx] = uIUS[i] + (fluxIUS[i - 2] - fluxIUS[i - 1]) * dtdx;
        uoldIPS[i + m_offx] = uIPS[i] + (fluxIPS[i - 2] - fluxIPS[i - 1]) * dtdx;
    }
}

void Tile::updateconservYscan(int32_t s, int32_t xmin, int32_t xmax, int32_t ymin, int32_t ymax,
                              real_t dtdx, RArray2D<real_t> &uoldID, RArray2D<real_t> &uoldIP,
                              RArray2D<real_t> &uoldIV, RArray2D<real_t> &uoldIU, Preal_t fluxIVS,
                              Preal_t fluxIUS, Preal_t fluxIPS, Preal_t fluxIDS, Preal_t uIDS,
                              Preal_t uIPS, Preal_t uIVS, Preal_t uIUS, Preal_t pl) {

#pragma omp simd
    for (int32_t j = xmin; j < xmax; j++) {
        pl[j] = uIDS[j] + (fluxIDS[j - 2] - fluxIDS[j - 1]) * dtdx;
    }
    uoldID.putFullCol(s + m_offx, xmin + m_offy, &pl[xmin], (xmax - xmin));

#pragma omp simd
    for (int32_t j = xmin; j < xmax; j++) {
        pl[j] = uIUS[j] + (fluxIUS[j - 2] - fluxIUS[j - 1]) * dtdx;
    }
    uoldIV.putFullCol(s + m_offx, xmin + m_offy, &pl[xmin], (xmax - xmin));

#pragma omp simd
    for (int32_t j = xmin; j < xmax; j++) {
        pl[j] = uIVS[j] + (fluxIVS[j - 2] - fluxIVS[j - 1]) * dtdx;
    }
    uoldIU.putFullCol(s + m_offx, xmin + m_offy, &pl[xmin], (xmax - xmin));

#pragma omp simd
    for (int32_t j = xmin; j < xmax; j++) {
        pl[j] = uIPS[j] + (fluxIPS[j - 2] - fluxIPS[j - 1]) * dtdx;
    }
    uoldIP.putFullCol(s + m_offx, xmin + m_offy, &pl[xmin], (xmax - xmin));
}

void Tile::updateconserv() {

#if 0
    double start, end;
    start = Custom_Timer::dcclock();
#endif

    int32_t xmin, xmax, ymin, ymax;

    auto uoldID = (deviceSharedVariables()->m_uold)(ID_VAR);
    auto uoldIP = (deviceSharedVariables()->m_uold)(IP_VAR);
    auto uoldIV = (deviceSharedVariables()->m_uold)(IV_VAR);
    auto uoldIU = (deviceSharedVariables()->m_uold)(IU_VAR);

    auto fluxIV = (m_flux)(IV_VAR);
    auto fluxIU = (m_flux)(IU_VAR);
    auto fluxIP = (m_flux)(IP_VAR);
    auto fluxID = (m_flux)(ID_VAR);

    auto uID = (m_u)(ID_VAR);
    auto uIP = (m_u)(IP_VAR);
    auto uIV = (m_u)(IV_VAR);
    auto uIU = (m_u)(IU_VAR);
    real_t dtdx = m_dt / m_dx;
    if (deviceSharedVariables()->m_prt)
        cout() << "dtdx " << dtdx << "\n";

    getExtendsDevice(TILE_INTERIOR, xmin, xmax, ymin, ymax);
    if (deviceSharedVariables()->m_prt) {
        cout() << "scan " << (int32_t)deviceSharedVariables()->m_scan << "\n"
               << "Tile uoldIP input updateconserv" << uoldIP << "Tile fluxID input updateconserv"
               << fluxID << "Tile fluxIU input updateconserv" << fluxIU
               << "Tile fluxIV input updateconserv" << fluxIV << "Tile uID updateconserv" << uID
               << "Tile uIU updateconserv" << uIU << "Tile uIV updateconserv" << uIV
               << "Tile uIP updateconserv" << uIP;
    }
    if (deviceSharedVariables()->m_scan == X_SCAN) {
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
            updateconservXscan(xmin, xmax, dtdx, uIDS, uIUS, uIVS, uIPS, uoldIDS, uoldIUS, uoldIVS,
                               uoldIPS, fluxIDS, fluxIVS, fluxIUS, fluxIPS);
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
            Preal_t pl = m_work->getPL();

            updateconservYscan(s, xmin, xmax, ymin, ymax, dtdx, uoldID, uoldIP, uoldIV, uoldIU,
                               fluxIVS, fluxIUS, fluxIPS, fluxIDS, uIDS, uIPS, uIVS, uIUS, pl);
        }
    }
    if (deviceSharedVariables()->m_prt) {
        cout() << "Tile uoldID updateconserv" << uoldID << "Tile uoldIU updateconserv" << uoldIU
               << "Tile uoldIV updateconserv" << uoldIV << "Tile uoldIP updateconserv" << uoldIP;
    }
#if 0
    double elaps = Custom_Timer::dcclock() - start;
    m_threadTimers[myThread()].add(UPDCVAR, elaps);
#endif

} // updateconserv

void Tile::gatherconservXscan(int32_t xmin, int32_t xmax, Preal_t uIDS, Preal_t uIUS, Preal_t uIVS,
                              Preal_t uIPS, Preal_t uoldIDS, Preal_t uoldIUS, Preal_t uoldIVS,
                              Preal_t uoldIPS) {
#if ALIGNED > 0
    // #pragma vector aligned // impossible !
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

void Tile::gatherconservYscan() {}

void Tile::gatherconserv() {
    int32_t xmin, xmax, ymin, ymax;
#if 0   
    double start, end;
    start = Custom_Timer::dcclock();
#endif

    auto uID = (m_u)(ID_VAR);
    auto uIP = (m_u)(IP_VAR);
    auto uIV = (m_u)(IV_VAR);
    auto uIU = (m_u)(IU_VAR);

    auto uoldID = (deviceSharedVariables()->m_uold)(ID_VAR);
    auto uoldIP = (deviceSharedVariables()->m_uold)(IP_VAR);
    auto uoldIV = (deviceSharedVariables()->m_uold)(IV_VAR);
    auto uoldIU = (deviceSharedVariables()->m_uold)(IU_VAR);

    getExtendsDevice(TILE_FULL, xmin, xmax, ymin, ymax);

    if (deviceSharedVariables()->m_prt) {
        cout() << "Tile uoldID gatherconserv" << uoldID << "Tile uoldIU gatherconserv" << uoldIU
               << "Tile uoldIV gatherconserv" << uoldIV << "Tile uoldIP gatherconserv" << uoldIP;
    }

    if (deviceSharedVariables()->m_scan == X_SCAN) {
        for (int32_t s = ymin; s < ymax; s++) {
            real_t *uoldIDS = uoldID.getRow(s + m_offy);
            real_t *uoldIPS = uoldIP.getRow(s + m_offy);
            real_t *uoldIVS = uoldIV.getRow(s + m_offy);
            real_t *uoldIUS = uoldIU.getRow(s + m_offy);
            real_t *uIDS = uID.getRow(s);
            real_t *uIPS = uIP.getRow(s);
            real_t *uIVS = uIV.getRow(s);
            real_t *uIUS = uIU.getRow(s);
            gatherconservXscan(xmin, xmax, uIDS, uIUS, uIVS, uIPS, uoldIDS, uoldIUS, uoldIVS,
                               uoldIPS);
        }
    } else {
        for (int32_t j = xmin; j < xmax; j++) {
            uID.putFullCol(j, 0, uoldID.getRow(j + m_offy) + m_offx, (ymax - ymin));
            uIU.putFullCol(j, 0, uoldIV.getRow(j + m_offy) + m_offx, (ymax - ymin));
            uIV.putFullCol(j, 0, uoldIU.getRow(j + m_offy) + m_offx, (ymax - ymin));
            uIP.putFullCol(j, 0, uoldIP.getRow(j + m_offy) + m_offx, (ymax - ymin));
        }
    }
    if (deviceSharedVariables()->m_prt) {
        cout() << "Tile uID gatherconserv" << uID << "Tile uIU gatherconserv" << uIU
               << "Tile uIV gatherconserv" << uIV << "Tile uIP gatherconserv" << uIP;
    }
#if 0
    double elaps = Custom_Timer::dcclock() - start;
    m_threadTimers[myThread()].add(GATHCVAR, elaps);
#endif

} // gatherconserv

void Tile::eosOnRow(int32_t xmin, int32_t xmax, real_t smallp, Preal_t qIDS, Preal_t eS,
                    Preal_t qIPS, Preal_t cS) {
    if (xmin > 0) {
#pragma omp simd
        for (int32_t k = xmin; k < xmax; k++) {
            real_t rho = qIDS[k];
            real_t rrho = one / rho;
            real_t base = (deviceSharedVariables()->m_gamma - one) * rho * eS[k];
            ;
            base = sycl::max(base, (real_t)(rho * smallp));
            qIPS[k] = base;
            cS[k] = sycl::sqrt(deviceSharedVariables()->m_gamma * base * rrho);
        }
    } else {

#pragma omp simd
        for (int32_t k = xmin; k < xmax; k++) {
            real_t rho = qIDS[k];
            real_t rrho = one / rho;
            real_t base = (deviceSharedVariables()->m_gamma - one) * rho * eS[k];
            ;
            base = sycl::max(base, (real_t)(rho * smallp));
            qIPS[k] = base;
            cS[k] = sycl::sqrt(deviceSharedVariables()->m_gamma * base * rrho);
        }
    }
}

void Tile::eos(tileSpan_t span) {
    int32_t xmin, xmax, ymin, ymax;
#if 0
    double start, end;
    start = Custom_Timer::dcclock();
#endif

    auto qID = m_work->getQ()(ID_VAR);
    auto qIP = m_work->getQ()(IP_VAR);

    real_t smallp =
        Square(deviceSharedVariables()->m_smallc) /
        deviceSharedVariables()->m_gamma; // TODO: We can precompute that to remove some divs

    getExtendsDevice(span, xmin, xmax, ymin, ymax);

    for (int32_t s = ymin; s < ymax; s++) {
        real_t *qIDS = qID.getRow(s);
        real_t *eS = m_work->getE().getRow(s);
        real_t *qIPS = qIP.getRow(s);
        real_t *cS = m_work->getC().getRow(s);
        eosOnRow(xmin, xmax, smallp, qIDS, eS, qIPS, cS);
    }
    if (deviceSharedVariables()->m_prt) {
        cout() << "Tile qIP eos" << qIP << "Tile c eos" << m_work->getC();
    }
#if 0
    double elaps = Custom_Timer::dcclock() - start;
    m_threadTimers[myThread()].add(EOS, elaps);
#endif

} // eos

void Tile::compute_dt_loop1OnRow(int32_t xmin, int32_t xmax, Preal_t qIDS, Preal_t qIPS,
                                 Preal_t qIUS, Preal_t qIVS, Preal_t uoldIDS, Preal_t uoldIUS,
                                 Preal_t uoldIVS, Preal_t uoldIPS, Preal_t eS) {
    for (int32_t i = xmin; i < xmax; i++) {
        real_t eken, tmp;
        qIDS[i] = uoldIDS[i + m_offx];
        qIDS[i] = sycl::max(qIDS[i], deviceSharedVariables()->m_smallr);
        qIUS[i] = uoldIUS[i + m_offx] / qIDS[i];
        qIVS[i] = uoldIVS[i + m_offx] / qIDS[i];
        eken = my_half * (Square(qIUS[i]) + Square(qIVS[i]));
        tmp = uoldIPS[i + m_offx] / qIDS[i] - eken;
        qIPS[i] = tmp;
        eS[i] = tmp;
    }
}

void Tile::compute_dt_loop2OnRow(real_t &tmp1, real_t &tmp2, int32_t xmin, int32_t xmax, Preal_t cS,
                                 Preal_t qIUS, Preal_t qIVS) {
    for (int32_t i = xmin; i < xmax; i++) {
        tmp1 = sycl::max(tmp1, cS[i] + sycl::abs(qIUS[i]));
    }
    for (int32_t i = xmin; i < xmax; i++) {
        tmp2 = sycl::max(tmp2, cS[i] + sycl::abs(qIVS[i]));
    }
}

real_t Tile::compute_dt() {
    int32_t xmin, xmax, ymin, ymax;
#if 0   
    double start, end;
    start = Custom_Timer::dcclock();
#endif
    real_t dt = 0, cournox, cournoy, tmp1 = 0, tmp2 = 0;
    auto uoldID = (deviceSharedVariables()->m_uold)(ID_VAR);
    auto uoldIP = (deviceSharedVariables()->m_uold)(IP_VAR);
    auto uoldIV = (deviceSharedVariables()->m_uold)(IV_VAR);
    auto uoldIU = (deviceSharedVariables()->m_uold)(IU_VAR);

    auto qID = m_work->getQ()(ID_VAR);
    auto qIP = m_work->getQ()(IP_VAR);
    auto qIV = m_work->getQ()(IV_VAR);
    auto qIU = m_work->getQ()(IU_VAR);

    godunovDir_t oldScan = deviceSharedVariables()->m_scan;

    if (deviceSharedVariables()->m_scan == Y_SCAN) {
        deviceSharedVariables()->swapScan();
        swapStorageDims();
    }

    getExtendsDevice(TILE_INTERIOR, xmin, xmax, ymin, ymax);

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
        // real_t *qIPS = qIP.getRow(s);
        real_t *qIVS = qIV.getRow(s);
        real_t *qIUS = qIU.getRow(s);

        real_t *eS = (m_work->getE()).getRow(s);
        compute_dt_loop1OnRow(xmin, xmax, qIDS, qIDS, qIUS, qIVS, uoldIDS, uoldIUS, uoldIVS,
                              uoldIPS, eS);
    }
    // stop timer here to avoid counting EOS twice
#if 0
    double elaps = Custom_Timer::dcclock() - start;
    m_threadTimers[myThread()].add(COMPDT, elaps);
#endif

    eos(TILE_INTERIOR); // needs    qID, e    returns    c, qIP

    // resume timing
#if 0
    start = Custom_Timer::dcclock();
#endif

    for (int32_t s = ymin; s < ymax; s++) {
        real_t *qIVS = qIV.getRow(s);
        real_t *qIUS = qIU.getRow(s);
        real_t *cS = m_work->getC().getRow(s);
        compute_dt_loop2OnRow(tmp1, tmp2, xmin, xmax, cS, qIUS, qIVS);
    }
    cournox = sycl::max(cournox, tmp1);
    cournoy = sycl::max(cournoy, tmp2);

    dt = deviceSharedVariables()->m_cfl * m_dx /
         sycl::max(cournox, sycl::max(cournoy, deviceSharedVariables()->m_smallc));

    if (deviceSharedVariables()->m_scan != oldScan) {
        deviceSharedVariables()->swapScan();
        swapStorageDims();
    }

    if (deviceSharedVariables()->m_prt)
        cout() << "tile dt " << dt << "\n";
#if 0
    elaps = Custom_Timer::dcclock() - start;
    m_threadTimers[myThread()].add(COMPDT, elaps);
#endif

    return dt;
} // compute_dt

void Tile::constprimOnRow(int32_t xmin, int32_t xmax, Preal_t qIDS, Preal_t qIPS, Preal_t qIVS,
                          Preal_t qIUS, const Preal_t uIDS, const Preal_t uIPS, const Preal_t uIVS,
                          const Preal_t uIUS, Preal_t eS) {

#if ALIGNED > 0
    // #pragma message "constprimOnRow aligned"
    // #pragma vector aligned

#endif
    for (int32_t i = xmin; i < xmax; i++) {
        real_t eken, qid, qiu, qiv, qip;
        qid = uIDS[i];
        qid = sycl::max(qid, deviceSharedVariables()->m_smallr);
        // if (qid < m_smallr) qid = m_smallr;
        qiu = uIUS[i] / qid;
        qiv = uIVS[i] / qid;

        eken = my_half * (Square(qiu) + Square(qiv));

        qip = uIPS[i] / qid - eken;
        qIUS[i] = qiu;
        qIVS[i] = qiv;
        qIDS[i] = qid;
        qIPS[i] = qip;
        eS[i] = qip;
    }
}

void Tile::constprim() {
    int32_t xmin, xmax, ymin, ymax;
#if 0   
    double start, end;
    start = Custom_Timer::dcclock();
#endif

    auto qID = m_work->getQ()(ID_VAR);
    auto qIP = m_work->getQ()(IP_VAR);
    auto qIV = m_work->getQ()(IV_VAR);
    auto qIU = m_work->getQ()(IU_VAR);

    auto uID = (m_u)(ID_VAR);
    auto uIP = (m_u)(IP_VAR);
    auto uIV = (m_u)(IV_VAR);
    auto uIU = (m_u)(IU_VAR);

    getExtendsDevice(TILE_FULL, xmin, xmax, ymin, ymax);

    for (int32_t s = ymin; s < ymax; s++) {
        real_t *eS = m_work->getE().getRow(s);
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
    if (deviceSharedVariables()->m_prt) {
        cout() << "Tile qIP constprim" << qIP << "Tile e constprim" << m_work->getE();
    }

#if 0
    double elaps = Custom_Timer::dcclock() - start;
    m_threadTimers[myThread()].add(CONSTPRIM, elaps);
#endif

} // constprim

void Tile::riemannOnRow(int32_t xmin, int32_t xmax, real_t smallp, real_t gamma6, real_t smallpp,
                        Preal_t qgdnvIDS, Preal_t qgdnvIUS, Preal_t qgdnvIPS, Preal_t qgdnvIVS,
                        Preal_t qleftIDS, Preal_t qleftIUS, Preal_t qleftIPS, Preal_t qleftIVS,
                        Preal_t qrightIDS, Preal_t qrightIUS, Preal_t qrightIPS, Preal_t qrightIVS,
                        long *__restrict__ go_on, Preal_t sgnm, Preal_t pstar, Preal_t rl,
                        Preal_t ul, Preal_t pl, Preal_t rr, Preal_t ur, Preal_t pr, Preal_t cl,
                        Preal_t cr) {
// #pragma message "riemannOnRow actif"
#if ALIGNED > 0
    // #pragma vector aligned

#endif
    for (int32_t i = xmin; i < xmax; i++) {
        go_on[i] = 1;
    }

    // Precompute values for this slice
    // #pragma ivdep
#if ALIGNED > 0
    // #pragma vector aligned

#endif

    for (int32_t i = xmin; i < xmax; i++) {
        real_t wl_i, wr_i;
        rl[i] = sycl::max(qleftIDS[i], deviceSharedVariables()->m_smallr);
        ul[i] = qleftIUS[i];
        pl[i] = sycl::max(qleftIPS[i], rl[i] * smallp);
        rr[i] = sycl::max(qrightIDS[i], deviceSharedVariables()->m_smallr);
        ur[i] = qrightIUS[i];
        pr[i] = sycl::max(qrightIPS[i], rr[i] * smallp);

        // Lagrangian sound speed
        cl[i] = deviceSharedVariables()->m_gamma * pl[i] * rl[i];
        cr[i] = deviceSharedVariables()->m_gamma * pr[i] * rr[i];

        // First guess
        wl_i = sycl::sqrt(cl[i]);
        wr_i = sycl::sqrt(cr[i]);
        pstar[i] = sycl::max(
            ((wr_i * pl[i] + wl_i * pr[i]) + wl_i * wr_i * (ul[i] - ur[i])) / (wl_i + wr_i), zero);
    }

    // solve the riemann problem on the interfaces of this slice
    // for (int32_t iter = 0; iter < m_niter_riemann; iter++) {

    // #pragma unroll(5)

    for (int32_t iter = 0; iter < deviceSharedVariables()->m_niter_riemann; iter++) {
#if ALIGNED > 0
        // #pragma vector aligned

#endif
        for (int32_t i = xmin; i < xmax; i++) {
            if (go_on[i] > 0) {
                real_t pst = pstar[i];
                // Newton-Raphson iterations to find pstar at the required accuracy
                real_t wwl = sycl::sqrt(cl[i] * (one + gamma6 * (pst - pl[i]) / pl[i]));
                real_t wwr = sycl::sqrt(cr[i] * (one + gamma6 * (pst - pr[i]) / pr[i]));
                real_t swwl = Square(wwl);
                real_t ql = two * wwl * swwl / (swwl + cl[i]);
                real_t qr = two * wwr * Square(wwr) / (Square(wwr) + cr[i]);
                real_t usl = ul[i] - (pst - pl[i]) / wwl;
                real_t usr = ur[i] + (pst - pr[i]) / wwr;
                real_t tmp = (qr * ql / (qr + ql) * (usl - usr));
                real_t delp_i = sycl::max(tmp, (-pst));
                // pstar[i] = pstar[i] + delp_i;
                pst += delp_i;
                // Convergence indicator
                real_t tmp2 = delp_i / (pst + smallpp);
                real_t uo_i = sycl::abs(tmp2);
                go_on[i] = uo_i > precision;
                // FLOPS(29, 10, 2, 0);
                pstar[i] = pst;
            }
        }
    } // iter_riemann
#if ALIGNED > 0
    // #pragma vector aligned

#endif

    for (int32_t i = xmin; i < xmax; i++) {

        real_t wr_i = sycl::sqrt(cr[i] * (one + gamma6 * (pstar[i] - pr[i]) / pr[i]));
        real_t wl_i = sycl::sqrt(cl[i] * (one + gamma6 * (pstar[i] - pl[i]) / pl[i]));

        real_t ustar_i =
            my_half * (ul[i] + (pl[i] - pstar[i]) / wl_i + ur[i] - (pr[i] - pstar[i]) / wr_i);

        real_t left = ustar_i > 0;

        real_t ro_i, uo_i, po_i, wo_i;

        // if (left) {sgnm[i] = 1;ro_i = rl[i];uo_i = ul[i];po_i = pl[i];wo_i =
        // wl_i; } else {sgnm[i] = -1;ro_i = rr[i];uo_i = ur[i];po_i = pr[i];wo_i =
        // wr_i;}
        sgnm[i] = 1 * left + (-1 + left);
        ro_i = left * rl[i] + (1 - left) * rr[i];
        uo_i = left * ul[i] + (1 - left) * ur[i];
        po_i = left * pl[i] + (1 - left) * pr[i];
        wo_i = left * wl_i + (1 - left) * wr_i;

        real_t co_i = sycl::sqrt(sycl::abs(deviceSharedVariables()->m_gamma * po_i / ro_i));
        co_i = sycl::max(deviceSharedVariables()->m_smallc, co_i);

        real_t rstar_i = ro_i / (one + ro_i * (po_i - pstar[i]) / Square(wo_i));
        rstar_i = sycl::max(rstar_i, deviceSharedVariables()->m_smallr);

        real_t cstar_i =
            sycl::sqrt(sycl::abs(deviceSharedVariables()->m_gamma * pstar[i] / rstar_i));
        cstar_i = sycl::max(deviceSharedVariables()->m_smallc, cstar_i);

        real_t spout_i = co_i - sgnm[i] * uo_i;
        real_t spin_i = cstar_i - sgnm[i] * ustar_i;
        real_t ushock_i = wo_i / ro_i - sgnm[i] * uo_i;

        if (pstar[i] >= po_i) {
            spin_i = ushock_i;
            spout_i = ushock_i;
        }

        real_t scr_i =
            sycl::max((real_t)(spout_i - spin_i),
                      (real_t)(deviceSharedVariables()->m_smallc + sycl::abs(spout_i + spin_i)));

        real_t frac_i = (one + (spout_i + spin_i) / scr_i) * my_half;
        frac_i = sycl::max(zero, (real_t)(sycl::min(one, frac_i)));

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
        // if (left) {qgdnvIVS[i] = qleftIVS[i];} else {qgdnvIVS[i] = qrightIVS[i];}
        qgdnvIVS[i] = left * qleftIVS[i] + (one - left) * qrightIVS[i];
    }
}

void Tile::riemannOnRowInRegs(int32_t xmin, int32_t xmax, real_t smallp, real_t gamma6,
                              real_t smallpp, Preal_t qgdnvIDS, Preal_t qgdnvIUS, Preal_t qgdnvIPS,
                              Preal_t qgdnvIVS, Preal_t qleftIDS, Preal_t qleftIUS,
                              Preal_t qleftIPS, Preal_t qleftIVS, Preal_t qrightIDS,
                              Preal_t qrightIUS, Preal_t qrightIPS, Preal_t qrightIVS,
                              Preal_t sgnm) {
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
        rlI = sycl::max(qleftIDS[i], deviceSharedVariables()->m_smallr);
        ulI = qleftIUS[i];
        plI = sycl::max(qleftIPS[i], rlI * smallp);
        rrI = sycl::max(qrightIDS[i], deviceSharedVariables()->m_smallr);
        urI = qrightIUS[i];
        prI = sycl::max(qrightIPS[i], rrI * smallp);

        // Lagrangian sound speed
        clI = deviceSharedVariables()->m_gamma * plI * rlI;
        crI = deviceSharedVariables()->m_gamma * prI * rrI;

        // First guess
        wl_i = sycl::sqrt(clI);
        wr_i = sycl::sqrt(crI);
        pstarI = sycl::max(((wr_i * plI + wl_i * prI) + wl_i * wr_i * (ulI - urI)) / (wl_i + wr_i),
                           zero);
        //  #pragma ivdep
        for (int32_t iter = 0; iter < 10; iter++) {
            if (goonI > 0) {
                real_t pst = pstarI;
                // Newton-Raphson iterations to find pstar at the required accuracy
                real_t wwl = sycl::sqrt(clI * (one + gamma6 * (pst - plI) / plI));
                real_t wwr = sycl::sqrt(crI * (one + gamma6 * (pst - prI) / prI));
                real_t swwl = Square(wwl);
                real_t ql = two * wwl * swwl / (swwl + clI);
                real_t qr = two * wwr * Square(wwr) / (Square(wwr) + crI);
                real_t usl = ulI - (pst - plI) / wwl;
                real_t usr = urI + (pst - prI) / wwr;
                real_t tmp = (qr * ql / (qr + ql) * (usl - usr));
                real_t delp_i = sycl::max(tmp, (-pst));
                // pstarI = pstarI + delp_i;
                pst += delp_i;
                // Convergence indicator
                real_t tmp2 = delp_i / (pst + smallpp);
                real_t uo_i = sycl::abs(tmp2);
                goonI = uo_i > precision;
                // FLOPS(29, 10, 2, 0);
                pstarI = pst;
            }
        }
        wr_i = sycl::sqrt(crI * (one + gamma6 * (pstarI - prI) / prI));
        wl_i = sycl::sqrt(clI * (one + gamma6 * (pstarI - plI) / plI));

        real_t ustar_i = my_half * (ulI + (plI - pstarI) / wl_i + urI - (prI - pstarI) / wr_i);

        double left = (double)(ustar_i > 0);

        real_t ro_i, uo_i, po_i, wo_i;

        sgnm[i] = one * left + (-one + left);
        ro_i = left * rlI + (one - left) * rrI;
        uo_i = left * ulI + (one - left) * urI;
        po_i = left * plI + (one - left) * prI;
        wo_i = left * wl_i + (one - left) * wr_i;

        real_t co_i = sycl::sqrt(sycl::abs(deviceSharedVariables()->m_gamma * po_i / ro_i));
        co_i = sycl::max(deviceSharedVariables()->m_smallc, co_i);

        real_t rstar_i = ro_i / (one + ro_i * (po_i - pstarI) / Square(wo_i));
        rstar_i = sycl::max(rstar_i, deviceSharedVariables()->m_smallr);

        real_t cstar_i = sycl::sqrt(sycl::abs(deviceSharedVariables()->m_gamma * pstarI / rstar_i));
        cstar_i = sycl::max(deviceSharedVariables()->m_smallc, cstar_i);

        real_t spout_i = co_i - sgnm[i] * uo_i;
        real_t spin_i = cstar_i - sgnm[i] * ustar_i;
        real_t ushock_i = wo_i / ro_i - sgnm[i] * uo_i;

        if (pstarI >= po_i) {
            spin_i = ushock_i;
            spout_i = ushock_i;
        }

        real_t scr_i =
            sycl::max((real_t)(spout_i - spin_i),
                      (real_t)(deviceSharedVariables()->m_smallc + sycl::abs(spout_i + spin_i)));

        real_t frac_i = (one + (spout_i + spin_i) / scr_i) * my_half;
        frac_i = sycl::max(zero, (real_t)(sycl::min(one, frac_i)));

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
        // if (left) {qgdnvIVS[i] = qleftIVS[i];} else {qgdnvIVS[i] = qrightIVS[i];}
        qgdnvIVS[i] = left * qleftIVS[i] + (one - left) * qrightIVS[i];
    }
}

void Tile::riemann() {
    int32_t xmin, xmax, ymin, ymax;
#if 0   
    double start, end;
    start = Custom_Timer::dcclock();
#endif

    auto qgdnvID = m_work->getQGDNV()(ID_VAR);
    auto qgdnvIU = m_work->getQGDNV()(IU_VAR);
    auto qgdnvIP = m_work->getQGDNV()(IP_VAR);
    auto qgdnvIV = m_work->getQGDNV()(IV_VAR);

    auto qleftID = m_work->getQLEFT()(ID_VAR);
    auto qleftIU = m_work->getQLEFT()(IU_VAR);
    auto qleftIP = m_work->getQLEFT()(IP_VAR);
    auto qleftIV = m_work->getQLEFT()(IV_VAR);

    auto qrightID = m_work->getQRIGHT()(ID_VAR);
    auto qrightIU = m_work->getQRIGHT()(IU_VAR);
    auto qrightIP = m_work->getQRIGHT()(IP_VAR);
    auto qrightIV = m_work->getQRIGHT()(IV_VAR);

    real_t smallp = Square(deviceSharedVariables()->m_smallc) / deviceSharedVariables()->m_gamma;
    real_t gamma6 =
        (deviceSharedVariables()->m_gamma + one) / (two * deviceSharedVariables()->m_gamma);
    real_t smallpp = deviceSharedVariables()->m_smallr * smallp;

    getExtendsDevice(TILE_FULL, xmin, xmax, ymin, ymax);

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

        riemannOnRowInRegs(xmin, xmax, smallp, gamma6, smallpp, qgdnvIDS, qgdnvIUS, qgdnvIPS,
                           qgdnvIVS, qleftIDS, qleftIUS, qleftIPS, qleftIVS, qrightIDS, qrightIUS,
                           qrightIPS, qrightIVS, m_work->getSGNM());
    }

    if (deviceSharedVariables()->m_prt) {
        cout() << "tile qgdnvID riemann" << qgdnvID << "tile qgdnvIU riemann" << qgdnvIU
               << "tile qgfnvIV riemann" << qgdnvIV << "tile qgdnvIP riemann" << qgdnvIP;
    }

#if 0
    double elaps = Custom_Timer::dcclock() - start;
    m_threadTimers[myThread()].add(RIEMANN, elaps);
#endif

} // riemann

void Tile::godunov() {
    auto uold = (deviceSharedVariables()->m_uold)(IP_VAR);
    auto uIP = (m_u)(IP_VAR);
    auto qIP = m_work->getQ()(IP_VAR);

    if (deviceSharedVariables()->m_prt)
        cout() << "= = = = = = = =  = ="
               << "\n";
    if (deviceSharedVariables()->m_prt)
        cout() << "      Godunov"
               << "\n";
    if (deviceSharedVariables()->m_prt)
        cout() << "= = = = = = = =  = ="
               << "\n";
    if (deviceSharedVariables()->m_prt)
        cout() << "\n"
               << " scan " << (int32_t)deviceSharedVariables()->m_scan << "\n";
    if (deviceSharedVariables()->m_prt)
        cout() << "\n"
               << " time " << m_tcur << "\n";
    if (deviceSharedVariables()->m_prt)
        cout() << "\n"
               << " dt " << m_dt << "\n";

    constprim();
    eos(TILE_FULL);

    if (deviceSharedVariables()->m_order > 1) {
        slope();
    }

    qleftr();

    riemann();

    compflx();
    if (deviceSharedVariables()->m_prt)
        cout() << "Tile uold godunov apres compflx" << uold;
}

real_t Tile::computeDt() {
    real_t dt = 0;
    // a sync on the tiles is required before entering here
    dt = compute_dt();
    return dt;
}

void Tile::setVoisins(int32_t left, int32_t right, int32_t up, int32_t down) {
    m_voisin[UP_TILE] = up;
    m_voisin[DOWN_TILE] = down;
    m_voisin[LEFT_TILE] = left;
    m_voisin[RIGHT_TILE] = right;
}

SYCL_EXTERNAL
void Tile::setBuffers(DeviceBuffers *buf) { m_work = buf; }

#if 0
long Tile::getLengthByte() { return m_u->getLengthByte() + m_flux->getLengthByte(); }

void Tile::read(const int f) {
    m_u->read(f);
    m_flux->read(f);
}

void Tile::write(const int f) {
    m_u->write(f);
    m_flux->write(f);
}
#endif

// EOF
