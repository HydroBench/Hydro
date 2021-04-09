//
// (C) Guillaume.Colin-de-Verdiere at CEA.Fr
//

#include "Tile.hpp"
#include "Utilities.hpp" // for square

#include <CL/sycl.hpp>
#include <algorithm>

//

Tile::Tile() {
    m_scan = X_SCAN;
    m_ExtraLayer = 2;
}

Tile::~Tile() {}

void Tile::initCandE() {
    int32_t lgr = m_nx * m_ny;
    auto pc = m_c.data();
    auto pe = m_e.data();
    for (int32_t i = 0; i < lgr; i++) {
        pc[i] = 0.0;
        pe[i] = 0.0;
    }
}

void Tile::infos() {}

// This is on Host, since we allocate the device space here
void Tile::initTile() {
    int32_t xmin, xmax, ymin, ymax;
    int32_t lgx, lgy;

    m_scan = X_SCAN;

    getExtends(TILE_FULL, xmin, xmax, ymin, ymax);

    lgx = (xmax - xmin);
    lgy = (ymax - ymin);

    // I am on the Host, I can call a global variable !

    m_u = SoaDevice<real_t>(NB_VAR, lgx, lgy);
    m_flux = SoaDevice<real_t>(NB_VAR, lgx, lgy);

    m_q = SoaDevice<real_t>(NB_VAR, lgx, lgy);
    m_qxm = SoaDevice<real_t>(NB_VAR, lgx, lgy);
    m_qxp = SoaDevice<real_t>(NB_VAR, lgx, lgy);
    m_dq = SoaDevice<real_t>(NB_VAR, lgx, lgy);
    m_qleft = SoaDevice<real_t>(NB_VAR, lgx, lgy);
    m_qright = SoaDevice<real_t>(NB_VAR, lgx, lgy);
    m_qgdnv = SoaDevice<real_t>(NB_VAR, lgx, lgy);

    m_c = Array2D<real_t>(lgx, lgy);
    m_e = Array2D<real_t>(lgx, lgy);

    int32_t lgmax = sycl::max(lgx, lgy);
    m_sgnm = Array1D<real_t>(lgmax);
    m_pl = Array1D<real_t>(lgmax);

    m_swapped = false;
}

void Tile::swapStorageDims() {
    m_u.swapDimOnly();
    m_flux.swapDimOnly();

    m_q.swapDimOnly();
    m_qxm.swapDimOnly();
    m_qxp.swapDimOnly();
    m_dq.swapDimOnly();
    m_qleft.swapDimOnly();
    m_qright.swapDimOnly();
    m_qgdnv.swapDimOnly();

    m_c.swapDimOnly();
    m_e.swapDimOnly();
    m_swapped = !m_swapped;
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
void Tile::slopeOnRow(int32_t xmin, int32_t xmax, Preal_t qS, Preal_t dqS, real_t ov_slope_type) {

    for (int32_t i = xmin + 1; i < xmax - 1; i++) {
        real_t dlft, drgt, dcen, dsgn;
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
    double ov_slope_type = one / deviceSharedVariables()->m_slope_type;

    for (int32_t nbv = 0; nbv < NB_VAR; nbv++) {
        auto q = getQ()(nbv);
        auto dq = getDQ()(nbv);

        getExtends(TILE_FULL, xmin, xmax, ymin, ymax);

        for (int32_t s = ymin; s < ymax; s++) {
            Preal_t qS = q.getRow(s);
            Preal_t dqS = dq.getRow(s);
            slopeOnRow(xmin, xmax, qS, dqS, ov_slope_type);
        }
    }

} // slope

void Tile::slope(int32_t row, real_t ov_slope_type) {
    int32_t xmin, xmax, ymin, ymax;
    getExtends(TILE_FULL, xmin, xmax, ymin, ymax);

    if (row >= ymin && row < ymax) {
        for (int nbv = 0; nbv < NB_VAR; nbv++) {
            auto qS = getQ()(nbv).getRow(row);
            auto dqS = getDQ()(nbv).getRow(row);
            slopeOnRow(xmin, xmax, qS, dqS, ov_slope_type);
        }
    }
}

void Tile::trace(int32_t row, real_t zerol, real_t zeror, real_t project, real_t dtdx) {
    int32_t xmin, xmax, ymin, ymax;
    getExtends(TILE_FULL, xmin, xmax, ymin, ymax);

    if (row >= ymin && row < ymax) {
        auto qIDS = getQ()(ID_VAR).getRow(row);
        auto qIVS = getQ()(IV_VAR).getRow(row);
        auto qIUS = getQ()(IU_VAR).getRow(row);
        auto qIPS = getQ()(IP_VAR).getRow(row);

        auto dqIDS = getDQ()(ID_VAR).getRow(row);
        auto dqIVS = getDQ()(IV_VAR).getRow(row);
        auto dqIUS = getDQ()(IU_VAR).getRow(row);
        auto dqIPS = getDQ()(IP_VAR).getRow(row);

        auto pqxmIDS = getQXM()(ID_VAR).getRow(row);
        auto pqxmIPS = getQXM()(IP_VAR).getRow(row);
        auto pqxmIVS = getQXM()(IV_VAR).getRow(row);
        auto pqxmIUS = getQXM()(IU_VAR).getRow(row);

        auto pqxpIDS = getQXP()(ID_VAR).getRow(row);
        auto pqxpIPS = getQXP()(IP_VAR).getRow(row);
        auto pqxpIVS = getQXP()(IV_VAR).getRow(row);
        auto pqxpIUS = getQXP()(IU_VAR).getRow(row);
        auto cS = getC().getRow(row);

        traceonRow(xmin, xmax, dtdx, zeror, zerol, project, cS, qIDS, qIUS, qIVS, qIPS, dqIDS,
                   dqIUS, dqIVS, dqIPS, pqxpIDS, pqxpIUS, pqxpIVS, pqxpIPS, pqxmIDS, pqxmIUS,
                   pqxmIVS, pqxmIPS);
    }
}
void Tile::trace() {
    int32_t xmin, xmax, ymin, ymax;
    auto qID = getQ()(ID_VAR);
    auto qIV = getQ()(IV_VAR);
    auto qIU = getQ()(IU_VAR);
    auto qIP = getQ()(IP_VAR);

    auto dqID = getDQ()(ID_VAR);
    auto dqIV = getDQ()(IV_VAR);
    auto dqIU = getDQ()(IU_VAR);
    auto dqIP = getDQ()(IP_VAR);

    auto pqxmID = getQXM()(ID_VAR);
    auto pqxmIP = getQXM()(IP_VAR);
    auto pqxmIV = getQXM()(IV_VAR);
    auto pqxmIU = getQXM()(IU_VAR);

    auto pqxpID = getQXP()(ID_VAR);
    auto pqxpIP = getQXP()(IP_VAR);
    auto pqxpIV = getQXP()(IV_VAR);
    auto pqxpIU = getQXP()(IU_VAR);

    real_t zerol = zero, zeror = zero, project = zero;
    real_t dtdx = m_dt / m_dx;

    if (deviceSharedVariables()->m_scheme == SCHEME_MUSCL) { // MUSCL-Hancock method
        zerol = -hundred / dtdx;
        zeror = hundred / dtdx;
        project = one;
    }

    if (deviceSharedVariables()->m_scheme == SCHEME_PLMDE) { // standard PLMDE
        zerol = zero;
        zeror = zero;
        project = one;
    }

    if (deviceSharedVariables()->m_scheme == SCHEME_COLLELA) { // Collela's method
        zerol = zero;
        zeror = zero;
        project = zero;
    }

    getExtends(TILE_FULL, xmin, xmax, ymin, ymax);

    for (int32_t s = ymin; s < ymax; s++) {
        Preal_t cS = getC().getRow(s);
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

    for (int32_t i = xmin; i < xmax; i++) {
        pqleftS[i] = pqxmS[i + 1];
        pqrightS[i] = pqxpS[i + 2];
    }
}

void Tile::qleftr() {
    int32_t xmin, xmax, ymin, ymax;

    getExtends(TILE_FULL, xmin, xmax, ymin, ymax);

    for (int32_t v = 0; v < NB_VAR; v++) {
        auto pqleft = getQLEFT()(v);
        auto pqright = getQRIGHT()(v);
        auto pqxm = getQXM()(v);
        auto pqxp = getQXP()(v);
        for (int32_t s = ymin; s < ymax; s++) {
            Preal_t pqleftS = pqleft.getRow(s);
            Preal_t pqrightS = pqright.getRow(s);
            Preal_t pqxmS = pqxm.getRow(s);
            Preal_t pqxpS = pqxp.getRow(s);
            qleftrOnRow(xmin, xmax, pqleftS, pqrightS, pqxmS, pqxpS);
        }
    }
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

    auto qgdnvID = getQGDNV()(ID_VAR);
    auto qgdnvIU = getQGDNV()(IU_VAR);
    auto qgdnvIP = getQGDNV()(IP_VAR);
    auto qgdnvIV = getQGDNV()(IV_VAR);

    auto fluxIV = (m_flux)(IV_VAR);
    auto fluxIU = (m_flux)(IU_VAR);
    auto fluxIP = (m_flux)(IP_VAR);
    auto fluxID = (m_flux)(ID_VAR);

    real_t entho = 1.0 / (deviceSharedVariables()->m_gamma - one);

    getExtends(TILE_FULL, xmin, xmax, ymin, ymax);

    for (int32_t y = ymin; y < ymax; y++) {
        Preal_t qgdnvIDS = qgdnvID.getRow(y);
        Preal_t qgdnvIUS = qgdnvIU.getRow(y);
        Preal_t qgdnvIPS = qgdnvIP.getRow(y);
        Preal_t qgdnvIVS = qgdnvIV.getRow(y);

        Preal_t fluxIVS = fluxIV.getRow(y);
        Preal_t fluxIUS = fluxIU.getRow(y);
        Preal_t fluxIPS = fluxIP.getRow(y);
        Preal_t fluxIDS = fluxID.getRow(y);

        compflxOnRow(xmin, xmax, entho, qgdnvIDS, qgdnvIUS, qgdnvIVS, qgdnvIPS, fluxIVS, fluxIUS,
                     fluxIPS, fluxIDS);
    }

} // compflx

void Tile::updateconservXscan(int32_t xmin, int32_t xmax, real_t dtdx, Preal_t uIDS, Preal_t uIUS,
                              Preal_t uIVS, Preal_t uIPS, Preal_t uoldIDS, Preal_t uoldIUS,
                              Preal_t uoldIVS, Preal_t uoldIPS, Preal_t fluxIDS, Preal_t fluxIVS,
                              Preal_t fluxIUS, Preal_t fluxIPS) {

    for (int32_t i = xmin; i < xmax; i++) {
        uoldIDS[i + m_offx] = uIDS[i] + (fluxIDS[i - 2] - fluxIDS[i - 1]) * dtdx;
        uoldIVS[i + m_offx] = uIVS[i] + (fluxIVS[i - 2] - fluxIVS[i - 1]) * dtdx;
        uoldIUS[i + m_offx] = uIUS[i] + (fluxIUS[i - 2] - fluxIUS[i - 1]) * dtdx;
        uoldIPS[i + m_offx] = uIPS[i] + (fluxIPS[i - 2] - fluxIPS[i - 1]) * dtdx;
    }
}

void Tile::updateconservYscan(int32_t y, int32_t xmin, int32_t xmax, int32_t ymin, int32_t ymax,
                              real_t dtdx, RArray2D<real_t> &uoldID, RArray2D<real_t> &uoldIP,
                              RArray2D<real_t> &uoldIV, RArray2D<real_t> &uoldIU, Preal_t fluxIVS,
                              Preal_t fluxIUS, Preal_t fluxIPS, Preal_t fluxIDS, Preal_t uIDS,
                              Preal_t uIPS, Preal_t uIVS, Preal_t uIUS, Preal_t pl) {

#pragma omp simd
    for (int32_t x = xmin; x < xmax; x++) {
        pl[x] = uIDS[x] + (fluxIDS[x - 2] - fluxIDS[x - 1]) * dtdx;
    }
    uoldID.putFullCol(y + m_offx, xmin + m_offy, &pl[xmin], (xmax - xmin));

#pragma omp simd
    for (int32_t x = xmin; x < xmax; x++) {
        pl[x] = uIUS[x] + (fluxIUS[x - 2] - fluxIUS[x - 1]) * dtdx;
    }
    uoldIV.putFullCol(y + m_offx, xmin + m_offy, &pl[xmin], (xmax - xmin));

#pragma omp simd
    for (int32_t x = xmin; x < xmax; x++) {
        pl[x] = uIVS[x] + (fluxIVS[x - 2] - fluxIVS[x - 1]) * dtdx;
    }
    uoldIU.putFullCol(y + m_offx, xmin + m_offy, &pl[xmin], (xmax - xmin));

#pragma omp simd
    for (int32_t x = xmin; x < xmax; x++) {
        pl[x] = uIPS[x] + (fluxIPS[x - 2] - fluxIPS[x - 1]) * dtdx;
    }
    uoldIP.putFullCol(y + m_offx, xmin + m_offy, &pl[xmin], (xmax - xmin));
}

void Tile::updateconserv() {

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

    getExtends(TILE_INTERIOR, xmin, xmax, ymin, ymax);

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
            Preal_t pl = getPL();

            updateconservYscan(s, xmin, xmax, ymin, ymax, dtdx, uoldID, uoldIP, uoldIV, uoldIU,
                               fluxIVS, fluxIUS, fluxIPS, fluxIDS, uIDS, uIPS, uIVS, uIUS, pl);
        }
    }

} // updateconserv

void Tile::gatherconservXscan(int32_t xmin, int32_t xmax, Preal_t uIDS, Preal_t uIUS, Preal_t uIVS,
                              Preal_t uIPS, Preal_t uoldIDS, Preal_t uoldIUS, Preal_t uoldIVS,
                              Preal_t uoldIPS) {

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

    auto uID = (m_u)(ID_VAR);
    auto uIP = (m_u)(IP_VAR);
    auto uIV = (m_u)(IV_VAR);
    auto uIU = (m_u)(IU_VAR);

    auto uoldID = (deviceSharedVariables()->m_uold)(ID_VAR);
    auto uoldIP = (deviceSharedVariables()->m_uold)(IP_VAR);
    auto uoldIV = (deviceSharedVariables()->m_uold)(IV_VAR);
    auto uoldIU = (deviceSharedVariables()->m_uold)(IU_VAR);

    getExtends(TILE_FULL, xmin, xmax, ymin, ymax);

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

} // gatherconserv

void Tile::eosOnRow(int32_t xmin, int32_t xmax, real_t smallp, Preal_t qIDS, Preal_t eS,
                    Preal_t qIPS, Preal_t cS) {

    const real_t gamma = deviceSharedVariables()->m_gamma;

    for (int32_t k = xmin; k < xmax; k++) {
        real_t rho = qIDS[k];
        real_t rrho = one / rho;
        real_t base = (gamma - one) * rho * eS[k];

        base = sycl::max(base, (real_t)(rho * smallp));
        qIPS[k] = base;
        cS[k] = sycl::sqrt(gamma * base * rrho);
    }
}

void Tile::eos(tileSpan_t span) {
    int32_t xmin, xmax, ymin, ymax;

    auto qID = getQ()(ID_VAR);
    auto qIP = getQ()(IP_VAR);

    real_t smallp =
        Square(deviceSharedVariables()->m_smallc) /
        deviceSharedVariables()->m_gamma; // TODO: We can precompute that to remove some divs

    getExtends(span, xmin, xmax, ymin, ymax);

    for (int32_t s = ymin; s < ymax; s++) {
        real_t *qIDS = qID.getRow(s);
        real_t *eS = getE().getRow(s);
        real_t *qIPS = qIP.getRow(s);
        real_t *cS = getC().getRow(s);
        eosOnRow(xmin, xmax, smallp, qIDS, eS, qIPS, cS);
    }

} // eos

void Tile::compute_dt_loop1OnRow(int32_t xmin, int32_t xmax, Preal_t qIDS, Preal_t qIPS,
                                 Preal_t qIUS, Preal_t qIVS, Preal_t uoldIDS, Preal_t uoldIUS,
                                 Preal_t uoldIVS, Preal_t uoldIPS, Preal_t eS) {
    real_t smallr = deviceSharedVariables()->m_smallr;
    for (int32_t i = xmin; i < xmax; i++) {
        real_t eken, tmp;
        qIDS[i] = uoldIDS[i + m_offx];
        qIDS[i] = sycl::max(qIDS[i], smallr);
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

    real_t dt = 0, cournox, cournoy, tmp1 = zero, tmp2 = zero;
    auto uoldID = (deviceSharedVariables()->m_uold)(ID_VAR);
    auto uoldIP = (deviceSharedVariables()->m_uold)(IP_VAR);
    auto uoldIV = (deviceSharedVariables()->m_uold)(IV_VAR);
    auto uoldIU = (deviceSharedVariables()->m_uold)(IU_VAR);

    auto qID = getQ()(ID_VAR);
    auto qIP = getQ()(IP_VAR);
    auto qIV = getQ()(IV_VAR);
    auto qIU = getQ()(IU_VAR);

    godunovDir_t oldScan = m_scan;

    if (oldScan == Y_SCAN) {
        swapScan();
        swapStorageDims();
    }

    getExtends(TILE_INTERIOR, xmin, xmax, ymin, ymax);

    cournox = zero;
    cournoy = zero;

    for (int32_t y = ymin; y < ymax; y++) {
        auto uoldIDS = uoldID.getRow(y + m_offy);
        auto uoldIPS = uoldIP.getRow(y + m_offy);
        auto uoldIVS = uoldIV.getRow(y + m_offy);
        auto uoldIUS = uoldIU.getRow(y + m_offy);

        auto *qIDS = qID.getRow(y);
        auto *qIVS = qIV.getRow(y);
        auto *qIUS = qIU.getRow(y);

        auto *eS = (getE()).getRow(y);
        compute_dt_loop1OnRow(xmin, xmax, qIDS, qIDS, qIUS, qIVS, uoldIDS, uoldIUS, uoldIVS,
                              uoldIPS, eS);
    }
    // stop timer here to avoid counting EOS twice

    eos(TILE_INTERIOR); // needs    qID, e    returns    c, qIP

    // resume timing

    for (int32_t s = ymin; s < ymax; s++) {
        real_t *qIVS = qIV.getRow(s);
        real_t *qIUS = qIU.getRow(s);
        real_t *cS = getC().getRow(s);
        compute_dt_loop2OnRow(tmp1, tmp2, xmin, xmax, cS, qIUS, qIVS);
    }
    cournox = sycl::max(cournox, tmp1);
    cournoy = sycl::max(cournoy, tmp2);

    dt = deviceSharedVariables()->m_cfl * m_dx /
         sycl::max(cournox, sycl::max(cournoy, deviceSharedVariables()->m_smallc));

    if (oldScan == Y_SCAN) {
        swapScan();
        swapStorageDims();
    }

    return dt;
} // compute_dt

void Tile::constprimOnRow(int32_t xmin, int32_t xmax, Preal_t qIDS, Preal_t qIPS, Preal_t qIVS,
                          Preal_t qIUS, const Preal_t uIDS, const Preal_t uIPS, const Preal_t uIVS,
                          const Preal_t uIUS, Preal_t eS) {

    for (int32_t x = xmin; x < xmax; x++) {
        real_t eken, qid, qiu, qiv, qip;
        qid = uIDS[x];
        qid = sycl::max(qid, deviceSharedVariables()->m_smallr);
        // if (qid < m_smallr) qid = m_smallr;
        qiu = uIUS[x] / qid;
        qiv = uIVS[x] / qid;

        eken = my_half * (Square(qiu) + Square(qiv));

        qip = uIPS[x] / qid - eken;
        qIUS[x] = qiu;
        qIVS[x] = qiv;
        qIDS[x] = qid;
        qIPS[x] = qip;
        eS[x] = qip;
    }
}

void Tile::constprim(int32_t row) {

    int32_t xmin, xmax, ymin, ymax;
    getExtends(TILE_FULL, xmin, xmax, ymin, ymax);
    if (row >= xmin && row < ymax) {

        auto qIDS = getQ()(ID_VAR).getRow(row);
        auto qIPS = getQ()(IP_VAR).getRow(row);
        auto qIVS = getQ()(IV_VAR).getRow(row);
        auto qIUS = getQ()(IU_VAR).getRow(row);

        auto uIDS = (m_u)(ID_VAR).getRow(row);
        auto uIPS = (m_u)(IP_VAR).getRow(row);
        auto uIVS = (m_u)(IV_VAR).getRow(row);
        auto uIUS = (m_u)(IU_VAR).getRow(row);

        real_t *eS = getE().getRow(row);

        constprimOnRow(xmin, xmax, qIDS, qIPS, qIVS, qIUS, uIDS, uIPS, uIVS, uIUS, eS);
    }

} // constprim

void Tile::constprim() {
    int32_t xmin, xmax, ymin, ymax;

    auto qID = getQ()(ID_VAR);
    auto qIP = getQ()(IP_VAR);
    auto qIV = getQ()(IV_VAR);
    auto qIU = getQ()(IU_VAR);

    auto uID = (m_u)(ID_VAR);
    auto uIP = (m_u)(IP_VAR);
    auto uIV = (m_u)(IV_VAR);
    auto uIU = (m_u)(IU_VAR);

    getExtends(TILE_FULL, xmin, xmax, ymin, ymax);

    for (int32_t s = ymin; s < ymax; s++) {
        real_t *eS = getE().getRow(s);

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

} // constprim

void Tile::riemannOnRowInRegs(int32_t xmin, int32_t xmax, real_t smallp, real_t gamma6,
                              real_t smallpp, Preal_t qgdnvIDS, Preal_t qgdnvIUS, Preal_t qgdnvIPS,
                              Preal_t qgdnvIVS, Preal_t qleftIDS, Preal_t qleftIUS,
                              Preal_t qleftIPS, Preal_t qleftIVS, Preal_t qrightIDS,
                              Preal_t qrightIUS, Preal_t qrightIPS, Preal_t qrightIVS,
                              Preal_t sgnm) {
    real_t smallr = deviceSharedVariables()->m_smallr;
    real_t gamma = deviceSharedVariables()->m_gamma;
    real_t smallc = deviceSharedVariables()->m_smallc;

#pragma omp simd
    for (int32_t i = xmin; i < xmax; i++) {
        bool go_onI = true;
        real_t pstarI;
        real_t rlI;
        real_t ulI;
        real_t plI;
        real_t rrI;
        real_t urI;
        real_t prI;
        real_t clI;
        real_t crI;

        // Precompute values for this slice

        rlI = sycl::max(qleftIDS[i], smallr);
        ulI = qleftIUS[i];
        plI = sycl::max(qleftIPS[i], rlI * smallp);
        rrI = sycl::max(qrightIDS[i], smallr);
        urI = qrightIUS[i];
        prI = sycl::max(qrightIPS[i], rrI * smallp);

        // Lagrangian sound speed
        clI = gamma * plI * rlI;
        crI = gamma * prI * rrI;

        // First guess
        real_t wl_i = sycl::sqrt(clI);
        real_t wr_i = sycl::sqrt(crI);
        pstarI = sycl::max(((wr_i * plI + wl_i * prI) + wl_i * wr_i * (ulI - urI)) / (wl_i + wr_i),
                           zero);
        //  #pragma ivdep
        for (int32_t iter = 0; iter < 10; iter++) {
            if (go_onI) {
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
                go_onI = uo_i > precision;
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

        real_t co_i = sycl::sqrt(sycl::abs(gamma * po_i / ro_i));
        co_i = sycl::max(smallc, co_i);

        real_t rstar_i = ro_i / (one + ro_i * (po_i - pstarI) / Square(wo_i));
        rstar_i = sycl::max(rstar_i, smallr);

        real_t cstar_i = sycl::sqrt(sycl::abs(gamma * pstarI / rstar_i));
        cstar_i = sycl::max(smallc, cstar_i);

        real_t spout_i = co_i - sgnm[i] * uo_i;
        real_t spin_i = cstar_i - sgnm[i] * ustar_i;
        real_t ushock_i = wo_i / ro_i - sgnm[i] * uo_i;

        if (pstarI >= po_i) {
            spin_i = ushock_i;
            spout_i = ushock_i;
        }

        real_t scr_i =
            sycl::max((real_t)(spout_i - spin_i), (real_t)(smallc + sycl::abs(spout_i + spin_i)));

        real_t frac_i = (one + (spout_i + spin_i) / scr_i) * my_half;
        frac_i = sycl::max(zero, (real_t)(sycl::min(one, frac_i)));

        int addSpout = spout_i < zero;
        int addSpin = spin_i > zero;

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

    auto qgdnvID = getQGDNV()(ID_VAR);
    auto qgdnvIU = getQGDNV()(IU_VAR);
    auto qgdnvIP = getQGDNV()(IP_VAR);
    auto qgdnvIV = getQGDNV()(IV_VAR);

    auto qleftID = getQLEFT()(ID_VAR);
    auto qleftIU = getQLEFT()(IU_VAR);
    auto qleftIP = getQLEFT()(IP_VAR);
    auto qleftIV = getQLEFT()(IV_VAR);

    auto qrightID = getQRIGHT()(ID_VAR);
    auto qrightIU = getQRIGHT()(IU_VAR);
    auto qrightIP = getQRIGHT()(IP_VAR);
    auto qrightIV = getQRIGHT()(IV_VAR);

    real_t smallp = Square(deviceSharedVariables()->m_smallc) / deviceSharedVariables()->m_gamma;
    real_t gamma6 =
        (deviceSharedVariables()->m_gamma + one) / (two * deviceSharedVariables()->m_gamma);
    real_t smallpp = deviceSharedVariables()->m_smallr * smallp;

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

        riemannOnRowInRegs(xmin, xmax, smallp, gamma6, smallpp, qgdnvIDS, qgdnvIUS, qgdnvIPS,
                           qgdnvIVS, qleftIDS, qleftIUS, qleftIPS, qleftIVS, qrightIDS, qrightIUS,
                           qrightIPS, qrightIVS, getSGNM());
    }

} // riemann

void Tile::riemann(int32_t row, real_t smallp, real_t gamma6, real_t smallpp) {
    int32_t xmin, xmax, ymin, ymax;
    getExtends(TILE_FULL, xmin, xmax, ymin, ymax);

    if (row >= ymin && row < ymax) {
        auto qgdnvIPS = getQGDNV()(IP_VAR).getRow(row);
        auto qgdnvIDS = getQGDNV()(ID_VAR).getRow(row);
        auto qgdnvIUS = getQGDNV()(IU_VAR).getRow(row);
        auto qgdnvIVS = getQGDNV()(IV_VAR).getRow(row);

        auto qleftIPS = getQLEFT()(IP_VAR).getRow(row);
        auto qleftIDS = getQLEFT()(ID_VAR).getRow(row);
        auto qleftIUS = getQLEFT()(IU_VAR).getRow(row);
        auto qleftIVS = getQLEFT()(IV_VAR).getRow(row);

        auto qrightIPS = getQRIGHT()(IP_VAR).getRow(row);
        auto qrightIDS = getQRIGHT()(ID_VAR).getRow(row);
        auto qrightIUS = getQRIGHT()(IU_VAR).getRow(row);
        auto qrightIVS = getQRIGHT()(IV_VAR).getRow(row);

        riemannOnRowInRegs(xmin, xmax, smallp, gamma6, smallpp, qgdnvIDS, qgdnvIUS, qgdnvIPS,
                           qgdnvIVS, qleftIDS, qleftIUS, qleftIPS, qleftIVS, qrightIDS, qrightIUS,
                           qrightIPS, qrightIVS, getSGNM());
    }
}

void Tile::godunov() {

    constprim();

    eos(TILE_FULL);

    if (deviceSharedVariables()->m_order > 1) {
        slope();
    }
    trace(); // Was missing !

    qleftr();

    riemann();

    compflx();
}

real_t Tile::computeDt() {
    real_t dt = compute_dt();
    return dt;
}

void Tile::boundary_process(int32_t boundary_left, int32_t boundary_right, int32_t boundary_up,
                            int32_t boundary_down) {

    int32_t xmin, xmax, ymin, ymax;

    getExtends(TILE_INTERIOR, xmin, xmax, ymin, ymax);

    if (m_scan == X_SCAN) {
        if (boundary_left > 0) {

            for (int32_t ivar = 0; ivar < NB_VAR; ivar++) {
                auto uold = m_onDevice->m_uold(ivar);
                int32_t i_min = sycl::max(xmin + m_offx, 0);
                int32_t i_max = sycl::min(xmax + m_offx, m_ExtraLayer);
                for (int i = i_min; i < i_max; i++) {
                    int i0;
                    real_t sign = 1.0;
                    if (boundary_left == 1) {
                        int i0 = 2 * m_ExtraLayer - i - 1; // CL reflexion
                        if (ivar == IU_VAR) {
                            sign = -1.0;
                        }
                    } else if (boundary_left == 2) {
                        i0 = m_ExtraLayer; // CL absorbante
                    } else {
                        i0 = m_gnx + i; // CL periodique
                    }
#pragma ivdep
                    for (int j = ymin; j < ymax; j++) {
                        uold(i, m_offy + j) = uold(i0, m_offy + j) * sign;
                    }
                }
            }
        }

        if (boundary_right > 0) {
            // Right boundary
            for (int32_t ivar = 0; ivar < NB_VAR; ivar++) {
                auto uold = m_onDevice->m_uold(ivar);
                int32_t i_min = sycl::max(xmin + m_offx, m_gnx + m_ExtraLayer);
                int32_t i_max = sycl::min(xmax + m_offx, m_gnx + 2 * m_ExtraLayer);
                for (int32_t i = i_min; i < i_max; i++) {
                    real_t sign = 1.0;
                    int32_t i0;
                    if (boundary_right == 1) {
                        i0 = 2 * m_gnx + 2 * m_ExtraLayer - i - 1;
                        if (ivar == IU_VAR) {
                            sign = -1.0;
                        }
                    } else if (boundary_right == 2) {
                        i0 = m_gnx + m_ExtraLayer;
                    } else {
                        i0 = i - m_gnx;
                    }
#pragma ivdep
                    for (int j = ymin; j < ymax; j++) {
                        uold(i, m_offy + j) = uold(i0, m_offy + j) * sign;
                    }
                }
            }
        }
    } // X_SCAN

    if (m_scan == Y_SCAN) {
        int32_t temp = xmin;
        xmin = ymin;
        ymin = temp;
        temp = xmax;
        xmax = ymax;
        ymax = temp;

        // Lower boundary
        if (boundary_down > 0) {
            for (int32_t ivar = 0; ivar < NB_VAR; ivar++) {
                auto uold = m_onDevice->m_uold(ivar);
                int32_t j_min = sycl::max(ymin + m_offy, 0);
                int32_t j_max = sycl::min(ymax + m_offy, m_ExtraLayer);
                for (int32_t j = j_min; j < j_max; j++) {
                    real_t sign = 1.0;

                    int32_t j0 = 0;
                    if (boundary_down == 1) {
                        j0 = 2 * m_ExtraLayer - j - 1;
                        if (ivar == IV_VAR) {
                            sign = -1;
                        }
                    } else if (boundary_down == 2) {
                        j0 = m_ExtraLayer;
                    } else {
                        j0 = m_gny + j;
                    }
#pragma ivdep
                    for (int32_t i = xmin; i < xmax; i++) {
                        uold(m_offx + i, j) = uold(m_offx + i, j0) * sign;
                    }
                }
            }
        }
        // Upper boundary
        if (boundary_up > 0) {
            for (int32_t ivar = 0; ivar < NB_VAR; ivar++) {
                auto uold = m_onDevice->m_uold(ivar);
                int32_t j_min = sycl::max(ymin + m_offy, m_gny + m_ExtraLayer);
                int32_t j_max = sycl::min(ymax + m_offy, m_gny + 2 * m_ExtraLayer);
                for (int32_t j = j_min; j < j_max; j++) {
                    real_t sign = 1;
                    int32_t j0;
                    if (boundary_up == 1) {
                        j0 = 2 * m_gny + 2 * m_ExtraLayer - j - 1;
                        if (ivar == IV_VAR) {
                            sign = -1;
                        }
                    } else if (boundary_up == 2) {
                        j0 = m_gny + 1;
                    } else {
                        j0 = j - m_gny;
                    }
#pragma ivdep
                    for (int32_t i = xmin; i < xmax; i++) {
                        uold(m_offx + i, j) = uold(m_offx + i, j0) * sign;
                    }
                }
            }
        }
    } // Y_SCAN
}
// EOF
