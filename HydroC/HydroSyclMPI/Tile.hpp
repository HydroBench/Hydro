//
// (C) Guillaume.Colin-de-Verdiere at CEA.Fr
//
#ifndef TILE_H
#define TILE_H
//


#include "EnumDefs.hpp"
#include "precision.hpp"
#include "SoaDevice.hpp"
#include "Tile_Shared_Variables.hpp"



class DeviceBuffers;
struct TilesSharedVariables;

class Tile {

  private:

    int32_t m_hasBeenProcessed;

    // dimensions
    int32_t m_nx, m_ny;     // internal box size
    int32_t m_offx, m_offy; // offset of the tile in the domain
    int32_t m_gnx, m_gny;   // total domain size
    int32_t m_ExtraLayer;

    // Godunov variables and arrays
    godunovDir_t m_scan;

    real_t m_tcur, m_dt, m_dx;

    // work arrays private to a tile
    SoaDevice<real_t> m_u;    // NXT, NYT
    SoaDevice<real_t> m_flux; // NX + 1, NY + 1

    // work arrays for a tile which can be shared across workitems
    DeviceBuffers *m_work;
    // Shared variables for all tiles

    const sycl::stream *m_cout;
    TilesSharedVariables *m_onDevice;

    bool m_swapped;
    //

    // compute routines
    SYCL_EXTERNAL
    void slopeOnRow(int32_t xmin, int32_t xmax, Preal_t qS, Preal_t dqS); // fait
    SYCL_EXTERNAL
    void slope(); // fait
    SYCL_EXTERNAL
    void traceonRow(int32_t xmin, int32_t xmax, real_t dtdx, real_t zeror, real_t zerol,
                    real_t project, Preal_t cS, Preal_t qIDS, Preal_t qIUS, Preal_t qIVS,
                    Preal_t qIPS, Preal_t dqIDS, Preal_t dqIUS, Preal_t dqIVS, Preal_t dqIPS,
                    Preal_t pqxpIDS, Preal_t pqxpIUS, Preal_t pqxpIVS, Preal_t pqxpIPS,
                    Preal_t pqxmIDS, Preal_t pqxmIUS, Preal_t pqxmIVS, Preal_t pqxmIPS); // fait
    SYCL_EXTERNAL
    void trace();
    SYCL_EXTERNAL // fait
        void
        qleftrOnRow(int32_t xmin, int32_t xmax, Preal_t pqleftS, Preal_t pqrightS, Preal_t pqxmS,
                    Preal_t pqxpS); // fait
    SYCL_EXTERNAL
    void qleftr();
    SYCL_EXTERNAL // fait
        void
        compflxOnRow(int32_t xmin, int32_t xmax, real_t entho, Preal_t qgdnvIDS, Preal_t qgdnvIUS,
                     Preal_t qgdnvIVS, Preal_t qgdnvIPS, Preal_t fluxIVS, Preal_t fluxIUS,
                     Preal_t fluxIPS, Preal_t fluxIDS); // fait
    SYCL_EXTERNAL
    void compflx(); // fait
    SYCL_EXTERNAL
    void eosOnRow(int32_t xmin, int32_t xmax, real_t smallp, Preal_t qIDS, Preal_t eS, Preal_t qIPS,
                  Preal_t cS); // fait
    SYCL_EXTERNAL
    void eos(tileSpan_t span); // fait
    SYCL_EXTERNAL
    void constprimOnRow(int32_t xmin, int32_t xmax, Preal_t qIDS, Preal_t qIPS, Preal_t qIVS,
                        Preal_t qIUS, Preal_t uIDS, Preal_t uIPS, Preal_t uIVS, Preal_t uIUS,
                        Preal_t eS);
    SYCL_EXTERNAL
    void constprim(); // fait
    SYCL_EXTERNAL
    void riemannOnRow(int32_t xmin, int32_t xmax, real_t smallp, real_t gamma6, real_t smallpp,
                      Preal_t qgdnvIDS, Preal_t qgdnvIUS, Preal_t qgdnvIPS, Preal_t qgdnvIVS,
                      Preal_t qleftIDS, Preal_t qleftIUS, Preal_t qleftIPS, Preal_t qleftIVS,
                      Preal_t qrightIDS, Preal_t qrightIUS, Preal_t qrightIPS, Preal_t qrightIVS,
                      long *__restrict__ goon, Preal_t sgnm, Preal_t pstar, Preal_t rl, Preal_t ul,
                      Preal_t pl, Preal_t rr, Preal_t ur, Preal_t pr, Preal_t cl,
                      Preal_t cr); // fait
    SYCL_EXTERNAL
    void riemannOnRowInRegs(int32_t xmin, int32_t xmax, real_t smallp, real_t gamma6,
                            real_t smallpp, Preal_t qgdnvIDS, Preal_t qgdnvIUS, Preal_t qgdnvIPS,
                            Preal_t qgdnvIVS, Preal_t qleftIDS, Preal_t qleftIUS, Preal_t qleftIPS,
                            Preal_t qleftIVS, Preal_t qrightIDS, Preal_t qrightIUS,
                            Preal_t qrightIPS, Preal_t qrightIVS, Preal_t sgnm);
    SYCL_EXTERNAL
    void riemann(); // fait
    SYCL_EXTERNAL
    void compute_dt_loop2OnRow(real_t &tmp1, real_t &tmp2, int32_t xmin, int32_t xmax, Preal_t cS,
                               Preal_t qIUS, Preal_t qIVS);
    SYCL_EXTERNAL
    void compute_dt_loop1OnRow(int32_t xmin, int32_t xmax, Preal_t qIDS, Preal_t qIPS, Preal_t qIUS,
                               Preal_t qIVS, Preal_t uoldIDS, Preal_t uoldIUS, Preal_t uoldIVS,
                               Preal_t uoldIPS, Preal_t eS);
    SYCL_EXTERNAL
    real_t compute_dt();
    SYCL_EXTERNAL
    void gatherconservXscan(int32_t xmin, int32_t xmax, Preal_t uIDS, Preal_t uIUS, Preal_t uIVS,
                            Preal_t uIPS, Preal_t uoldIDS, Preal_t uoldIUS, Preal_t uoldIVS,
                            Preal_t uoldIPS);
    SYCL_EXTERNAL
    void gatherconservYscan();
    SYCL_EXTERNAL
    void updateconservXscan(int32_t xmin, int32_t xmax, real_t dtdx, Preal_t uIDS, Preal_t uIUS,
                            Preal_t uIVS, Preal_t uIPS, Preal_t uoldIDS, Preal_t uoldIUS,
                            Preal_t uoldIVS, Preal_t uoldIPS, Preal_t fluxIDS, Preal_t fluxIVS,
                            Preal_t fluxIUS, Preal_t fluxIPS);
    SYCL_EXTERNAL
    void updateconservYscan(int32_t s, int32_t xmin, int32_t xmax, int32_t ymin, int32_t ymax,
                            real_t dtdx, RArray2D<real_t> &uoldID, RArray2D<real_t> &uoldIP,
                            RArray2D<real_t> &uoldIV, RArray2D<real_t> &uoldIU, Preal_t fluxIVS,
                            Preal_t fluxIUS, Preal_t fluxIPS, Preal_t fluxIDS, Preal_t uIDS,
                            Preal_t uIPS, Preal_t uIVS, Preal_t uIUS, Preal_t pl);

    SYCL_EXTERNAL
    void getExtends(tileSpan_t span, int32_t &xmin, int32_t &xmax, int32_t &ymin, int32_t &ymax) {
        // returns the dimension of the tile with or without ghost cells.
        if (span == TILE_INTERIOR) {
            xmin = m_ExtraLayer;
            xmax = m_ExtraLayer + m_nx;
            ymin = m_ExtraLayer;
            ymax = m_ExtraLayer + m_ny;
        } else { // TILE_FULL
            xmin = 0;
            xmax = 2 * m_ExtraLayer + m_nx;
            ymin = 0;
            ymax = 2 * m_ExtraLayer + m_ny;
        }

        if (m_scan == Y_SCAN) {
            int t = xmin;
            xmin = ymin;
            ymin = t;
            t = xmax;
            xmax = ymax;
            ymax = t;
        }
    };

  protected:
  public:
    // basic constructor
    Tile(void); // default constructor
    // destructor
    ~Tile();

    void initTile();
    bool isSwapped() const { return m_swapped; }

    SYCL_EXTERNAL
    void setOut(const sycl::stream &out) { m_cout = &out; }

    // Has to be defined at the beginning of the kernel with setOut
    SYCL_EXTERNAL
    const sycl::stream &cout() { return *m_cout; }

    void setShared(TilesSharedVariables *ptr) { m_onDevice = ptr; }

    SYCL_EXTERNAL
    TilesSharedVariables *deviceSharedVariables() { return m_onDevice; }

    SYCL_EXTERNAL
    void swapScan() {
        if (m_scan == X_SCAN)
            m_scan = Y_SCAN;
        else
            m_scan = X_SCAN;
    }

    SYCL_EXTERNAL
    godunovDir_t godunovDir() const { return m_scan; }
    SYCL_EXTERNAL
    void swapStorageDims();

    SYCL_EXTERNAL
    void godunov();

    SYCL_EXTERNAL //
        void
        gatherconserv(); // fait
    SYCL_EXTERNAL
    void updateconserv(); // fait
    SYCL_EXTERNAL
    real_t computeDt(); // returns local time step

    // set/get
    SYCL_EXTERNAL
    void setTcur(real_t t) { m_tcur = t; }
    SYCL_EXTERNAL
    void setDt(real_t t) { m_dt = t; }

    void setExtend(int32_t nx, int32_t ny, int32_t gnx, int32_t gny, int32_t offx, int32_t offy,
                   real_t dx);

    SYCL_EXTERNAL
    void setBuffers(DeviceBuffers *buf);

    void notProcessed() { m_hasBeenProcessed = 0; }
    void doneProcessed(int step) { m_hasBeenProcessed = step; }
    int32_t isProcessed(int step) { return m_hasBeenProcessed == step; }

    SYCL_EXTERNAL
    void infos();

#if 0

        return r;
    }
    // for checkpoint / restart
    long getLengthByte();
    void read(const int f);
    void write(const int f);
#endif
};
#endif
// EOF
