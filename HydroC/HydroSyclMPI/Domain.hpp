//
// (C) Guillaume.Colin-de-Verdiere at CEA.Fr
//
#ifndef DOMAIN_H
#define DOMAIN_H
//

#include "EnumDefs.hpp"
#include "Soa.hpp"
#include "Tile.hpp"
#include "Tile_Shared_Variables.hpp"
#include "TimeLimit.hpp"

#include <vector>

#if WITHPNG > 0
#include <png.h>
#endif

class Domain {
  private:
    // variables to protect between runs

    Soa *m_uold; // on the full domain

    // Tile shared variables
    TilesSharedVariables *m_onDevice;
    real_t *local_dt; // share variable to store dt has to be replaced by a reduce when working!

    real_t m_tcur, m_dt;        //=
    int32_t m_globNx, m_globNy; // global size of the simulation //=
    godunovDir_t m_scan;        //=
    int m_nvtk;
    int m_npng;
    int32_t m_iter; // current iteration
    int32_t m_nbRun;
    double m_elapsTotal;
    Timers m_mainTimer;

    // those variables are not protected to allow for modifications
    TimeLimit m_tr;
    int m_prt;
    int32_t m_stats; // print various stats
    Tile *m_tiles;   //=
    Tile *m_tilesOnDevice;

    int32_t m_nbTiles;  //=
    int32_t m_tileSize; //=

    int32_t m_nbWorkItems; // nb of threads available
    int32_t m_withMorton;
    int32_t *m_mortonIdx;
    Matrix2<int32_t> *m_morton;

    long m_maxrss, m_ixrss; // memory usage;

    int32_t m_nx, m_ny;  // size of the local domain
    int m_box[MAXBOX_D]; // size of our current domain in global coordinate
    real_t m_dx;         //=
    int32_t m_boundary_left, m_boundary_right, m_boundary_down, m_boundary_up; //=
    int32_t m_ExtraLayer;
    real_t m_gamma, m_smallc, m_smallr, m_slope_type; //=
    int32_t m_testcase;                               //=
    int32_t m_iorder;                                 //=
    real_t m_cfl;                                     //=
    int32_t m_nIterRiemann;                           //=

    int32_t m_nStepMax; //=
    int32_t m_nOutput;  //=
    int32_t m_checkPoint;
    int32_t m_forceStop; //=

    real_t *m_localDt; //=
    real_t m_tend;     //=
    real_t m_dtOutput; //=
    real_t m_nextOutput;
    real_t m_dtImage; //=
    int32_t m_nImage; //=
    real_t m_nextImage;
    godunovScheme_t m_scheme; //=
    bool m_StepbyStep;

    //

    // working arrays
    real_t *m_recvbufru; // receive right or up  //=
    real_t *m_recvbufld; // receive left or down //=
    real_t *m_sendbufru; // send right or up     //=
    real_t *m_sendbufld; // send left or down    //=

    // misc.
    char *m_inputFile;
    int32_t m_fakeRead;
    int64_t m_fakeReadSize;
    int32_t m_nDumpline;

    // PNG output
    int32_t m_withPng;
    FILE *m_fp;

#if WITHPNG > 0
    png_structp m_png_ptr;
    png_infop m_info_ptr;
    png_bytep *m_row_pointers;
    png_byte *²ffer;
#else
    uint8_t *m_buffer;
#endif

    int32_t m_shrink;
    int32_t m_shrinkSize;
    double m_timeGuard;
    int32_t m_numa; // try to cope with numa effects
    int32_t m_forceSync;
    // timing of functions
    double **m_timerLoops;
    Timers *m_threadTimers; // one Timers per thread
    int32_t m_tasked;       // use tasks
    int32_t m_taskeddep;    // use tasks with dependecies

    // member functions
    void vtkfile(int step);
    void vtkOutput(int step);
    void parseParams(int argc, char **argv);
    void readInput();
    void domainDecompose();
    void setTiles();
    void printSummary();

    void sendUoldToDevice();
    void getUoldFromDevice();

    void boundary_init();

    void boundary_process();
    real_t computeTimeStep();
    real_t computeTimeStepByStep(bool doComputeDt);
    void compTStask1();
    void compTStask2();
    void compTStask3();
    void computeDt();
    void createTestCase();
    void changeDirection();
    real_t reduceMin(real_t val);
    real_t reduceMaxAndBcast(real_t val);

    void getExtends(tileSpan_t span, int32_t &xmin, int32_t &xmax, int32_t &ymin, int32_t &ymax) {
        // returns the dimension of the tile with ghost cells.
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
    };
    // pack/unpack array for ghost cells exchange. Works for MPI
    int32_t pack_arrayv(int32_t xoffset, real_t *buffer);
    int32_t unpack_arrayv(int32_t xoffset, real_t *buffer);
    int32_t pack_arrayh(int32_t yoffset, real_t *buffer);
    int32_t unpack_arrayh(int32_t yoffset, real_t *buffer);

    int32_t nbTiles_fromsize(int32_t tileSize) {

        int nbtx = (m_nx + tileSize - 1) / tileSize;
        int nbty = (m_ny + tileSize - 1) / tileSize;

        return nbtx * nbty;
    }

    Domain(void) = delete;
    void swapScan() {
        if (m_scan == X_SCAN)
            m_scan = Y_SCAN;
        else
            m_scan = X_SCAN;
    }

    void pngWriteFile(char *name);
    void pngProcess(void);
    void pngCloseFile(void);
    void getMaxVarValues(real_t *mxP, real_t *mxD, real_t *mxUV);
    void pngFillGap(int curx, int cury, int nx, int ny, int Iptr[4], int &cpt, int32_t imgSizeX,
                    int32_t imgSizeY);

    // Checkpoint helper routines
    long protectScalars(const protectionMode_t mode, const int f);
    long protectArrays(const protectionMode_t mode, const int f);
    long protectTiles(const protectionMode_t mode, const int f);
    void writeProtectionVars(const int f);
    void readProtectionVars(const int f);
    void writeProtectionHeader(const int f);
    void readProtectionHeader(const int f);

    bool hasProtection();

    void saveProtection();
    void writeProtection();
    void readProtection();

    bool StopComputation();
    void dumpLine(void);
    void dumpOneArray(FILE *f, Matrix2<real_t> &p);
    void dumpLineArray(FILE *f, Matrix2<real_t> &p, char *name, char *ext);

    void debuginfos();

  protected:
  public:
    // basic constructor
    Domain(int argc, char **argv);
    // destructor
    ~Domain();
    bool isStopped();

    void compute();
    int32_t myThread() {
        int32_t r = 0;

        return r;
    }
};
#endif
// EOF
