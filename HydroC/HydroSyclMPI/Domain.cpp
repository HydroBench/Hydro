//
// (C) Guillaume.Colin-de-Verdiere at CEA.Fr
//


#include "ParallelInfo.hpp"

//
#include "Domain.hpp"



#include <unistd.h>

Domain::Domain(int argc, char **argv) {
    // default values
    m_numa = 1;
    m_nextOutput = 0;
    m_nextImage = 0;
    m_nImage = 0;
    m_nbRun = 0;
    m_elapsTotal = 0.0;
    m_dtImage = 0;
    m_iter = 0;
    m_checkPoint = 0;
    m_forceStop = 0;
    m_forceSync = 0; // leave the OS alone for I/O management
    m_withPng = 0;
    m_npng = 0;
    m_shrink = 1;
    m_shrinkSize = 2000;
#if WITHPNG > 0
    m_buffer = 0;
#endif
    m_prt = 0;
    m_stats = 1; // Default we print everything
    m_inputFile = 0;
    m_withMorton = 0;
    m_ExtraLayer = 2;
    m_globNx = 128;
    m_globNy = 128;
    m_tileSize = 16;
    m_nbtiles = 1;
    m_dx = 0.05;
    m_boundary_left = 1;
    m_boundary_right = 1;
    m_boundary_down = 1;
    m_boundary_up = 1;
    m_nIterRiemann = 10;
    m_testcase = 0;
    m_cfl = 0.8;
    m_nStepMax = 0;
    m_nOutput = 0;
    m_nvtk = 0;
    m_tend = 100;
    m_tcur = 0.0;
    m_dt = 0;
    m_dx = 0.05;
    m_dtOutput = 0;
    m_scan = X_SCAN;
    m_scheme = SCHEME_MUSCL;
    m_tiles = 0;
    m_localDt = 0;
    m_gamma = 1.4;
    m_smallc = 1e-10;
    m_smallr = 1e-10;
    m_iorder = 2;
    m_slope_type = 1.;
    m_recvbufru = 0; // receive right or up
    m_recvbufld = 0; // receive left or down
    m_sendbufru = 0; // send right or up
    m_sendbufld = 0; // send left or down


    m_nbWorkItems = 1; // runs serially by default
    m_fakeRead = 0;
    m_fakeReadSize = 3000000;
    m_tasked = 0;
    m_taskeddep = 0;
    m_nDumpline = -1;

    ParallelInfo::init(argc, argv, m_stats!=0);

    int myPe = ParallelInfo::mype();
    

    if (myPe == 0 && m_stats > 0) {
        int src = system("cpupower frequency-info | egrep 'CPU frequency|steps'");
    }

    double tRemain = m_tr.timeRemainAll();
    if (tRemain <= 1) {
        // useless run which can be harmful to files
        if (myPe == 0 && m_stats > 0) {
            std::cerr << "HydroC: allocated time too short " << tRemain << "s" << std::endl;
        }
#ifdef MPI_ON
        MPI_Abort(MPI_COMM_WORLD, 1);
#else
        abort();
#endif
    }

    m_timeGuard = 900;
    if (tRemain < 30000)
        m_timeGuard = 900;
    if (tRemain < 3600)
        m_timeGuard = 600;
    if (tRemain < 1800)
        m_timeGuard = 300;
    if (tRemain < 60)
        m_timeGuard = 20;

    if (myPe == 0 && m_stats > 0) {
        std::cout << "HydroC: allocated time " << m_tr.getTimeAllocated() << "s"
                  << " time guard " << m_timeGuard << "s" << std::endl;
        std::cout.flush();
    }

    parseParams(argc, argv);
    readInput();
    domainDecompose();

    createTestCase(); // will be overwritten if in the middle of test case

    if ((m_nOutput > 0) || (m_dtOutput > 0)) {
        vtkOutput(m_nvtk);
        m_nvtk++;
    }
    setTiles();
    if (myPe == 0) {
        printSummary();
    }
}

Domain::~Domain() {
    delete m_uold;
    if (m_tiles) {
        for (int32_t i = 0; i < m_nbtiles; i++) {
            if (m_tiles[i])
                delete m_tiles[i];
        }
        delete[] m_tiles;
    }
    if (m_recvbufru)
        free(m_recvbufru);
    if (m_recvbufld)
        free(m_recvbufld);
    if (m_sendbufru)
        free(m_sendbufru);
    if (m_sendbufld)
        free(m_sendbufld);
    if (m_localDt)
        free(m_localDt);

    for (int32_t i = 0; i < m_nbWorkItems; i++) {
        delete m_buffers[i];
    }
    delete[] m_buffers;
    delete[] m_threadTimers;
    delete[] m_mortonIdx;
    if (m_morton)
        delete m_morton;
    if (m_timerLoops) {
        for (int32_t i = 0; i < m_nbWorkItems; i++) {
            delete[] m_timerLoops[i];
        }
        delete[] m_timerLoops;
    }
    if (m_inputFile)
        free(m_inputFile);

}

void Domain::domainDecompose() {
    int32_t xmin, xmax, ymin, ymax;
    int32_t lgx, lgy, lgmax;

    m_nx = m_globNx;
    m_ny = m_globNy;

    m_box[XMIN_D] = 0;
    m_box[XMAX_D] = m_nx;
    m_box[YMIN_D] = 0;
    m_box[YMAX_D] = m_ny;
    m_box[LEFT_D] = -1;
    m_box[RIGHT_D] = -1;
    m_box[DOWN_D] = -1;
    m_box[UP_D] = -1;

    int nProc = ParallelInfo::nb_procs();
    int myPe = ParallelInfo::mype();
    if (nProc > 1) {
        CalcSubSurface(0, m_globNx, 0, m_globNy, 0, nProc - 1, m_box, myPe);

        m_nx = m_box[XMAX_D] - m_box[XMIN_D];
        m_ny = m_box[YMAX_D] - m_box[YMIN_D];

        // adapt the boundary conditions
        if (m_box[LEFT_D] != -1) {
            m_boundary_left = 0;
        }
        if (m_box[RIGHT_D] != -1) {
            m_boundary_right = 0;
        }
        if (m_box[DOWN_D] != -1) {
            m_boundary_down = 0;
        }
        if (m_box[UP_D] != -1) {
            m_boundary_up = 0;
        }

        if (m_prt) {
            std::cout << m_globNx << " ";
            std::cout << m_globNy << " ";
            std::cout << nProc << " ";
            std::cout << myPe << " - ";
            std::cout << m_nx << " ";
            std::cout << m_ny << " - ";
            std::cout << m_box[LEFT_D] << " ";
            std::cout << m_box[RIGHT_D] << " ";
            std::cout << m_box[DOWN_D] << " ";
            std::cout << m_box[UP_D] << std::endl;
        }
    }

    getExtends(TILE_FULL, xmin, xmax, ymin, ymax);

    lgx = (xmax - xmin);
    lgy = (ymax - ymin);
    lgmax = lgx;
    if (lgmax < lgy)
        lgmax = lgy;

    lgmax *= m_ExtraLayer;

    m_recvbufru = AlignedAllocReal(lgmax * NB_VAR);
    memset(m_recvbufru, 0, lgmax * NB_VAR * sizeof(real_t));
    m_recvbufld = AlignedAllocReal(lgmax * NB_VAR);
    memset(m_recvbufld, 0, lgmax * NB_VAR * sizeof(real_t));
    m_sendbufru = AlignedAllocReal(lgmax * NB_VAR);
    memset(m_sendbufru, 0, lgmax * NB_VAR * sizeof(real_t));
    m_sendbufld = AlignedAllocReal(lgmax * NB_VAR);
    memset(m_sendbufld, 0, lgmax * NB_VAR * sizeof(real_t));
}

void Domain::printSummary() {
    if (ParallelInfo::mype() == 0 && m_stats > 0) {
        printf("|+=+=+=+=+=+=+=\n");
        printf("|    GlobNx=     %d\n", m_globNx);
        printf("|    GlobNy=     %d\n", m_globNy);
        printf("|    nx=         %d\n", m_nx);
        printf("|    ny=         %d\n", m_ny);

        printf("|    ts=         %d\n", m_tileSize);

        printf("|    nt=         %d\n", m_nbtiles);
        printf("|    tasked=     %u\n", m_tasked);
        printf("|    taskeddep=  %u\n", m_taskeddep);
        printf("|    morton=     %u\n", m_withMorton);
        printf("|    numa=       %u\n", m_numa);
        printf("|    tend=       %lf\n", m_tend);
        printf("|    nstepmax=   %d\n", m_nStepMax);
        printf("|    ndumpline=  %d\n", m_nDumpline);
        printf("|    stats=      %d\n", m_stats);
        printf("|    noutput=    %d\n", m_nOutput);
        printf("|    dtoutput=   %lf\n", m_dtOutput);
        printf("|    dtimage=    %lf\n", m_dtImage);
        printf("|    nimage=     %d\n", m_nImage);
        printf("|    chkpt=      %d\n", m_checkPoint);
        printf("|    forcestop=  %d\n", m_forceStop);
        printf("|    forcesync=  %d\n", m_forceSync);
        printf("|    fakeread=   %d\n", m_fakeRead);
        printf("|fakereadsize=   %ld\n", m_fakeReadSize);
        printf("|+=+=+=+=+=+=+=\n\n");
    }
}

static void keyval(char *buffer, char **pkey, char **pval) {
    char *ptr;
    *pkey = buffer;
    *pval = buffer;

    // kill the newline
    ptr = strchr(buffer, '\n');
    if (ptr)
        *ptr = 0;
    // strip comment from value
    ptr = strchr(buffer, '#');
    if (ptr)
        *ptr = 0;

    // suppress leading whites or tabs from key
    while ((**pkey == ' ') || (**pkey == '\t'))
        (*pkey)++;

    ptr = strchr(buffer, '=');
    if (ptr) {
        *ptr = 0;
        ptr++;
    } else {
        ptr = buffer; // no = sign, probably a comment
    }
    // suppress leading whites or tabs from value
    while ((*ptr == ' ') || (*ptr == '\t'))
        ptr++;
    *pval = ptr;

    // strip key from ending white or tab
    while ((ptr = strchr(*pkey, ' ')) != NULL) {
        *ptr = 0;
    }
    while ((ptr = strchr(*pkey, '\t')) != NULL) {
        *ptr = 0;
    }
    // strip val from ending white or tab
    while ((ptr = strchr(*pval, ' ')) != NULL) {
        *ptr = 0;
    }
    while ((ptr = strchr(*pval, '\t')) != NULL) {
        *ptr = 0;
    }
}

void Domain::readInput() {
    FILE *fd = NULL;
    char buffer[1024];
    char *pval, *pkey;
    char *realFmt;
    int nbvalint = 0;
    int nbvallng = 0;
    int nbvalflt = 0;
    int nbvaldbl = 0;
    int tabint[100];
    long tablng[100];
    float tabflt[100];
    double tabdbl[100];

    if (sizeof(real_t) == sizeof(double)) {
        realFmt = (char *)("%lf");
    } else {
        realFmt = (char *)("%f");
    }

#ifdef WITHBCAST
    if (ParallelInfo::mype() == 0)
#endif
    {
        fd = fopen(m_inputFile, "r");
        if (fd == NULL) {
            std::cerr << "can't read input file\n" << std::endl;
            abort();
        }
        while (fgets(buffer, 1024, fd) == buffer) {
            keyval(buffer, &pkey, &pval);

            // int parameters
            if (strcmp(pkey, "nstepmax") == 0) {
                sscanf(pval, "%d", &m_nStepMax);
                continue;
            }
            if (strcmp(pkey, "chkpt") == 0) {
                sscanf(pval, "%d", &m_checkPoint);
                continue;
            }
            if (strcmp(pkey, "prt") == 0) {
                sscanf(pval, "%d", &m_prt);
                continue;
            }
            if (strcmp(pkey, "nx") == 0) {
                sscanf(pval, "%d", &m_globNx);
                continue;
            }
            if (strcmp(pkey, "ny") == 0) {
                sscanf(pval, "%d", &m_globNy);
                continue;
            }
            if (strcmp(pkey, "tilesize") == 0) {
                sscanf(pval, "%d", &m_tileSize);
                continue;
            }
            if (strcmp(pkey, "boundary_left") == 0) {
                sscanf(pval, "%d", &m_boundary_left);
                continue;
            }
            if (strcmp(pkey, "boundary_right") == 0) {
                sscanf(pval, "%d", &m_boundary_right);
                continue;
            }
            if (strcmp(pkey, "boundary_up") == 0) {
                sscanf(pval, "%d", &m_boundary_up);
                continue;
            }
            if (strcmp(pkey, "boundary_down") == 0) {
                sscanf(pval, "%d", &m_boundary_down);
                continue;
            }
            if (strcmp(pkey, "niter_riemann") == 0) {
                sscanf(pval, "%d", &m_nIterRiemann);
                continue;
            }
            if (strcmp(pkey, "noutput") == 0) {
                sscanf(pval, "%d", &m_nOutput);
                continue;
            }
            if (strcmp(pkey, "numa") == 0) {
                sscanf(pval, "%d", &m_numa);
                continue;
            }
            if (strcmp(pkey, "iorder") == 0) {
                sscanf(pval, "%d", &m_iorder);
                continue;
            }
            if (strcmp(pkey, "tasked") == 0) {
                sscanf(pval, "%d", &m_tasked);
                continue;
            }
            if (strcmp(pkey, "taskeddep") == 0) {
                sscanf(pval, "%d", &m_taskeddep);
                continue;
            }
            if (strcmp(pkey, "morton") == 0) {
                sscanf(pval, "%d", &m_withMorton);
                continue;
            }
            // float parameters
            if (strcmp(pkey, "slope_type") == 0) {
                // sscanf(pval, realFmt, &H->slope_type);
                continue;
            }
            if (strcmp(pkey, "tend") == 0) {
                sscanf(pval, realFmt, &m_tend);
                continue;
            }
            if (strcmp(pkey, "dx") == 0) {
                sscanf(pval, realFmt, &m_dx);
                continue;
            }
            if (strcmp(pkey, "courant_factor") == 0) {
                sscanf(pval, realFmt, &m_cfl);
                continue;
            }
            if (strcmp(pkey, "smallr") == 0) {
                // sscanf(pval, realFmt, &H->smallr);
                continue;
            }
            if (strcmp(pkey, "smallc") == 0) {
                // sscanf(pval, realFmt, &H->smallc);
                continue;
            }
            if (strcmp(pkey, "dtoutput") == 0) {
                sscanf(pval, realFmt, &m_dtOutput);
                continue;
            }
            if (strcmp(pkey, "dtimage") == 0) {
                sscanf(pval, realFmt, &m_dtImage);
                continue;
            }
            if (strcmp(pkey, "nimage") == 0) {
                sscanf(pval, "%d", &m_nImage);
                continue;
            }
            if (strcmp(pkey, "ndumpline") == 0) {
                sscanf(pval, "%d", &m_nDumpline);
                continue;
            }
            if (strcmp(pkey, "forcesync") == 0) {
                sscanf(pval, "%d", &m_forceSync);
                continue;
            }
            if (strcmp(pkey, "forcestop") == 0) {
                sscanf(pval, "%d", &m_forceStop);
                continue;
            }
            if (strcmp(pkey, "testcase") == 0) {
                sscanf(pval, "%d", &m_testcase);
                continue;
            }
            if (strcmp(pkey, "stats") == 0) {
                sscanf(pval, "%d", &m_stats);
                continue;
            }
            if (strcmp(pkey, "fakeread") == 0) {
                sscanf(pval, "%d", &m_fakeRead);
                continue;
            }
            if (strcmp(pkey, "fakereadsize") == 0) {
                sscanf(pval, "%ld", &m_fakeReadSize);
                // std::cerr << m_fakeReadSize << std::endl;
                continue;
            }
            // string parameter
            if (strcmp(pkey, "scheme") == 0) {
                // std::cerr << "[" << pval << "]" << std::endl;
                if (strstr(pval, "muscl") != 0) {
                    m_scheme = SCHEME_MUSCL;
                } else if (strstr(pval, "plmde") != 0) {
                    m_scheme = SCHEME_PLMDE;
                } else if (strstr(pval, "collela") != 0) {
                    m_scheme = SCHEME_COLLELA;
                } else {
                    std::cerr << "Scheme name <%s> is unknown, should be one of "
                                 "[muscl,plmde,collela]\n"
                              << pval << std::endl;
                    abort();
                }
                continue;
            }
        }
    }
#ifdef MPI_ON
#ifdef WITHBCAST
    {
        int checkValint = 0;
        nbvalint = 0;
        tabint[nbvalint++] = m_nStepMax;
        tabint[nbvalint++] = m_checkPoint;
        tabint[nbvalint++] = m_prt;
        tabint[nbvalint++] = m_globNx;
        tabint[nbvalint++] = m_globNy;
        tabint[nbvalint++] = m_tileSize;
        tabint[nbvalint++] = m_boundary_left;
        tabint[nbvalint++] = m_boundary_right;
        tabint[nbvalint++] = m_boundary_up;
        tabint[nbvalint++] = m_boundary_down;
        tabint[nbvalint++] = m_nIterRiemann;
        tabint[nbvalint++] = m_nOutput;
        tabint[nbvalint++] = m_numa;
        tabint[nbvalint++] = m_iorder;
        tabint[nbvalint++] = m_tasked;
        tabint[nbvalint++] = m_taskeddep;
        tabint[nbvalint++] = m_withMorton;
        tabint[nbvalint++] = m_nImage;
        tabint[nbvalint++] = m_forceSync;
        tabint[nbvalint++] = m_forceStop;
        tabint[nbvalint++] = m_testcase;
        tabint[nbvalint++] = m_fakeRead;
        tabint[nbvalint++] = m_scheme;
        tabint[nbvalint++] = m_nDumpline;
        checkValint = nbvalint;
        MPI_Bcast(tabint, nbvalint, MPI_INT, 0, MPI_COMM_WORLD);
        if (m_myPe > 0) {
            nbvalint = 0;
            m_nStepMax = tabint[nbvalint++];
            m_checkPoint = tabint[nbvalint++];
            m_prt = tabint[nbvalint++];
            m_globNx = tabint[nbvalint++];
            m_globNy = tabint[nbvalint++];
            m_tileSize = tabint[nbvalint++];
            m_boundary_left = tabint[nbvalint++];
            m_boundary_right = tabint[nbvalint++];
            m_boundary_up = tabint[nbvalint++];
            m_boundary_down = tabint[nbvalint++];
            m_nIterRiemann = tabint[nbvalint++];
            m_nOutput = tabint[nbvalint++];
            m_numa = tabint[nbvalint++];
            m_iorder = tabint[nbvalint++];
            m_tasked = tabint[nbvalint++];
            m_taskeddep = tabint[nbvalint++];
            m_withMorton = tabint[nbvalint++];
            m_nImage = tabint[nbvalint++];
            m_forceSync = tabint[nbvalint++];
            m_forceStop = tabint[nbvalint++];
            m_testcase = tabint[nbvalint++];
            m_fakeRead = tabint[nbvalint++];
            m_scheme = (godunovScheme_t)tabint[nbvalint++];
            m_nDumpline = tabint[nbvalint++];
            // here we check that we have the right number of entries
            assert(checkValint == nbvalint);
        }

        nbvallng = 0;
        tablng[nbvallng++] = m_fakeReadSize;
        MPI_Bcast(tablng, nbvallng, MPI_INT, 0, MPI_COMM_WORLD);
        if (m_myPe > 0) {
            nbvallng = 0;
            m_fakeReadSize = tablng[nbvallng++];
        }
        if (sizeof(real_t) == sizeof(double)) {
            int checkValdbl = 0;
            nbvaldbl = 0;
            // tabdbl[nbvaldbl++] = H->slope_type;
            tabdbl[nbvaldbl++] = m_tend;
            tabdbl[nbvaldbl++] = m_dx;
            tabdbl[nbvaldbl++] = m_cfl;
            tabdbl[nbvaldbl++] = m_smallr;
            tabdbl[nbvaldbl++] = m_smallc;
            tabdbl[nbvaldbl++] = m_dtOutput;
            tabdbl[nbvaldbl++] = m_dtImage;
            checkValdbl = nbvaldbl;
            MPI_Bcast(tabdbl, nbvaldbl, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            if (m_myPe > 0) {
                nbvaldbl = 0;
                // H->slope_type =tabdbl[nbvaldbl++];
                m_tend = tabdbl[nbvaldbl++];
                m_dx = tabdbl[nbvaldbl++];
                m_cfl = tabdbl[nbvaldbl++];
                m_smallr = tabdbl[nbvaldbl++];
                m_smallc = tabdbl[nbvaldbl++];
                m_dtOutput = tabdbl[nbvaldbl++];
                m_dtImage = tabdbl[nbvaldbl++];
                assert(checkValdbl == nbvaldbl);
            }
        } else {
            int checkValflt = 0;
            nbvalflt = 0;
            // tabflt[nbvalflt++] = H->slope_type;
            tabflt[nbvalflt++] = m_tend;
            tabflt[nbvalflt++] = m_dx;
            tabflt[nbvalflt++] = m_cfl;
            tabflt[nbvalflt++] = m_smallr;
            tabflt[nbvalflt++] = m_smallc;
            tabflt[nbvalflt++] = m_dtOutput;
            tabflt[nbvalflt++] = m_dtImage;
            checkValflt = nbvalflt;
            MPI_Bcast(tabflt, nbvalflt, MPI_FLOAT, 0, MPI_COMM_WORLD);
            if (m_myPe > 0) {
                nbvalflt = 0;
                // tabflt[nbvalflt++] = H->slope_type;
                tabflt[nbvalflt++] = m_tend;
                tabflt[nbvalflt++] = m_dx;
                tabflt[nbvalflt++] = m_cfl;
                tabflt[nbvalflt++] = m_smallr;
                tabflt[nbvalflt++] = m_smallc;
                tabflt[nbvalflt++] = m_dtOutput;
                tabflt[nbvalflt++] = m_dtImage;
                assert(checkValflt == nbvalflt);
            }
        }
    }
#endif
#endif
}

void Domain::parseParams(int argc, char **argv) {
    int32_t n = 1;
    while (n < argc) {
        if (strcmp(argv[n], "--help") == 0) {
            // usage();
            n++;
            continue;
        }
        if (strcmp(argv[n], "-v") == 0) {
            n++;
            continue;
        }
        if (strcmp(argv[n], "-i") == 0) {
            n++;
            m_inputFile = strdup(argv[n]);
            n++;
            continue;
        }
        std::cerr << "Key " << argv[n] << " is unkown\n" << std::endl;
        n++;
    }
}

void Domain::setTiles() {
    int32_t i, j, offx, offy, tileSizeX, tileSizeY, mortonW, mortonH;
    int32_t tileSizeM, tileSize;
    int32_t tileSizeOrg;
    Matrix2<int32_t> *mortonIdx; // to hold the array of tiles ids.
    //
    m_nbtiles = 0;
    tileSize = m_tileSize;

    int myPe = ParallelInfo::mype();

    if (tileSize <= 0) {
        int nTh = 1, nbT = 0, tsMin, tsMax, remMin, remain;
        
        int thMax, thCur;
        tileSize = 60;
        m_tileSize = tileSize;

        nTh = ParallelInfo::nb_workers();

        
        m_nbtiles = 1;
        tsMin = 58;
        tsMax = 256;

        // we want at least TILE_PER_THREAD tiles per thread.
        while (nbTiles_fromsize(tsMax) < (nTh * TILE_PER_THREAD))
            tsMax--;

        tsMax = std::max(tsMin, tsMax);
        
        int thMin = nbTiles_fromsize(tsMin) % nTh;
        int tsCur;
        for (tsCur = tsMin; tsCur < tsMax; tsCur++) {
            thCur = nbTiles_fromsize(tsCur) % nTh;
            if (thCur == 0) {
                tsMin = tsCur;
                break;
            }
            if (thCur < thMin) {
                tsMin = tsCur;
            }
        }
        if (tsCur >= tsMax)
            tsCur = tsMin;
        tileSize = tsCur;
        m_tileSize = tileSize;

        m_nbtiles = this->nbTiles_fromsize(tileSize);
        // std::cout << "End loop " << m_nbtiles << " " << tileSize << " " << (m_nbtiles
        // % nTh) << std::endl;

        if (myPe == 0 && m_stats > 0)
            std::cout << "Computing tilesize to " << tileSize
                      << " R=" << (float)m_nbtiles / (float)nTh << std::endl;
    } else {
        if (myPe == 0 && m_stats > 0)
            std::cout << "Forcing tilesize to " << tileSize << std::endl;
    }

    m_nbtiles = this->nbTiles_fromsize(tileSize);

    mortonH = (m_ny + tileSize - 1) / tileSize;
    mortonW = (m_nx + tileSize - 1) / tileSize;

    m_localDt = AlignedAllocReal(m_nbtiles);

    // Create the Morton holder to wander around the tiles
    m_morton = new Matrix2<int32_t>(mortonW, mortonH);
    // std::cerr << mortonW << " " << mortonH << std::endl;
    m_mortonIdx = m_morton->listMortonIdx();
    assert(m_mortonIdx != 0);

    m_nbtiles = 0;
    for (j = 0; j < mortonH; j++) {
        for (i = 0; i < mortonW; i++) {
            m_mortonIdx[m_nbtiles] = morton2(i, j);
            (*m_morton)(i, j) = m_nbtiles;
            m_nbtiles++;
        }
    }

    if (m_withMorton) {
        int32_t maxim = (*m_morton).maxMorton();
        int32_t *temp = new int32_t[maxim];
        for (int32_t i = 0; i < maxim; i++)
            temp[i] = -1;
        for (int32_t i = 0, tt = 0; i < m_nbtiles; i++) {
            temp[m_mortonIdx[i]] = tt++;
        }
        // compacter le tableau

        for (int32_t i = 0, t = 0; i < maxim; i++) {
            if (temp[i] != -1)
                m_mortonIdx[t++] = temp[i];
        }
        // for (int32_t ir = 0; ir < m_nbtiles; ir++) std::cerr << temp[ir] << " "; std::cerr
        // << std::endl;
        delete[] temp;
    }
    //

    m_tiles = new Tile *[m_nbtiles];
#ifdef _OPENMP
#pragma omp parallel for private(i) if (m_numa) SCHEDULE
#endif
    for (int32_t t = 0; t < m_nbtiles; t++) {
        i = t;
        if (m_withMorton)
            i = m_mortonIdx[t];
        m_tiles[i] = new Tile;
    }

    m_nbtiles = 0;
    offy = 0;
    for (j = 0; j < m_ny; j += tileSize) {
        tileSizeY = tileSize;
        if (offy + tileSizeY >= m_ny)
            tileSizeY = m_ny - offy;
        assert(tileSizeY <= tileSize);
        offx = 0;
        for (i = 0; i < m_nx; i += tileSize) {
            tileSizeX = tileSize;
            if (offx + tileSizeX >= m_nx)
                tileSizeX = m_nx - offx;
            assert(tileSizeX <= tileSize);
            m_tiles[m_nbtiles]->setPrt(m_prt);
            m_tiles[m_nbtiles++]->setExtend(tileSizeX, tileSizeY, m_nx, m_ny, offx, offy, m_dx);
            if (m_prt) {
                std::cout << "tsx " << tileSizeX << " tsy " << tileSizeY;
                std::cout << " ofx " << offx << " offy " << offy;
                std::cout << " nx " << m_nx << " m_ny " << m_ny << std::endl;
            }
            offx += tileSize;
        }
        offy += tileSize;
    }

    // Make links to neighbors
    int t = 0;
    for (j = 0; j < mortonH; j++) {
        for (i = 0; i < mortonW; i++, t++) {
            int tright = -1, tleft = -1, tup = -1, tdown = -1;
            Tile *pright = 0, *pleft = 0, *pup = 0, *pdown = 0;
            if (i < (mortonW - 1)) {
                tright = (*m_morton)(i + 1, j);
                pright = m_tiles[tright];
            }
            if (i > 0) {
                tleft = (*m_morton)(i - 1, j);
                pleft = m_tiles[tleft];
            }
            if (j < (mortonH - 1)) {
                tup = (*m_morton)(i, j + 1);
                pup = m_tiles[tup];
            }
            if (j > 0) {
                tdown = (*m_morton)(i, j - 1);
                pdown = m_tiles[tdown];
            }
            // std::cerr << t << " : ";
            // std::cerr << tleft << " ";
            // std::cerr << tright << " ";
            // std::cerr << tup << " ";
            // std::cerr << tdown << " ";
            // std::cerr << std::endl;
            m_tiles[t]->setVoisins(pleft, pright, pup, pdown);
        }
    }

// Create the shared buffers
#ifdef _OPENMP
    m_nbWorkItems = omp_get_max_threads();
#else
    m_nbWorkItems = 1;
#endif
    m_timerLoops = new double *[m_nbWorkItems + 1];

#ifdef _OPENMP
#pragma omp parallel for private(i) if (m_numa) schedule(static, 1)
#endif

    for (int32_t i = 0; i < m_nbWorkItems; i++) {
        m_timerLoops[i] = new double[LOOP_END];
        assert(m_timerLoops[i] != 0);
        memset(m_timerLoops[i], 0, LOOP_END * sizeof(double));
    }

    int32_t tileSizeTot = tileSize + 2 * m_ExtraLayer;
    m_buffers = new DeviceBuffers *[m_nbWorkItems];
    assert(m_buffers != 0);

    m_threadTimers = new Timers[m_nbWorkItems];
    assert(m_threadTimers != 0);

#ifdef _OPENMP
#pragma omp parallel for private(i) if (m_numa) schedule(static, 1)
#endif
    for (int32_t i = 0; i < m_nbWorkItems; i++) {
        // #pragma omp critical
        // {
        //    std::cerr << i << " attendu " << myThread() << std::endl << flush;
        // }
        m_buffers[i] = new DeviceBuffers(0, tileSizeTot, 0, tileSizeTot);
        assert(m_buffers[myThread()] != 0);
    }
// std::cerr << "Buffer cree" << std::endl;
#ifdef _OPENMP
#pragma omp parallel for private(i) if (m_numa) SCHEDULE
#endif
    for (int32_t t = 0; t < m_nbtiles; t++) {
        i = t;
        if (m_withMorton)
            i = m_mortonIdx[t];

        m_tiles[i]->setTimers(m_threadTimers);
        m_tiles[i]->initTile(m_uold);
        m_tiles[i]->initPhys(m_gamma, m_smallc, m_smallr, m_cfl, m_slope_type, m_nIterRiemann,
                             m_iorder, m_scheme);
        m_tiles[i]->setScan(X_SCAN);
    }

    if (ParallelInfo::mype() == 0 && m_stats > 0) {
        char hostn[1024];
        char cpuName[1024];
        memset(hostn, 0, 1024);
        memset(cpuName, 0, 1024);

        gethostname(hostn, 1023);
        std::cout << "HydroC starting run with " << m_nbtiles << " tiles";
        std::cout << " on " << hostn << std::endl;
        getCPUName(cpuName);
        std::cout << "CPU name" << cpuName << std::endl;
    }
    // exit(0);
}



// EOF
