//
// (C) Guillaume.Colin-de-Verdiere at CEA.Fr
//
#ifdef MPI_ON
#include <mpi.h>
#endif
//
#include "Timers.hpp"

Timers::Timers(void) {
    m_elaps = new double[LASTENTRY];
    m_vmin = new double[LASTENTRY];
    m_vmax = new double[LASTENTRY];
    m_vavg = new double[LASTENTRY];
    for (int i = 0; i < LASTENTRY; ++i) {
        m_elaps[i] = 0.0;
        m_vmin[i] = 0.0;
        m_vmax[i] = 0.0;
        m_vavg[i] = 0.0;
    }
}

// template <typename T>
// Timers::Timers() { }

// template <typename T>
Timers::~Timers() {
    delete[] m_elaps;
    delete[] m_vmin;
    delete[] m_vmax;
    delete[] m_vavg;
}

// template <typename T>
// Timers::Timers(const Timers & obj) { }

void Timers::getStats(void) {
    for (int i = 0; i < LASTENTRY; ++i) {
        Fname_t f = Fname_t(i);
        double lvmin = m_elaps[f];
        double lvmax = m_elaps[f];
        double lvavg = m_elaps[f];
#ifdef MPI_ON
        int n = 0;
        MPI_Comm_size(MPI_COMM_WORLD, &n);
        MPI_Allreduce(&m_elaps[f], &lvmin, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(&m_elaps[f], &lvmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(&m_elaps[f], &lvavg, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        lvavg /= n;
        // cerr << m_elaps[f] << " " << lvavg << endl; cerr.flush();
#else
        lvmin = lvmax = lvavg = m_elaps[f];
#endif
        m_vmin[f] = lvmin;
        m_vmax[f] = lvmax;
        m_vavg[f] = lvavg;
    }
}

const char *Timers::name(Fname_t f) {
    const char *lname;
    switch (f) {
    case SLOPE:
        lname = "slope        ";
        break;
    case TRACE:
        lname = "trace        ";
        break;
    case QLEFTR:
        lname = "qleftr       ";
        break;
    case COMPFLX:
        lname = "compflx      ";
        break;
    case UPDCVAR:
        lname = "updateconsvar";
        break;
    case GATHCVAR:
        lname = "gathercvar   ";
        break;
    case EOS:
        lname = "eos          ";
        break;
    case COMPDT:
        lname = "computedt    ";
        break;
    case CONSTPRIM:
        lname = "constoprim   ";
        break;
    case RIEMANN:
        lname = "riemann      ";
        break;
    case BOUNDINIT:
        lname = "boundary_init ";
        break;
    case BOUNDINITBW:
        lname = "boundary_init_bw";
        break;
    case BOUNDEXEC:
        lname = "boundary_process";
        break;
    case REDUCEMIN:
        lname = "reducemin";
        break;
    case REDUCEMAX:
        lname = "reducemax";
        break;
    case ALLTILECMP:
        lname = "allTileCmp";
        break;
    case SENDUOLD:
        lname = "sendUoldToDevice";
        break;


    case GETUOLD:
        lname = "getUoldFromDevice";
        break;

    case BANDWIDTH:
        lname = "-- bandwidth --";
        break;
    case WAITQUEUE:
        lname = "queueWait";
        break;
    case TILEOMP:
        lname = "-- tileomp --";
        break;
    case LASTENTRY:
        lname = "LASTENTRY";
        break;
    }
    return lname;
}

void Timers::print(void) {
    const char *lname;
    double sum = 0.;
    for (int i = 0; i < BANDWIDTH; ++i) {
        Fname_t f = Fname_t(i);
        lname = this->name(f);
        sum += m_elaps[f];
    }
    for (int i = 0; i < LASTENTRY; ++i) {
        if (i == TILEOMP)
            continue; // ignore this marker
        Fname_t f = Fname_t(i);
        lname = this->name(f);
        double percent = m_elaps[f] * 100.0 / sum;
        if (i > BANDWIDTH)
            percent = 0.0;
        if (i != BANDWIDTH) {
            printf("%-17s\t%lf\t%.1lf%%\n", lname, m_elaps[f], percent);
        } else {
            printf("%s\n", lname);
        }
    };
}

void Timers::printStats(void) {
    const char *lname;
    double sum = 0.;
    printf("%-17s\t%-8s\t%-8s\t%-8s\t%5s\t\t%5s\n", "Name", "average", "min", "max", "elaps",
           "%avg");
    for (int i = 0; i < BANDWIDTH; ++i) {
        if (i == TILEOMP)
            continue; // ignore this marker
        Fname_t f = Fname_t(i);
        lname = this->name(f);
        sum += m_elaps[f];
    }

    // safeguard
    if (sum == 0)
        sum = 1.0;

    for (int i = 0; i < LASTENTRY; ++i) {
        if (i == TILEOMP)
            continue; // ignore this marker
        Fname_t f = Fname_t(i);
        lname = this->name(f);
        double percent = m_vavg[f] * 100.0 / sum;
        if (i != BANDWIDTH) {
            if (i < BANDWIDTH) {
                printf("%-17s\t%lf\t%lf\t%lf\t%lf\t%.1lf\n", lname, m_vavg[f], m_vmin[f], m_vmax[f],
                       m_elaps[f], percent);
            } else {
                printf("%-17s\t%lf\t%lf\t%lf\n", lname, m_vavg[f], m_vmin[f], m_vmax[f]);
            }
        } else {
            printf("%s\n", lname);
        }
    };
}

// template <typename T>
Timers &Timers::operator=(const Timers &rhs) {
    if (this != &rhs) {
        for (int i = 0; i < LASTENTRY; ++i) {
            Fname_t f = Fname_t(i);
            m_elaps[f] = rhs.get(f);
        };
    }
    return *this;
}

// template <typename T>
Timers &Timers::operator+=(const Timers &rhs) {
    for (int i = 0; i < LASTENTRY; ++i) {
        Fname_t f = Fname_t(i);
        m_elaps[f] += rhs.get(f);
    };
    return *this;
}

// template <typename T>
// Timers & Timers::operator() (uint32_t i) const { }

// Instantion of the needed types for the linker
// template class Timers<the_type>;
// EOF
