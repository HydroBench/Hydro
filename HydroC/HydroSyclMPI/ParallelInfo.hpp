//
// ParallelInfo
// Give information about the parallel environnement MPI + SYCL
//
// A singleton
//

#ifndef PARALLEL_INFO_H
#define PARALLEL_INFO_H be

class ParallelInfoOpaque; // To be less opaque in implementation !


class ParallelInfo {
    int m_myPe;
    int m_nProc;
    int m_nWorkers;
    bool m_verbosity;

    ParallelInfoOpaque *m_opaque;

  protected:
    ParallelInfo() {
        m_verbosity = true;
        m_myPe = 0;
        m_nProc = 1;
        m_nWorkers = 1;
        m_opaque = nullptr;

    }

    static ParallelInfo &GetInstance() {
        static ParallelInfo theGlobalVariable;
        return theGlobalVariable;
    }

  public:
    ParallelInfo(ParallelInfo &) = delete;
    void operator=(const ParallelInfo &) = delete;

    static int mype() { return GetInstance().m_myPe; }
    static int nb_procs() { return GetInstance().m_nProc; }
    static int nb_workers() { return GetInstance().m_nWorkers; }

    static void init(int & argc, char  ** &argv, bool verbose=true);
    static void end();

    static ParallelInfoOpaque * extraInfos() { return GetInstance().m_opaque;}
};

#endif // PARALLEL_INFO_H