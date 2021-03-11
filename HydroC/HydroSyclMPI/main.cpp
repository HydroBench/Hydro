#include "Domain.hpp"
#include "ParallelInfo.hpp"

#ifdef MPI_ON
#include <mpi.h>
#endif

#include <assert.h>
#include <cmath>
#include <iostream>



using namespace std;

void validmatrix() {
    int p = 0;
    Matrix2<double> tst(3, 4);

    for (int j = 0; j < tst.getH(); j++) {
        for (int i = 0; i < tst.getW(); i++) {
            tst(i, j) = p++;
        }
    }
    std::cout << "tst" << tst;
    tst.swapDimOnly();
    std::cout << "tst" << tst;

    exit(1);
}

int main(int argc, char **argv) {

#ifdef WITHHBW
    hbw_set_policy(HBW_POLICY_PREFERRED);
// #pragma message "HBW policy set to PREFERRED"
#endif

    Domain *domain = new Domain(argc, argv);

    if (domain->isStopped()) {
#ifdef MPI_ON
        // #pragma message "MPI is activated"
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Finalize();
        cout << "Hydroc: computation already finished" << endl;
        exit(1);
#endif
    }
    domain->compute();
#ifdef MPI_ON
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    if (ParallelInfo::mype() == 0)
        cout << "Hydro: done." << endl;

#ifdef MPI_ON
    MPI_Finalize();
#endif
    delete domain;
    exit(0);
}
