#ifdef MPI_ON
#include <mpi.h>
#endif
#include <iostream>
#include <assert.h>
#include <cmath>
#include "Domain.hpp"

using namespace std;

void validmatrix()
{
	int p = 0;
	Matrix2 < double >tst(3, 4);

	for (int j = 0; j < tst.getH(); j++) {
		for (int i = 0; i < tst.getW(); i++) {
			tst(i, j) = p++;
		}
	}
	tst.printFormatted("tst");
	tst.swapDimOnly();
	tst.printFormatted("tst");

	exit(1);
}

int main(int argc, char **argv)
{
	Domain domain(argc, argv);

	if (domain.isStopped()) {
#ifdef MPI_ON
#pragma message "MPI is activated"
		MPI_Barrier(MPI_COMM_WORLD);
		MPI_Finalize();
		cout << "Hydroc: computation already finished" << endl;
		exit(0);
#endif
	}
	domain.compute();
#ifdef MPI_ON
	MPI_Barrier(MPI_COMM_WORLD);
#endif
//	if (domain.getMype() == 0) {
//		system("top -b -n1 -u coling");
//	}
#ifdef MPI_ON
	MPI_Finalize();
#endif
	// cerr << "End main " << domain.getMype() << endl;
	exit(0);
}
