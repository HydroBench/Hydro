#ifdef MPI_ON
#include <mpi.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

#include <cstring>
#include <cstdlib>
#include <unistd.h>
#include <cmath>
#include <cstdio>
#include <climits>
#include <cerrno>
#include <iostream>
#include <iomanip>

#include <strings.h>
#include <unistd.h>
#include <malloc.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <float.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <stdarg.h>
#include <fcntl.h>
#include <errno.h>

/*
 */
using namespace std;

#include "EnumDefs.hpp"
#include "Domain.hpp"
#include "Soa.hpp"
#include "cclock.h"

const long pageSize = 4 * 1024 * 1024;
typedef struct _textMarker {
	char t[64];
} TextMarker_t;

template < typename T > inline void lgrdstScal(const int f, const protectionMode_t mode, long &l, T * x)
{
	int byt = 0;
	switch (mode) {
	case PROT_LENGTH:{
			l += sizeof(T);
		};
		break;
	case PROT_WRITE:{
			byt = write(f, (x), sizeof(T));
		};
		break;
	case PROT_READ:{
			byt = read(f, (x), sizeof(T));
		};
		break;
	default:
		cerr << "Checkpoint: unknown mode" << endl;
		abort();
	}
}

template < typename T > inline void lgrdstArr(const int f, const protectionMode_t mode, long &l, T x)
{
	switch (mode) {
	case PROT_LENGTH:{
			l += x->getLengthByte();
		};
		break;
	case PROT_WRITE:{
			x->write(f);
		};
		break;
	case PROT_READ:{
			x->read(f);
		};
		break;
	default:
		cerr << "Checkpoint: unknown mode" << endl;
		abort();
	}
}

template < typename T > inline void lgrdstArrSimple(const int f, const protectionMode_t mode, long &l, T x, const int32_t lgr)
{
	int byt = 0;
	switch (mode) {
	case PROT_LENGTH:
		l += (sizeof(x) * lgr);
		break;
	case PROT_WRITE:
		byt = write(f, &(x), sizeof(x) * lgr);
		break;
	case PROT_READ:
		byt = read(f, &(x), sizeof(x) * lgr);
		break;
	default:
		cerr << "Checkpoint: unknown mode" << endl;
		abort();
	}
}

long Domain::protectScalars(const protectionMode_t mode, const int f)
{
	long l = 0;
	int m_GlobNx;

	lgrdstScal(f, mode, l, &m_globNx);
	// cerr << "wr/lu: m_globNx " << m_globNx << endl;
	lgrdstScal(f, mode, l, &m_globNy);
	// cerr << "wr/lu: m_globNy " << m_globNy << endl;
	lgrdstScal(f, mode, l, &m_npng);
	// cerr << "wr/lu: m_npng " << m_npng << endl;
	lgrdstScal(f, mode, l, &m_nvtk);
	// cerr << "wr/lu: m_nvtk " << m_nvtk << endl;
	lgrdstScal(f, mode, l, &m_tcur);
	// cerr << "wr/lu: m_tcur " << m_tcur << endl;
	lgrdstScal(f, mode, l, &m_dt);
	// cerr << "wr/lu: m_dt " << m_dt << endl;
	lgrdstScal(f, mode, l, &m_scan);
	// cerr << "wr/lu: m_scan " << m_scan << endl;
	lgrdstScal(f, mode, l, &m_iter);
	lgrdstScal(f, mode, l, &m_nbRun);
	lgrdstScal(f, mode, l, &m_elapsTotal);
	lgrdstScal(f, mode, l, &m_nextOutput);
	lgrdstScal(f, mode, l, &m_nextImage);
	lgrdstScal(f, mode, l, &m_nImage);
	lgrdstScal(f, mode, l, &m_nStepMax);
	// lgrdstScal(f, mode, l, &m_forceStop);
	return l;
}

long Domain::protectArrays(const protectionMode_t mode, const int f)
{
	long l = 0;
	int m_GlobNx;

	lgrdstArr(f, mode, l, m_uold);
	// lgrdstArrSimple(f, mode, l, m_mortonIdx, m_nbtiles);
	return l;
}

void Domain::writeProtectionVars(const int f)
{
	TextMarker_t magic, offmark, protmark, endprot;
	sprintf(protmark.t, "HYDROC BEGPROT %06d", m_myPe);
	sprintf(endprot.t, "HYDROC ENDPROT %06d", m_myPe);

	// Write of protection marker
	int byt = 0;
	byt = write(f, protmark.t, sizeof(protmark));
	protectScalars(PROT_WRITE, f);
	protectArrays(PROT_WRITE, f);
	byt = write(f, endprot.t, sizeof(endprot));
}

void Domain::readProtectionVars(const int f)
{
	int byt = 0;
	TextMarker_t protmark, endprot;
	TextMarker_t protmarkR, endprotR;
	sprintf(protmark.t, "HYDROC BEGPROT %06d", m_myPe);
	sprintf(endprot.t, "HYDROC ENDPROT %06d", m_myPe);

	// read of protection marker
	byt = read(f, protmarkR.t, sizeof(protmarkR));
	// cerr << "marqueur lu: " << protmarkR.t << endl;
	// cerr << "marqueur at: " << protmark.t << endl;
	assert(strcmp(protmarkR.t, protmark.t) == 0);	// crude protection corruption detection
	protectScalars(PROT_READ, f);
	protectArrays(PROT_READ, f);
	byt = read(f, endprotR.t, sizeof(endprotR));
	// cerr << "marqueur lu: " << endprotR.t << endl;
	// cerr << "marqueur at: " << endprot.t << endl;
	assert(strcmp(endprotR.t, endprot.t) == 0);	// crude protection corruption detection
}

void Domain::writeProtectionHeader(const int f)
{
#ifdef MPI_ON
	MPI_Request *requests;
	MPI_Status *status;
	MPI_Datatype mpiFormat = MPI_DOUBLE;
	int err = 0, reqcnt = 0;
#endif
	long l = 0, byt = 0;
	long *lgrs = (long *)calloc(m_nProc, sizeof(long));
	long *offsets = (long *)calloc(m_nProc, sizeof(long));
	long myOffset = 0;
	TextMarker_t magic, offmark, protmark, endprot;
	long headerLength = 0;

	sprintf(magic.t, "HYDROC CHECKPOINT");
	sprintf(endprot.t, "HYDROC ENDPROT %06d", m_myPe);
	sprintf(protmark.t, "HYDROC BEGPROT %06d", m_myPe);
	sprintf(offmark.t, "HYDROC OFFSETS TABLE");

	l += sizeof(endprot);
	l += sizeof(protmark);
	l += protectScalars(PROT_LENGTH, f);
	l += protectArrays(PROT_LENGTH, f);
	l = (l + pageSize - 1) / pageSize;	// nb of pages needed for this domain.
	l = l * pageSize;

	headerLength += sizeof(magic);
	headerLength += sizeof(offmark);
	headerLength += sizeof(long) * m_nProc;
	headerLength = (headerLength + pageSize - 1) / pageSize;	// nb of pages needed for the header.
	headerLength = headerLength * pageSize;

	// by default we have only one proc.
	myOffset = headerLength;
	offsets[0] = headerLength;

#ifdef MPI_ON
	if (m_nProc > 1) {
		requests = (MPI_Request *) calloc(m_nProc, sizeof(MPI_Request));
		status = (MPI_Status *) calloc(m_nProc, sizeof(MPI_Status));
		if (m_myPe == 0) {
			lgrs[0] = l;
			for (int i = 1; i < m_nProc; i++) {
				MPI_Irecv(&lgrs[i], 1, MPI_LONG, i, 987, MPI_COMM_WORLD, &requests[reqcnt]);
				reqcnt++;
			}
		} else {
			MPI_Isend(&l, 1, MPI_LONG, 0, 987, MPI_COMM_WORLD, &requests[reqcnt]);
		}
		err = MPI_Waitall(reqcnt, requests, status);
		assert(err == MPI_SUCCESS);
		// Here we know the size of each elements
		if (m_myPe == 0) {
			offsets[0] = headerLength;
			for (int i = 1; i < m_nProc; i++) {
				offsets[i] = lgrs[i - 1] + offsets[i - 1];
			}
		}
		reqcnt = 0;
		if (m_myPe == 0) {
			for (int i = 1; i < m_nProc; i++) {
				MPI_Isend(&offsets[i], 1, MPI_LONG, i, 988, MPI_COMM_WORLD, &requests[reqcnt]);
				reqcnt++;
			}
		} else {
			MPI_Irecv(&myOffset, 1, MPI_LONG, 0, 988, MPI_COMM_WORLD, &requests[reqcnt]);
			reqcnt++;
		}
		err = MPI_Waitall(reqcnt, requests, status);
		assert(err == MPI_SUCCESS);
		free(requests);
		free(status);
	}
#endif
	if (m_myPe == 0) {
		lseek(f, 0, SEEK_SET);
		// Write of magic number
		byt = write(f, magic.t, sizeof(magic));
		// Write of offset marker
		byt = write(f, offmark.t, sizeof(offmark));
		byt = write(f, offsets, sizeof(long) * m_nProc);
	}
	lseek(f, myOffset, SEEK_SET);
	free(offsets);
	free(lgrs);
}

void Domain::writeProtection()
{
	int needToStopGlob = true;
	int f = -1;
	char protName[256];
	errno = 0;

	sprintf(protName, "%s", "Continue.dump");

	if (m_myPe == 0) {
		cerr << " Opening " << protName << " for writing " << endl;
		f = open(protName, O_LARGEFILE | O_RDWR | O_CREAT, S_IRWXU);
	}
#ifdef MPI_ON
	MPI_Bcast(&needToStopGlob, 1, MPI_INT, 0, MPI_COMM_WORLD);
#endif
	if (m_nProc > 1) {
		if (m_myPe > 0) {
			f = open(protName, O_LARGEFILE | O_RDWR | O_CREAT, S_IRWXU);
		}
	}
	if (f == -1) {
		perror("writeProtection");
#ifdef MPI_ON
		MPI_Abort(MPI_COMM_WORLD, 1);
#else
		abort();
#endif
	}
#ifdef MPI_ON
	MPI_Bcast(&needToStopGlob, 1, MPI_INT, 0, MPI_COMM_WORLD);
#endif
	writeProtectionHeader(f);
	writeProtectionVars(f);
	close(f);
#ifdef MPI_ON
	MPI_Bcast(&needToStopGlob, 1, MPI_INT, 0, MPI_COMM_WORLD);
#endif
	if (m_myPe == 0)
		cerr << "Protection written" << endl;
}

void Domain::readProtectionHeader(const int f)
{
#ifdef MPI_ON
	MPI_Request *requests;
	MPI_Status *status;
	MPI_Datatype mpiFormat = MPI_DOUBLE;
	int err = 0, reqcnt = 0;
#endif
	int byt = 0;
	long *offsets = (long *)calloc(m_nProc, sizeof(long));
	long myOffset = 0;
	TextMarker_t magic, offmark, protmark, endmark;
	TextMarker_t magicR, offmarkR, protmarkR;
	long headerLength = 0;

	sprintf(magic.t, "HYDROC CHECKPOINT");
	sprintf(offmark.t, "HYDROC OFFSETS TABLE");
	sprintf(protmark.t, "HYDROC BEGPROT %06d", m_myPe);
	sprintf(endmark.t, "HYDROC ENDPROT %06d", m_myPe);

	headerLength += sizeof(magic);
	headerLength += sizeof(offmark);
	headerLength += sizeof(long) * m_nProc;
	headerLength = (headerLength + pageSize - 1) / pageSize;	// nb of pages needed for the header.
	headerLength = headerLength * pageSize;

	if (m_myPe == 0) {
		cerr << "Opening protection" << endl;
		lseek(f, 0, SEEK_SET);
		// Read magic number
		byt = read(f, magicR.t, sizeof(magicR));
		assert(strcmp(magicR.t, magic.t) == 0);	// crude protection corruption detection
		// Read offset marker
		byt = read(f, offmarkR.t, sizeof(offmarkR));
		assert(strcmp(offmarkR.t, offmark.t) == 0);	// crude protection corruption detection
		byt = read(f, offsets, sizeof(long) * m_nProc);
	}
	// by default we have only one proc.
	myOffset = offsets[0];

#ifdef MPI_ON
	if (m_nProc > 1) {
		requests = (MPI_Request *) calloc(m_nProc, sizeof(MPI_Request));
		status = (MPI_Status *) calloc(m_nProc, sizeof(MPI_Status));
		reqcnt = 0;
		if (m_myPe == 0) {
			for (int i = 1; i < m_nProc; i++) {
				MPI_Isend(&offsets[i], 1, MPI_LONG, i, 988, MPI_COMM_WORLD, &requests[reqcnt]);
				reqcnt++;
			}
		} else {
			MPI_Irecv(&myOffset, 1, MPI_LONG, 0, 988, MPI_COMM_WORLD, &requests[reqcnt]);
			reqcnt++;
		}
		err = MPI_Waitall(reqcnt, requests, status);
		assert(err == MPI_SUCCESS);
		free(requests);
		free(status);
	}
#endif
	lseek(f, myOffset, SEEK_SET);
	free(offsets);
}

bool Domain::isStopped()
{
	struct stat buf;
	return (stat("STOP", &buf) == 0);
}

bool Domain::StopComputation()
{
	if (m_myPe == 0) {
		int f = open("STOP", O_WRONLY | O_CREAT | O_TRUNC, S_IRWXU);
		if (f < 0) {
			perror("StopComputation");
#ifdef MPI_ON
			MPI_Abort(MPI_COMM_WORLD, 1);
#else
			abort();
#endif
		}
	}
	return false;
}

bool Domain::hasProtection()
{
	struct stat buf;
	return (stat("Continue.dump", &buf) == 0);
}

void Domain::readProtection()
{
	int f = -1;
	char protName[256];
	errno = 0;

	sprintf(protName, "%s", "Continue.dump");

	if (m_myPe == 0) {
		f = open(protName, O_LARGEFILE | O_RDONLY, S_IRWXU);
	}
	if (m_nProc > 1) {
#ifdef MPI_ON
		MPI_Barrier(MPI_COMM_WORLD);
#endif
		if (m_myPe > 0) {
			f = open(protName, O_LARGEFILE | O_RDWR);
		}
	}
	if (f == -1) {
		perror("readProtection");
#ifdef MPI_ON
		MPI_Abort(MPI_COMM_WORLD, 1);
#else
		abort();
#endif
	}
	readProtectionHeader(f);
	readProtectionVars(f);
	close(f);
}
