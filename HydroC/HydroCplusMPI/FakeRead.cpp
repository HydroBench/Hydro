//
// (C) Guillaume.Colin-de-Verdiere at CEA.Fr
//
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <cstdio>
#include <climits>
#include <cerrno>

#include <strings.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <malloc.h>
#include <sys/time.h>
#include <float.h>

//
#include "FakeRead.hpp"

// template <typename T> 
// FakeRead::FakeRead(void) { }

// template <typename T> 
FakeRead::FakeRead(long fSize = 2000000000, int rank = 0)
{
	size_t n = 0;
	char fName[356];
	int rc = 0;
	struct stat buf;
	int needFill = 0;

	m_rank = rank;
	m_bSize = 1024 * 1024;
	m_fSize = fSize;
	m_fd = 0;
	m_buf = (double *)malloc(m_bSize * sizeof(m_buf[0]));
	// create a random file
	sprintf(fName, "%s%06d.tmp", "FAKEFILE", m_rank);
	// fprintf(stderr, "FakeRead::FakeRead %s\n", fName); fflush(stderr);
	
	// check that the fake file exists, if so just reuse it to
	// save time
	rc = stat(fName, &buf);
	if (rc == -1) {
		// file not found, create it
		needFill = 1;
	}
	if (buf.st_size < m_fSize) {
		// what is on disk is too small, re-create it.
		needFill = 1;
	}

	if (needFill) {
		m_fd = fopen(fName, "w");
		assert(m_fd != 0);
		// Fill the file with some values
		for (int64_t i = 0; i < m_fSize; i += m_bSize) {
			int64_t towrite = m_bSize;
			// check boundary
			if ((i + m_bSize) >= m_fSize) {
				// remainder to write
				towrite = m_fSize % m_bSize;
			}
			for (int64_t j = 0; j < towrite; j++) {
				m_buf[j] = i * m_bSize + j;
			}
			// fprintf(stderr, "FakeRead::FakeRead %d %ld %ld %ld\n", towrite, i, fSize, (long) (fSize - i));
			errno = 0;
			rc = fwrite(m_buf, sizeof(m_buf[0]), towrite, m_fd);
			if (rc == -1) {
				perror("Error on FakeRead creation");
			}
		}
		errno = 0;
		rc = fflush(m_fd);
		if (rc) {
			perror("Error on FakeRead flush");
		}
		errno = 0;
		rc = fclose(m_fd);
		if (rc) {
			perror("Error on FakeRead flush");
		}
		sync();
	}
	// leave the file open for reading
	m_fd = fopen(fName, "r");
	assert(m_fd != 0);
	srandom(m_rank);
}

// template <typename T> 
FakeRead::~FakeRead()
{
	if (m_fd)
		fclose(m_fd);
	if (m_buf)
		free(m_buf);
}

int FakeRead::Read(int64_t lg)
{
	int rc = 0;
	// choose a random position to read the buffer (meaningless here)
	assert(lg < m_fSize);	// sanity check
	long int val = random();
	double vald = ((double)val / (double)RAND_MAX) * (m_fSize - lg);
	long pos = (long) vald;
	// if (m_rank == 0) fprintf(stderr, "--> %d\n", pos);
	fseek(m_fd, pos, SEEK_SET);
	// bornes !
	for (int64_t i = 0; i < lg; i += m_bSize) {
		int toread = m_bSize;
		// check boundary
		if ((i + m_bSize) >= lg) {
			// remainder to read
			toread = lg % m_bSize;
		}
		// fprintf(stderr, "FakeRead::Read %d\n", toread);
		errno = 0;
		rc = fread(m_buf, sizeof(m_buf[0]), toread, m_fd);
		if (rc == EOF) {
			perror("Error on FakeRead::Read fread");
		}
	}
	return 0;
}

// template <typename T> 
// FakeRead::FakeRead(const FakeRead & obj) { }

// template <typename T> 
// FakeRead & FakeRead::operator=(const FakeRead & rhs) { }

// template <typename T> 
// FakeRead & FakeRead::operator() (uint32_t i) { }

// template <typename T> 
// FakeRead & FakeRead::operator() (uint32_t i) const { }

// Instantion of the needed types for the linker
// template class FakeRead<the_type>;
//EOF
