#ifdef MPI_ON
#include <mpi.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

#if WITHPNG > 0
#include <png.h>
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
#include <assert.h>

using namespace std;

//
#include "EnumDefs.hpp"
#include "Domain.hpp"
#include "cclock.h"

static void abort_(const char *s, ...)
{
	va_list args;
	va_start(args, s);
	vfprintf(stderr, s, args);
	fprintf(stderr, "\n");
	va_end(args);
#ifdef MPI_ON
	MPI_Abort(MPI_COMM_WORLD, 123);
#else
	abort();
#endif
}

#define PIXRGBA 4

// template <typename T> 
void Domain::getMaxVarValues(real_t * mxP, real_t * mxD, real_t * mxUV)
{
	int32_t xmin, xmax, ymin, ymax, x, y;
	real_t ipmax = 0;
	real_t idmax = 0;
	real_t iuvmax = 0;
	Matrix2 < real_t > &puold = *(*m_uold) (IP_VAR);
	Matrix2 < real_t > &duold = *(*m_uold) (ID_VAR);
	Matrix2 < real_t > &uuold = *(*m_uold) (IU_VAR);
	Matrix2 < real_t > &vuold = *(*m_uold) (IV_VAR);
	getExtends(TILE_INTERIOR, xmin, xmax, ymin, ymax);

	ipmax = puold(xmin, ymin);
	idmax = duold(xmin, ymin);
	iuvmax = sqrt(uuold(xmin, ymin) * uuold(xmin, ymin) + vuold(xmin, ymin) * vuold(xmin, ymin));

	for (y = ymin; y < ymax; y++) {
		for (x = xmin; x < xmax; x++) {
			real_t v = puold(x, y);
			if (v > ipmax)
				ipmax = v;
		}
	}
	for (y = ymin; y < ymax; y++) {
		for (x = xmin; x < xmax; x++) {
			real_t v = duold(x, y);
			if (v > idmax)
				idmax = v;
		}
	}
	for (y = ymin; y < ymax; y++) {
		for (x = xmin; x < xmax; x++) {
			real_t v = sqrt(uuold(x, y) * uuold(x, y) + vuold(x, y) * vuold(x, y));
			if (v > iuvmax)
				iuvmax = v;
		}
	}
	*mxP = ipmax;
	*mxD = idmax;
	*mxUV = iuvmax;
}

void Domain::pngProcess(void)
{
	int32_t d, x, y;
	int32_t xmin, xmax, ymin, ymax;
	int32_t imgSizeX, imgSizeY;
	real_t ipmax = 0;
	real_t idmax = 0;
	real_t iuvmax = 0;
	int mySize;

	assert(m_uold != 0);
	Matrix2 < real_t > &puold = *(*m_uold) (IP_VAR);
	Matrix2 < real_t > &duold = *(*m_uold) (ID_VAR);
	Matrix2 < real_t > &uuold = *(*m_uold) (IU_VAR);
	Matrix2 < real_t > &vuold = *(*m_uold) (IV_VAR);

	m_shrink = 1;
	while ((m_globNx / m_shrink) > m_shrinkSize || (m_globNy / m_shrink) > m_shrinkSize) {
		m_shrink++;
	}

	if (m_myPe == 0) {
		getExtends(TILE_INTERIOR, xmin, xmax, ymin, ymax);
		imgSizeX = (m_globNx / m_shrink);
		imgSizeY = (m_globNy / m_shrink);
	} else {
		imgSizeX = (m_nx / m_shrink);
		imgSizeY = (m_ny / m_shrink);
	}

#if WITHPNG == 0
	/*
	   PPM file
	 */
	mySize = PIXRGBA * imgSizeX * imgSizeY;

	m_buffer = new uint8_t[mySize];
	assert(m_buffer != 0);
	memset(m_buffer, 0, mySize * sizeof(*m_buffer));
	if (m_nProc == 1) {
		getMaxVarValues(&ipmax, &idmax, &iuvmax);
		getExtends(TILE_INTERIOR, xmin, xmax, ymin, ymax);

		// cerr << "Processing final image -- fill " << m_shrink << endl;

		int32_t cury = 0;
		int32_t curx = 0;
		uint8_t *ptr = m_buffer;
		for (cury = 0; cury < imgSizeY; cury++) {
			for (curx = 0; curx < imgSizeX; curx++) {
				x = xmin + curx * m_shrink;
				y = ymin + cury * m_shrink;
				*(ptr++) = (uint8_t) fabs(255 * puold(x, y) / ipmax);
				*(ptr++) = (uint8_t) fabs(255 * duold(x, y) / idmax);
				*(ptr++) = (uint8_t) fabs(255 * sqrt(uuold(x, y) * uuold(x, y) + vuold(x, y) * vuold(x, y)) / iuvmax);
				*(ptr++) = 255;	// alpha unused here.
			}
		}
		// cerr << "Processing final image -- fill done" << endl;
	} else {
#ifdef MPI_ON
		uint8_t *bufferR = 0;
		// each tile computes its max values
		getMaxVarValues(&ipmax, &idmax, &iuvmax);
		// keep the max of the max
		ipmax = reduceMaxAndBcast(ipmax);
		idmax = reduceMaxAndBcast(idmax);
		iuvmax = reduceMaxAndBcast(iuvmax);
		// compute my own sub image
		getExtends(TILE_INTERIOR, xmin, xmax, ymin, ymax);
		int32_t curImgX = (xmax - xmin) / m_shrink;
		int32_t curImgY = (ymax - ymin) / m_shrink;
		int32_t cury = 0;
		int32_t curx = 0;
		// cerr << m_myPe << " remplir " << endl;
		uint8_t *ptr = m_buffer;
		for (cury = 0; cury < curImgY; cury++) {
			ptr = &m_buffer[cury * PIXRGBA * imgSizeX];
			for (curx = 0; curx < curImgX; curx++) {
				int r;
				double v1, v2;
				x = xmin + curx * m_shrink;
				y = ymin + cury * m_shrink;
				*(ptr++) = (int)fabs(255 * puold(x, y) / ipmax);
				*(ptr++) = (int)fabs(255 * duold(x, y) / idmax);
				v1 = uuold(x, y) * uuold(x, y);
				v2 = vuold(x, y) * vuold(x, y);
				r = (int)fabs(255 * sqrt(v1 + v2) / iuvmax);
				*(ptr++) = r;
				*(ptr++) = 255;	// 
			}
		}

		// if (m_myPe == 0) cerr << "PNG: receiving";
		for (d = 1; d < m_nProc; d++) {
			MPI_Request request[1];
			MPI_Status status[1];
			int nbreq = 0;
			int itsbox[MAXBOX_D];
			int err, size, xoff, yoff;
			// send our sub image
			size = ((xmax - xmin) / m_shrink) * ((ymax - ymin) / m_shrink) * PIXRGBA;
			if (m_myPe == 0) {
				// determine the box of the distant domain
				CalcSubSurface(0, m_globNx, 0, m_globNy, 0, m_nProc - 1, itsbox, d);
				xmin = xoff = itsbox[XMIN_D];
				xmax = itsbox[XMAX_D];
				ymin = yoff = itsbox[YMIN_D];
				ymax = itsbox[YMAX_D];
				size = ((xmax - xmin) / m_shrink) * ((ymax - ymin) / m_shrink) * PIXRGBA;
				// cerr << m_myPe << " Irecv " << endl;
				bufferR = new uint8_t[size];
				assert(bufferR != 0);
				memset(bufferR, 0, size * sizeof(*bufferR));
				MPI_Irecv(bufferR, size * sizeof(*bufferR), MPI_CHAR, d, 987 + d, MPI_COMM_WORLD, &request[0]);
				nbreq++;
			} else {
				if (m_myPe == d) {
					// cerr << m_myPe << " Isend " << endl;
					MPI_Isend(m_buffer, size * sizeof(*m_buffer), MPI_CHAR, 0, 987 + d, MPI_COMM_WORLD, &request[0]);
					nbreq++;
				}
			}
			if (nbreq) {
				err = MPI_Waitall(nbreq, request, status);
				assert(err == MPI_SUCCESS);
			}
			if (m_myPe == 0) {
				int32_t lcurx, lgr;
				int32_t DimgSizeX = ((xmax - xmin) / m_shrink);
				int32_t DimgSizeY = ((ymax - ymin) / m_shrink);
				int src = 0, dest = 0;
				curx = 0;
				cury = 0;
				lcurx = PIXRGBA * (xmin / m_shrink);
				lgr = PIXRGBA * DimgSizeX;
				for (cury = 0; cury < DimgSizeY; cury++) {
					dest = (ymin / m_shrink) + cury;
					dest *= (imgSizeX);
					dest += (xmin / m_shrink);
					dest *= PIXRGBA;
					memcpy(&m_buffer[dest], &bufferR[src], lgr * sizeof(*bufferR));
					src += lgr;
				}
				delete[]bufferR;
				bufferR = 0;
			}
		}
		// at this point the shrinked image can have unfilled pixels
		if ((m_shrink > 1) && (m_myPe == 0)) {
			ptr = m_buffer;
			for (cury = 0; cury < imgSizeY; cury++) {
				for (curx = 0; curx < imgSizeX; curx++) {
					ptr = &m_buffer[(cury * imgSizeX + curx) * PIXRGBA];
					if (ptr[3] < 255) {
						int cpt = 0;
						int Iptr[PIXRGBA];
						int extrem = 1;
						for (int c = 0; c < PIXRGBA; c++)
							Iptr[c] = 0;
						for (int vy = -extrem; vy < extrem + 1; vy++) {
							for (int vx = -extrem; vx < extrem + 1; vx++) {
								pngFillGap(curx, cury, curx + vx, cury + vy, Iptr, cpt, imgSizeX, imgSizeY);
							}
						}
						if (cpt) {
							for (int c = 0; c < PIXRGBA; c++) {
								ptr[c] = Iptr[c] / cpt;
							}
						}
					}
				}
			}
		}
#endif
	}
#else
	/*
	   PNG file 
	 */
	mySize = PIXRGBA * imgSizeX * imgSizeY;

	if (m_myPe == 0) {
		m_row_pointers = (png_bytep *) calloc(imgSizeY, sizeof(png_bytep));
		assert(m_row_pointers != 0);
		for (y = 0; y < imgSizeY; y++) {
			m_row_pointers[y] = (png_byte *) calloc(imgSizeX, PIXRGBA);
			assert(m_row_pointers[y] != 0);
		}
	} else {
		m_buffer = (png_byte *) calloc(mySize, 1);
		assert(m_buffer != 0);
		// memset(m_buffer, 255, mySize);
		m_row_pointers = (png_bytep *) calloc(imgSizeY, sizeof(png_bytep));
		assert(m_row_pointers != 0);
		for (y = 0; y < imgSizeY; y++)
			m_row_pointers[y] = &m_buffer[y * PIXRGBA * imgSizeX];
	}

	if (m_nProc == 1) {
		getMaxVarValues(&ipmax, &idmax, &iuvmax);
		getExtends(TILE_INTERIOR, xmin, xmax, ymin, ymax);

		int32_t cury = 0;
		int32_t curx = 0;
		for (cury = 0; cury < imgSizeY; cury++) {
			png_byte *row = m_row_pointers[cury];
			for (curx = 0; curx < imgSizeX; curx++) {
				x = xmin + curx * m_shrink;
				y = ymin + cury * m_shrink;
				png_byte *ptr = &(row[curx * PIXRGBA]);
				ptr[0] = (png_byte) fabs(255 * puold(x, y) / ipmax);
				ptr[1] = (png_byte) fabs(255 * duold(x, y) / idmax);
				ptr[2] = (png_byte) fabs(255 * sqrt(uuold(x, y) * uuold(x, y) + vuold(x, y) * vuold(x, y)) / iuvmax);
				ptr[3] = 255;
			}
		}
	} else {
#ifdef MPI_ON
		// each tile computes its max values
		getMaxVarValues(&ipmax, &idmax, &iuvmax);
		// keep the max of the max
		ipmax = reduceMaxAndBcast(ipmax);
		idmax = reduceMaxAndBcast(idmax);
		iuvmax = reduceMaxAndBcast(iuvmax);
		// compute my own sub image
		getExtends(TILE_INTERIOR, xmin, xmax, ymin, ymax);
		int32_t curImgX = (xmax - xmin) / m_shrink;
		int32_t curImgY = (ymax - ymin) / m_shrink;
		int32_t cury = 0;
		int32_t curx = 0;
		// cerr << m_myPe << " remplir " << endl;
		for (cury = 0; cury < curImgY; cury++) {
			png_byte *row = m_row_pointers[cury];
			for (curx = 0; curx < curImgX; curx++) {
				int r;
				double v1, v2;
				png_byte *ptr = &(row[curx * PIXRGBA]);
				x = xmin + curx * m_shrink;
				y = ymin + cury * m_shrink;
				ptr[0] = (int)fabs(255 * puold(x, y) / ipmax);
				ptr[1] = (int)fabs(255 * duold(x, y) / idmax);
				v1 = uuold(x, y) * uuold(x, y);
				v2 = vuold(x, y) * vuold(x, y);
				r = (int)fabs(255 * sqrt(v1 + v2) / iuvmax);
				ptr[2] = r;
				ptr[3] = 255;
			}
		}

		// if (m_myPe == 0) cerr << "PNG: receiving";
		for (d = 1; d < m_nProc; d++) {
			MPI_Request request[1];
			MPI_Status status[1];
			int nbreq = 0;
			int itsbox[MAXBOX_D];
			int err, size, xoff, yoff;
			// send our sub image
			size = ((xmax - xmin) / m_shrink) * ((ymax - ymin) / m_shrink) * PIXRGBA;
			if (m_myPe == 0) {
				// determine the box of the distant domain
				CalcSubSurface(0, m_globNx, 0, m_globNy, 0, m_nProc - 1, itsbox, d);
				xmin = xoff = itsbox[XMIN_D];
				xmax = itsbox[XMAX_D];
				ymin = yoff = itsbox[YMIN_D];
				ymax = itsbox[YMAX_D];
				size = ((xmax - xmin) / m_shrink) * ((ymax - ymin) / m_shrink) * PIXRGBA;
				// cerr << m_myPe << " Irecv " << endl;
				m_buffer = (png_byte *) malloc(size);
				assert(m_buffer != 0);
				MPI_Irecv(m_buffer, size, MPI_CHAR, d, 987 + d, MPI_COMM_WORLD, &request[0]);
				nbreq++;
			} else {
				if (m_myPe == d) {
					// cerr << m_myPe << " Isend " << endl;
					MPI_Isend(m_buffer, size, MPI_CHAR, 0, 987 + d, MPI_COMM_WORLD, &request[0]);
					nbreq++;
				}
			}
			if (nbreq) {
				err = MPI_Waitall(nbreq, request, status);
				assert(err == MPI_SUCCESS);
			}
			if (m_myPe == 0) {
				int32_t lcurx, lgr;
				int32_t DimgSizeX = ((xmax - xmin) / m_shrink);
				int32_t DimgSizeY = ((ymax - ymin) / m_shrink);
				curx = 0;
				cury = 0;
				lcurx = PIXRGBA * (xmin / m_shrink);
				lgr = PIXRGBA * DimgSizeX;
				for (cury = 0; cury < DimgSizeY; cury++) {
					y = (ymin / m_shrink) + cury;
					png_byte *row = m_row_pointers[y];
					// cerr << " memcpy " << y << " " << lcurx << " " << lgr << endl;
					memcpy(&row[lcurx], &m_buffer[cury * lgr], lgr);
				}
				free(m_buffer);
				m_buffer = 0;
			}
		}
		// at this point the shrinked image can have unfilled pixels
		if ((m_shrink > 1) && (m_myPe == 0)) {
			for (cury = 0; cury < imgSizeY; cury++) {
				for (curx = 0; curx < imgSizeX; curx++) {
					png_byte *row = m_row_pointers[cury];
					png_byte *ptr = &(row[curx * PIXRGBA]);
					if (ptr[PIXRGBA - 1] < 255) {
						int cpt = 0;
						int Iptr[PIXRGBA];
						int extrem = 1;
						for (int c = 0; c < PIXRGBA; c++)
							Iptr[c] = 0;
						for (int vy = -extrem; vy < extrem + 1; vy++) {
							for (int vx = -extrem; vx < extrem + 1; vx++) {
								pngFillGap(curx, cury, curx + vx, cury + vy, Iptr, cpt, imgSizeX, imgSizeY);
							}
						}
						if (cpt) {
							for (int c = 0; c < PIXRGBA; c++) {
								ptr[c] = Iptr[c] / cpt;
							}
						}
					}
				}
			}
		}
#endif
	}
#endif
}

void Domain::pngFillGap(int curx, int cury, int nx, int ny, int Iptr[4], int &cpt, int32_t imgSizeX, int32_t imgSizeY)
{
#if WITHPNG > 0
	png_byte *nrow, *nptr;
	if ((nx >= 0) && (nx < imgSizeX) && (ny >= 0) && (ny < imgSizeY)) {
		nrow = m_row_pointers[ny];
		nptr = &(nrow[nx * PIXRGBA]);
		if (nptr[PIXRGBA - 1] == 255) {
			for (int c = 0; c < PIXRGBA; c++) {
				Iptr[c] += nptr[c];
			}
			cpt++;
		}
	}
#else
	uint8_t *nptr, *nrow;
	if ((nx >= 0) && (nx < imgSizeX) && (ny >= 0) && (ny < imgSizeY)) {
		nrow = &(m_buffer[ny * imgSizeX * PIXRGBA]);
		nptr = &(nrow[nx * PIXRGBA]);
		if (nptr[3] == 255) {
			for (int c = 0; c < PIXRGBA; c++) {
				Iptr[c] += nptr[c];
			}
			cpt++;
		}
	}
#endif
}

void Domain::pngWriteFile(char *name)
{
	int32_t cury = 0;
	int32_t curx = 0;
	int imgSizeX = m_globNx;
	int imgSizeY = m_globNy;
	imgSizeX = m_globNx / m_shrink;
	imgSizeY = m_globNy / m_shrink;

#if WITHPNG == 0
	if (m_myPe == 0) {
		// cerr << "shrink factor " << m_shrink << " " << name << endl;
		m_fp = fopen(name, "w");
#define BINARY 1
#if BINARY == 1
		fprintf(m_fp, "P6\n");
		fprintf(m_fp, "%d %d\n", imgSizeX, imgSizeY);
		fprintf(m_fp, "255\n");
		uint8_t *ptr = m_buffer;
		for (cury = 0; cury < imgSizeY; cury++) {
			for (curx = 0; curx < imgSizeX; curx++) {
				fwrite(ptr, PIXRGBA - 1, 1, m_fp);
				ptr += (PIXRGBA);
			}
		}
		fprintf(m_fp, "#EOF\n");
#else
		fprintf(m_fp, "P3\n");
		fprintf(m_fp, "%d %d\n", imgSizeX, imgSizeY);
		fprintf(m_fp, "255\n");
		uint8_t *ptr = m_buffer;
		for (cury = 0; cury < imgSizeY; cury++) {
			for (curx = 0; curx < imgSizeX; curx++) {
				fprintf(m_fp, "%d %d %d  ", *ptr++, *ptr++, *ptr++);
				ptr++;
			}
			fprintf(m_fp, "\n");
		}
		fprintf(m_fp, "#EOF\n");
#endif
	}
#else
	png_byte color_type = PNG_COLOR_TYPE_RGBA;
	png_byte bit_depth = 8;

	/* create file */
	if (m_myPe == 0) {
		// cerr << "shrink factor " << m_shrink << " " << name << endl;
		m_fp = fopen(name, "w");
		if (!m_fp)
			abort_("[write_png_file] File %s could not be opened for writing", name);

		/* initialize stuff */
		m_png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

		if (!m_png_ptr)
			abort_("[write_png_file] png_create_write_struct failed");

		m_info_ptr = png_create_info_struct(m_png_ptr);
		if (!m_info_ptr)
			abort_("[write_png_file] png_create_info_struct failed");

		// if (setjmp(png_jmpbuf(m_png_ptr)))
		//      abort_("[write_png_file] Error during init_io");

		png_init_io(m_png_ptr, m_fp);
		/* write header */
		// if (setjmp(png_jmpbuf(m_png_ptr)))
		//      abort_("[write_png_file] Error during writing header");

		png_set_IHDR(m_png_ptr, m_info_ptr, imgSizeX, imgSizeY, bit_depth, color_type, PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);

		png_write_info(m_png_ptr, m_info_ptr);
		// if (m_myPe == 0) cerr << "Header of final image written " << m_globNx << " " << m_globNy << endl;
	}
#ifdef MPI_ON
	MPI_Barrier(MPI_COMM_WORLD);
#endif
#endif
}

void Domain::pngCloseFile(void)
{
	int32_t y;
	int imgSizeX, imgSizeY;
#if WITHPNG == 0
	if (m_myPe == 0) {
		fclose(m_fp);
		m_fp = 0;
	}
#ifdef MPI_ON
	MPI_Barrier(MPI_COMM_WORLD);
	delete[]m_buffer;
	m_buffer = 0;
#endif
#else
	if (m_myPe == 0) {
		png_bytep *rptrs;
		// cerr << "Write PNG image "; cerr.flush();
		png_write_image(m_png_ptr, m_row_pointers);
		/* end write */
		// if (setjmp(png_jmpbuf(m_png_ptr)))
		//      abort_("[write_png_file] Error during end of write");

		png_write_end(m_png_ptr, NULL);
		png_destroy_write_struct(&m_png_ptr, &m_info_ptr);
		fclose(m_fp);
		// cerr << "done." << endl;
	}
#ifdef MPI_ON
	MPI_Barrier(MPI_COMM_WORLD);
#endif
	/* cleanup heap allocation */
	if (m_myPe == 0) {
		for (y = 0; y < (m_globNy / m_shrink); y++)
			free(m_row_pointers[y]);
		free(m_row_pointers);
	} else {
		free(m_row_pointers);
	}

#endif
}

//EOF
