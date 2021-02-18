//
// (C) Guillaume.Colin-de-Verdiere at CEA.Fr
//
#include <assert.h>
#include <float.h>
#include <limits.h>
#include <malloc.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef MPI
#include <mpi.h>
#endif

#if WITHPNG > 0
#include <png.h>
#endif

//
#include "Image.h"
#include "SplitSurface.h"

#if WITHPNG > 0
png_structp m_png_ptr;
png_infop m_info_ptr;
png_bytep *m_row_pointers;
png_byte *m_buffer;
#else
uint8_t *m_buffer;
#endif

#define IHU(i, j, v) ((i) + Hnxt * ((j) + Hnyt * (v)))
// real_t uold[Hnvar * Hnxt * Hnyt]

static real_t reduceMaxAndBcast(real_t dt) {
    real_t dtmax = dt;
#ifdef MPI
    if (sizeof(real_t) == sizeof(double)) {
        MPI_Allreduce(&dt, &dtmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        // MPI_Bcast(&dtmax, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    } else {
        MPI_Allreduce(&dt, &dtmax, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
        // MPI_Bcast(&dtmax, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    }
#endif
    return dtmax;
}

static void abort_(const char *s, ...) {
    va_list args;
    va_start(args, s);
    vfprintf(stderr, s, args);
    fprintf(stderr, "\n");
    va_end(args);
#ifdef MPI
    MPI_Abort(MPI_COMM_WORLD, 123);
#else
    abort();
#endif
}

#define PIXRGBA 4

void getMaxVarValues(real_t *mxP, real_t *mxD, real_t *mxUV, hydroparam_t *H, hydrovar_t *Hv) {
    int32_t xmin = H->imin, xmax = H->imax, ymin = H->jmin, ymax = H->jmax;
    int32_t x, y;
    real_t ipmax = 0;
    real_t idmax = 0;
    real_t iuvmax = 0;
    real_t *uold = Hv->uold;
    int32_t Hnxt = H->nxt, Hnyt = H->nyt;

    ipmax = uold[IHV(xmin, ymin, IP)];
    idmax = uold[IHV(xmin, ymin, ID)];
    iuvmax = sqrt(uold[IHV(xmin, ymin, IU)] * uold[IHV(xmin, ymin, IU)] +
                  uold[IHV(xmin, ymin, IV)] * uold[IHV(xmin, ymin, IV)]);

    for (y = ymin; y < ymax; y++) {
        for (x = xmin; x < xmax; x++) {
            real_t v = uold[IHV(x, y, IP)];
            if (v > ipmax)
                ipmax = v;
        }
    }
    for (y = ymin; y < ymax; y++) {
        for (x = xmin; x < xmax; x++) {
            real_t v = uold[IHV(x, y, ID)];
            if (v > idmax)
                idmax = v;
        }
    }
    for (y = ymin; y < ymax; y++) {
        for (x = xmin; x < xmax; x++) {
            real_t v = sqrt(uold[IHV(x, y, IU)] * uold[IHV(x, y, IU)] +
                            uold[IHV(x, y, IV)] * uold[IHV(x, y, IV)]);
            if (v > iuvmax)
                iuvmax = v;
        }
    }
    *mxP = ipmax;
    *mxD = idmax;
    *mxUV = iuvmax;
}

void pngProcess(hydroparam_t *H, hydrovar_t *Hv) {
    int32_t d, x, y;
    int32_t xmin = H->imin + ExtraLayer, xmax = H->imax - ExtraLayer;
    int32_t ymin = H->jmin + ExtraLayer, ymax = H->jmax - ExtraLayer;
    int32_t imgSizeX, imgSizeY;
    real_t ipmax = 0;
    real_t idmax = 0;
    real_t iuvmax = 0;
    int mySize;
    real_t *uold = Hv->uold;
    int32_t Hnxt = H->nxt, Hnyt = H->nyt;

    H->shrink = 1;
    while ((H->globnx / H->shrink) > H->shrinkSize || (H->globny / H->shrink) > H->shrinkSize) {
        H->shrink++;
    }

    if (H->mype == 0) {
        imgSizeX = (H->globnx / H->shrink);
        imgSizeY = (H->globny / H->shrink);
    } else {
        imgSizeX = (H->nx / H->shrink);
        imgSizeY = (H->ny / H->shrink);
    }

    /*
       PPM file
     */
    mySize = PIXRGBA * imgSizeX * imgSizeY;

    m_buffer = (uint8_t *)malloc(mySize * sizeof(*m_buffer));
    assert(m_buffer != 0);
    memset(m_buffer, 0, mySize * sizeof(*m_buffer));
    if (H->nproc == 1) {
        getMaxVarValues(&ipmax, &idmax, &iuvmax, H, Hv);
        // cerr << "Processing final image -- fill " << m_shrink << endl;

        int32_t cury = 0;
        int32_t curx = 0;
        uint8_t *ptr = m_buffer;
        for (cury = 0; cury < imgSizeY; cury++) {
            for (curx = 0; curx < imgSizeX; curx++) {
                uint8_t vr, vp, vd;
                double v1, v2;
                x = xmin + curx * H->shrink;
                y = ymin + cury * H->shrink;
		vp = (uint8_t)fabs(255 * uold[IHV(x, y, IP)] / ipmax);
		vd = (uint8_t)fabs(255 * uold[IHV(x, y, ID)] / idmax);
		v1 = uold[IHV(x, y, IU)] * uold[IHV(x, y, IU)];
		v2 = uold[IHV(x, y, IV)] * uold[IHV(x, y, IV)];
		vr = (uint8_t)fabs(255 * sqrt(v1 + v2) / iuvmax);
                *(ptr++) = vd; // vp;
                *(ptr++) = vd; // vd;
                *(ptr++) = vd; // vr;
                *(ptr++) = 255; // alpha unused here.
            }
        }
        // cerr << "Processing final image -- fill done" << endl;
    } else {
#ifdef MPI
        uint8_t *bufferR = 0;
        // each tile computes its max values
        getMaxVarValues(&ipmax, &idmax, &iuvmax, H, Hv);
        // keep the max of the max
        ipmax = reduceMaxAndBcast(ipmax);
        idmax = reduceMaxAndBcast(idmax);
        iuvmax = reduceMaxAndBcast(iuvmax);
        // compute my own sub image
        int32_t curImgX = (xmax - xmin) / H->shrink;
        int32_t curImgY = (ymax - ymin) / H->shrink;
        int32_t cury = 0;
        int32_t curx = 0;
        // cerr << m_myPe << " remplir " << endl;
        uint8_t *ptr = m_buffer;
        for (cury = 0; cury < curImgY; cury++) {
            ptr = &m_buffer[cury * PIXRGBA * imgSizeX];
            for (curx = 0; curx < curImgX; curx++) {
                uint8_t vr, vp, vd;
                double v1, v2;
                x = xmin + curx * H->shrink;
                y = ymin + cury * H->shrink;
		vp = (uint8_t)fabs(255 * uold[IHV(x, y, IP)] / ipmax);
		vd = (uint8_t)fabs(255 * uold[IHV(x, y, ID)] / idmax);
                v1 = uold[IHV(x, y, IU)] * uold[IHV(x, y, IU)];
                v2 = uold[IHV(x, y, IV)] * uold[IHV(x, y, IV)];
                vr = (uint8_t) fabs(255 * sqrt(v1 + v2) / iuvmax);
                *(ptr++) = vp; // vp;
                *(ptr++) = vd; // vd;
                *(ptr++) = vr; // vr;
                *(ptr++) = 255; //
            }
        }

        // if (m_myPe == 0) cerr << "PNG: receiving";
	// fprintf(stderr, "[dom=%d x %d X %d y %d Y %d]\n", H->mype, xmin, xmax, ymin, ymax);
        for (d = 1; d < H->nproc; d++) {
            MPI_Request request[1];
            MPI_Status status[1];
            int nbreq = 0;
            int itsbox[MAX_BOX];
            int err, size, xoff, yoff;
            // send our sub image
            size = ((xmax - xmin) / H->shrink) * ((ymax - ymin) / H->shrink) * PIXRGBA;
            if (H->mype == 0) {
                // determine the box of the distant domain
                CalcSubSurface(0, H->globnx, 0, H->globny, 0, H->nproc - 1, 0, itsbox, d, 0);
                xmin = xoff = itsbox[XMIN_BOX];
                xmax = itsbox[XMAX_BOX];
                ymin = yoff = itsbox[YMIN_BOX];
                ymax = itsbox[YMAX_BOX];
		// fprintf(stderr, "<dom=%d x %d X %d y %d Y %d>\n", d, xmin, xmax, ymin, ymax);
                size = ((xmax - xmin) / H->shrink) * ((ymax - ymin) / H->shrink) * PIXRGBA;
                // cerr << m_myPe << " Irecv " << endl;
                bufferR = (uint8_t *)malloc(size * sizeof(*bufferR));
                assert(bufferR != 0);
                memset(bufferR, 0, size * sizeof(*bufferR));
                MPI_Irecv(bufferR, size * sizeof(*bufferR), MPI_CHAR, d, 9980 + d, MPI_COMM_WORLD, &request[0]);
		// fprintf(stderr, "Reception de %d octets, tag = %d\n", size, 9980 + d);
                nbreq++;
            } else {
                if (H->mype == d) {
                    // cerr << m_myPe << " Isend " << endl;
		    // fprintf(stderr, "Envoi de %d octets, tag = %d\n", size, 9980 + d);
                    MPI_Isend(m_buffer, size * sizeof(*m_buffer), MPI_CHAR, 0, 9980 + d, MPI_COMM_WORLD, &request[0]);
                    nbreq++;
                }
            }
	    // MPI_Barrier(MPI_COMM_WORLD);
	    if (nbreq) {
                err = MPI_Waitall(nbreq, request, status);
                assert(err == MPI_SUCCESS);
            }
            if (H->mype == 0) {
                int32_t lcurx, lgr;
                int32_t DimgSizeX = ((xmax - xmin) / H->shrink);
                int32_t DimgSizeY = ((ymax - ymin) / H->shrink);
                int src = 0, dest = 0;
                curx = 0;
                cury = 0;
                lcurx = PIXRGBA * (xmin / H->shrink);
                lgr = PIXRGBA * DimgSizeX;
                for (cury = 0; cury < DimgSizeY; cury++) {
                    dest = (ymin / H->shrink) + cury;
                    dest *= (imgSizeX);
                    dest += (xmin / H->shrink);
                    dest *= PIXRGBA;
                    memcpy(&m_buffer[dest], &bufferR[src], lgr * sizeof(*bufferR));
                    src += lgr;
                }
                free(bufferR);
                bufferR = 0;
            }
        }
        // at this point the shrinked image can have unfilled pixels
        if ((H->shrink > 1) && (H->mype == 0)) {
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
                                pngFillGap(curx, cury, curx + vx, cury + vy, Iptr, &cpt, imgSizeX, imgSizeY);
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
#endif // MPI
    }
}

void pngFillGap(int curx, int cury, int nx, int ny, int Iptr[4], int *cpt, int32_t imgSizeX,
                int32_t imgSizeY) {
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
}

void pngWriteFile(char *name, hydroparam_t *H) {
    int32_t cury = 0;
    int32_t curx = 0;
    int imgSizeX = H->globnx;
    int imgSizeY = H->globny;
    imgSizeX = imgSizeX / H->shrink;
    imgSizeY = imgSizeY / H->shrink;

    // Open image file
    if (H->mype == 0) {
        char *imPrefix = getenv("HYDROC_IMG_PREFIX");
        char imgname[1024];
        strcpy(imgname, name);
        if (imPrefix == NULL) {
            H->fp = fopen(name, "w");
        } else {
            sprintf(imgname, "%s%s", imPrefix, name);
            H->fp = fopen(imgname, "w");
        }
        if (!H->fp) {
#ifdef MPI
            fprintf(stderr, "File %s could not be opened for writing\n", imgname);
            MPI_Abort(MPI_COMM_WORLD, 1);
#else
            abort_("File %s could not be opened for writing", imgname);
#endif
        }
#define BINARY 0
#if BINARY == 1
        fprintf(H->fp, "P6\n");
        fprintf(H->fp, "%d %d\n", imgSizeX, imgSizeY);
        fprintf(H->fp, "255\n");
        uint8_t *ptr = m_buffer;
        for (cury = 0; cury < imgSizeY; cury++) {
            for (curx = 0; curx < imgSizeX; curx++) {
                fwrite(ptr, PIXRGBA - 1, 1, H->fp);
                ptr += (PIXRGBA);
            }
        }
        fprintf(H->fp, "#EOF\n");
#else  // BINARY != 1
        fprintf(H->fp, "P3\n");
        fprintf(H->fp, "%d %d\n", imgSizeX, imgSizeY);
        fprintf(H->fp, "255\n");
        uint8_t *ptr = m_buffer;
        for (cury = 0; cury < imgSizeY; cury++) {
            for (curx = 0; curx < imgSizeX; curx++) {
                fprintf(H->fp, "%d %d %d  ", *ptr, *(ptr+1), *(ptr+2));
                ptr+=(PIXRGBA);
            }
            fprintf(H->fp, "\n");
        }
        fprintf(H->fp, "#EOF\n");
#endif // BINARY != 1
    }
#ifdef MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif
}

void pngCloseFile(hydroparam_t *H) {
    int32_t y;
    int imgSizeX, imgSizeY;
    if (H->mype == 0) {
        fclose(H->fp);
        H->fp = 0;
    }
#ifdef MPI
    MPI_Barrier(MPI_COMM_WORLD);
    free(m_buffer);
    m_buffer = 0;
#endif
}

void DumpImage(int n, hydroparam_t *H, hydrovar_t *Hv) {
    char pngName[256];
    pngProcess(H, Hv);
#if WITHPNG > 0
    sprintf(pngName, "%s_%06d.png", "IMAGE", n);
#else
    sprintf(pngName, "%s_%06d.ppm", "IMAGE", n);
#endif
    pngWriteFile(pngName, H);
    pngCloseFile(H);
}
// EOF
