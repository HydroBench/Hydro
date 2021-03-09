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

//
#include "Image.h"
#include "SplitSurface.h"

static png_structp m_png_ptr;
static png_infop m_info_ptr;
static png_bytep *m_row_pointers;
static png_byte *m_buffer;

#define IHU(i, j, v) ((i) + Hnxt * ((j) + Hnyt * (v)))

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
       PNG file
     */
    mySize = PIXRGBA * imgSizeX * imgSizeY;

    if (H->mype == 0) {
        m_row_pointers = (png_bytep *)calloc(imgSizeY, sizeof(png_bytep));
        assert(m_row_pointers != 0);
        for (y = 0; y < imgSizeY; y++) {
            m_row_pointers[y] = (png_byte *)calloc(imgSizeX, PIXRGBA);
            assert(m_row_pointers[y] != 0);
        }
    } else {
        m_buffer = (png_byte *)calloc(mySize, 1);
        assert(m_buffer != 0);
        // memset(m_buffer, 255, mySize);
        m_row_pointers = (png_bytep *)calloc(imgSizeY, sizeof(png_bytep));
        assert(m_row_pointers != 0);
        for (y = 0; y < imgSizeY; y++)
            m_row_pointers[y] = &m_buffer[y * PIXRGBA * imgSizeX];
    }

    if (H->nproc  == 1) {
        getMaxVarValues(&ipmax, &idmax, &iuvmax, H, Hv);

        int32_t cury = 0;
        int32_t curx = 0;
        for (cury = 0; cury < imgSizeY; cury++) {
            png_byte *row = m_row_pointers[cury];
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
                png_byte *ptr = &(row[curx * PIXRGBA]);
                ptr[0] = vp;
                ptr[1] = vd;
                ptr[2] = vr;
                ptr[3] = 255;
            }
        }
    } else {
#ifdef MPI
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
        // cerr << H->mype << " remplir " << endl;
        for (cury = 0; cury < curImgY; cury++) {
            png_byte *row = m_row_pointers[cury];
            for (curx = 0; curx < curImgX; curx++) {
                int r;
                uint8_t vr, vp, vd;
                double v1, v2;
                png_byte *ptr = &(row[curx * PIXRGBA]);
                x = xmin + curx * H->shrink;
                y = ymin + cury * H->shrink;
		vp = (uint8_t)fabs(255 * uold[IHV(x, y, IP)] / ipmax);
		vd = (uint8_t)fabs(255 * uold[IHV(x, y, ID)] / idmax);
                v1 = uold[IHV(x, y, IU)] * uold[IHV(x, y, IU)];
                v2 = uold[IHV(x, y, IV)] * uold[IHV(x, y, IV)];
                vr = (uint8_t) fabs(255 * sqrt(v1 + v2) / iuvmax);
                ptr[0] = vp;
                ptr[1] = vd;
                ptr[2] = vr;
                ptr[3] = 255;
            }
        }

        // if (H->mype == 0) cerr << "PNG: receiving";
        for (d = 1; d < H->nproc ; d++) {
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
                size = ((xmax - xmin) / H->shrink) * ((ymax - ymin) / H->shrink) * PIXRGBA;
                // cerr << H->mype << " Irecv " << endl;
                m_buffer = (png_byte *)malloc(size);
                assert(m_buffer != 0);
                MPI_Irecv(m_buffer, size, MPI_CHAR, d, 9980 + d, MPI_COMM_WORLD, &request[0]);
                nbreq++;
            } else {
                if (H->mype == d) {
                    // cerr << H->mype << " Isend " << endl;
                    MPI_Isend(m_buffer, size, MPI_CHAR, 0, 9980 + d, MPI_COMM_WORLD, &request[0]);
                    nbreq++;
                }
            }
            if (nbreq) {
                err = MPI_Waitall(nbreq, request, status);
                assert(err == MPI_SUCCESS);
            }
            if (H->mype == 0) {
                int32_t lcurx, lgr;
                int32_t DimgSizeX = ((xmax - xmin) / H->shrink);
                int32_t DimgSizeY = ((ymax - ymin) / H->shrink);
                curx = 0;
                cury = 0;
                lcurx = PIXRGBA * (xmin / H->shrink);
                lgr = PIXRGBA * DimgSizeX;
                for (cury = 0; cury < DimgSizeY; cury++) {
                    y = (ymin / H->shrink) + cury;
                    png_byte *row = m_row_pointers[y];
                    // cerr << " memcpy " << y << " " << lcurx << " " << lgr << endl;
                    memcpy(&row[lcurx], &m_buffer[cury * lgr], lgr);
                }
                free(m_buffer);
                m_buffer = 0;
            }
        }
        // at this point the shrinked image can have unfilled pixels
        if ((H->shrink > 1) && (H->mype == 0)) {
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
                                imgFillGap(curx, cury, curx + vx, cury + vy, Iptr, &cpt, imgSizeX,
                                           imgSizeY);
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
void pngWriteFile(char *name, hydroparam_t *H) {
    int32_t cury = 0;
    int32_t curx = 0;
    int imgSizeX = H->globnx;
    int imgSizeY = H->globny;
    imgSizeX = imgSizeX / H->shrink;
    imgSizeY = imgSizeY / H->shrink;
    int y;

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
        png_byte color_type = PNG_COLOR_TYPE_RGBA;
        png_byte bit_depth = 8;

        /* create file */
        // cerr << "shrink factor " << H->shrink << " " << name << endl;

        /* initialize stuff */
        m_png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

        if (!m_png_ptr)
            abort_("[write_png_file] png_create_write_struct failed");

        m_info_ptr = png_create_info_struct(m_png_ptr);
        if (!m_info_ptr)
            abort_("[write_png_file] png_create_info_struct failed");

        // if (setjmp(png_jmpbuf(m_png_ptr)))
        //      abort_("[write_png_file] Error during init_io");

        png_init_io(m_png_ptr, H->fp);
        /* write header */
        // if (setjmp(png_jmpbuf(m_png_ptr)))
        //      abort_("[write_png_file] Error during writing header");

        png_set_IHDR(m_png_ptr, m_info_ptr, imgSizeX, imgSizeY, bit_depth, color_type,
                     PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);

        png_write_info(m_png_ptr, m_info_ptr);
        // if (H->mype == 0) cerr << "Header of final image written " << m_globNx << " " << m_globNy
        // << endl;
    }
    if (H->mype == 0) {
        png_bytep *rptrs;
        // cerr << "Write PNG image "; cerr.flush();
        png_write_image(m_png_ptr, m_row_pointers);
        /* end write */
        // if (setjmp(png_jmpbuf(m_png_ptr)))
        //      abort_("[write_png_file] Error during end of write");

        png_write_end(m_png_ptr, NULL);
        png_destroy_write_struct(&m_png_ptr, &m_info_ptr);
    }
    /* cleanup heap allocation */
    if (H->mype == 0) {
        for (y = 0; y < (H->globny / H->shrink); y++)
            free(m_row_pointers[y]);
        free(m_row_pointers);
    } else {
        free(m_row_pointers);
    }

#ifdef MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif
}
#endif // WITHPNG > 0

