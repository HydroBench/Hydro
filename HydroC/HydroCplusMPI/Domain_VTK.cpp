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
#include <assert.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <float.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <stdarg.h>

using namespace std;

#include "EnumDefs.hpp"
#include "Domain.hpp"
#include "Soa.hpp"
#include "cclock.h"

typedef unsigned char byte;

const char s_CharPlusSign = '+';
const char s_CharSlash = '/';

static char SixBitToChar(byte b);
static char *ToBase64(unsigned char *data, int length);

char SixBitToChar(byte b)
{
	char c;
	if (b < 26) {
		c = (char)((int)b + (int)'A');
	} else if (b < 52) {
		c = (char)((int)b - 26 + (int)'a');
	} else if (b < 62) {
		c = (char)((int)b - 52 + (int)'0');
	} else if (b == 62) {
		c = s_CharPlusSign;
	} else {
		c = s_CharSlash;
	}
	return c;
}

char *ToBase64(unsigned char *data, int length)
{
	int padding = length % 3;
	int blocks = (length - 1) / 3 + 1;
	size_t lalloc;
	char *s;
	int i;

	if (length == 0)
		return NULL;

	if (padding > 0)
		padding = 3 - padding;

	// lalloc = (blocks * 4 + 1 + 16);
	lalloc = blocks;
	lalloc *= 4;
	lalloc += 17;

	s = (char *)malloc(lalloc);
	if (s == NULL) {
		fprintf(stderr, "Length=%d, blocks=%d lalloc=%ld\n", length,
			blocks, lalloc);
	}
	assert(s != NULL);

	for (i = 0; i < blocks; i++) {
		bool finalBlock = i == blocks - 1;
		bool pad2 = false;
		bool pad1 = false;
		if (finalBlock) {
			pad2 = padding == 2;
			pad1 = padding > 0;
		}

		int index = i * 3;
		byte b1 = data[index];
		byte b2 = pad2 ? (byte) 0 : data[index + 1];
		byte b3 = pad1 ? (byte) 0 : data[index + 2];

		byte temp1 = (byte) ((b1 & 0xFC) >> 2);

		byte temp = (byte) ((b1 & 0x03) << 4);
		byte temp2 = (byte) ((b2 & 0xF0) >> 4);
		temp2 += temp;

		temp = (byte) ((b2 & 0x0F) << 2);
		byte temp3 = (byte) ((b3 & 0xC0) >> 6);
		temp3 += temp;

		byte temp4 = (byte) (b3 & 0x3F);

		index = i * 4;
		s[index] = SixBitToChar(temp1);
		s[index + 1] = SixBitToChar(temp2);
		s[index + 2] = pad2 ? '=' : SixBitToChar(temp3);
		s[index + 3] = pad1 ? '=' : SixBitToChar(temp4);
	}
	s[blocks * 4] = (byte) 0;
	return s;
}

#define BINARY 1

void vtkwpvd(int nout, const char *r)
{
	char n[1024];
	char vfname[1024];
	int i;
	FILE *vf = NULL;
	char tmp[10];

	vf = fopen("Hydro.pvd", "w");
	fprintf(vf, "<?xml version=\"1.0\"?>\n");
	fprintf(vf,
		" <VTKFile type=\"Collection\" version=\"0.1\" byte_order=\"LittleEndian\">\n");
	fprintf(vf, "  <Collection>\n");

#pragma novector
	for (i = 1; i <= nout; i++) {
		sprintf(tmp, "%06d", i);
		sprintf(n, "Dep/%c%c%c%c", tmp[0], tmp[1], tmp[2], tmp[3]);
		sprintf(n, "%s/%c%c", n, tmp[4], tmp[5]);
		sprintf(vfname, "%s/Hydro_%04d.pvtr", n, i);
		fprintf(vf,
			"  <DataSet timestep=\"%d\" part=\"0\" file=\"%s\"  name=\"Asmb:FRAME\"/>\n",
			i, vfname);
	}

	fprintf(vf, " </Collection>\n");
	fprintf(vf, "</VTKFile>\n");
	fclose(vf);
}

void vtknm(char *n, int me, int nout)
{
	char tmp[10];

	sprintf(tmp, "%06d", nout);
	sprintf(n, "Dep");
	if (me == 0) {
		mkdir(n, 0777);
	}
	sprintf(n, "%s/%c%c%c%c", n, tmp[0], tmp[1], tmp[2], tmp[3]);
	if (me == 0) {
		mkdir(n, 0777);
	}
	sprintf(n, "%s/%c%c", n, tmp[4], tmp[5]);

	if (me == 0) {
		mkdir(n, 0777);
	}
}

void Domain::vtkfile(int step)
{
	char name[1024];
	char vfrname[1024];
	FILE *fic, *vf;
	int i, j, nv;

	// First step : create the directory structure ONLY using PE0
#ifdef MPI_ON
	if (m_nProc > 1)
		MPI_Barrier(MPI_COMM_WORLD);
#endif
	vtknm(vfrname, m_myPe, step);	// create the directory structure
	// if (H.mype == 0) fprintf(stderr, "%s\n", vfrname);
#ifdef MPI_ON
	if (m_nProc > 1)
		MPI_Barrier(MPI_COMM_WORLD);
#endif

	// Write a domain per PE
	sprintf(name, "%s/Hydro_%05d_%04d.vtr", vfrname, m_myPe, step);
	fic = fopen(name, "w");
	if (fic == NULL) {
		fprintf(stderr, "Ouverture du fichier %s impossible\n", name);
		abort();
	}
	fprintf(fic, "<?xml version=\"1.0\"?>\n");
	fprintf(fic,
		"<VTKFile type=\"RectilinearGrid\" byte_order=\"LittleEndian\">\n");
	fprintf(fic, " <RectilinearGrid WholeExtent=\" %d %d %d %d %d %d\">\n",
		m_box[XMIN_D], m_box[XMAX_D], m_box[YMIN_D], m_box[YMAX_D], 0,
		1);
	fprintf(fic,
		"  <Piece Extent=\" %d %d %d %d %d %d\" GhostLevel=\"0\">\n",
		m_box[XMIN_D], m_box[XMAX_D], m_box[YMIN_D], m_box[YMAX_D], 0,
		1);
	fprintf(fic, "   <Coordinates>\n");

	fprintf(fic,
		"    <DataArray type=\"Float32\" format=\"ascii\" NumberOfComponents=\"1\">\n");
	for (i = m_box[XMIN_D]; i <= m_box[XMAX_D]; i++) {
		fprintf(fic, "%f ", i * m_dx);
	}
	fprintf(fic, "\n");
	fprintf(fic, "    </DataArray>\n");
	fprintf(fic,
		"    <DataArray type=\"Float32\" format=\"ascii\" NumberOfComponents=\"1\">\n");
	for (j = m_box[YMIN_D]; j <= m_box[YMAX_D]; j++) {
		fprintf(fic, "%f ", j * m_dx);
	}
	fprintf(fic, "\n");
	fprintf(fic, "    </DataArray>\n");
	fprintf(fic,
		"    <DataArray type=\"Float32\" format=\"ascii\" NumberOfComponents=\"1\">\n");
	fprintf(fic, "%f %f\n", 0., 1. * m_dx);
	fprintf(fic, "    </DataArray>\n");
	fprintf(fic, "   </Coordinates>\n");
	name[0] = 0;
	for (nv = 0; nv < NB_VAR; nv++) {
		if (nv == ID_VAR)
			sprintf(name, "%s varID", name);
		if (nv == IU_VAR)
			sprintf(name, "%s varIU", name);
		if (nv == IV_VAR)
			sprintf(name, "%s varIV", name);
		if (nv == IP_VAR)
			sprintf(name, "%s varIP", name);
	}

	// declaration of the variable list
	fprintf(fic, "   <CellData Scalars=\"%s\">\n", name);
	name[0] = 0;
	for (nv = 0; nv < NB_VAR; nv++) {
		if (nv == ID_VAR)
			sprintf(name, "varID");
		if (nv == IU_VAR)
			sprintf(name, "varIU");
		if (nv == IV_VAR)
			sprintf(name, "varIV");
		if (nv == IP_VAR)
			sprintf(name, "varIP");

		//Definition of the cell values
#if BINARY == 1
		fprintf(fic,
			"    <DataArray Name=\"%s\" type=\"Float32\" format=\"binary\" encoding=\"base64\" NumberOfComponents=\"1\">\n",
			name);
		{
			Matrix2 < real_t > &uold = *(*m_uold) (nv);
			// float tuold[H.nxt * H.nyt];
			uint32_t xmin, xmax, ymin, ymax;
			getExtends(TILE_FULL, xmin, xmax, ymin, ymax);
			float *tuold = NULL;
			char *r64;
			size_t p = 0, lst;

			assert((xmax * ymax) > 0);
			tuold =
			    (float *)calloc(xmax * ymax + 16, sizeof(float));
			assert(tuold != NULL);

			for (j = ymin + m_ExtraLayer; j < ymax - m_ExtraLayer;
			     j++) {
				for (i = xmin + m_ExtraLayer;
				     i < xmax - m_ExtraLayer; i++) {
					tuold[p++] = (float)uold(i, j);
				}
			}
			// Header = size of the following items
			assert(p < xmax * ymax);

			p *= sizeof(float);
			r64 = ToBase64((byte *) & p, sizeof(int));
			lst = strlen(r64);
			fwrite(r64, 1, lst, fic);
			free(r64);
			r64 = ToBase64((byte *) tuold, p);
			lst = strlen(r64);
			fwrite(r64, 1, lst, fic);
			free(r64);
			free(tuold);
		}
#else
		fprintf(fic,
			"    <DataArray type=\"Float32\" Name=\"%s\" format=\"ascii\" NumberOfComponents=\"1\">\n",
			name);

		// the image is the interior of the computed domain
		for (j = H.jmin + ExtraLayer; j < H.jmax - ExtraLayer; j++) {
			for (i = H.imin + ExtraLayer; i < H.imax - ExtraLayer;
			     i++) {
				fprintf(fic, "%lf ", Hv->uold[IHv(i, j, nv)]);
			}
			fprintf(fic, "\n");
		}
#endif
		fprintf(fic, "    </DataArray>\n");
	}
	fprintf(fic, "   </CellData>\n");
	fprintf(fic, "  </Piece>\n");
	fprintf(fic, " </RectilinearGrid>\n");
	fprintf(fic, "</VTKFile>\n");
	fclose(fic);

	// At this stage we can write VTK containers. Since only one file is
	// necessary even for multiple domains, it has to be written by one
	// PE only.

#ifdef MPI_ON
	if (m_nProc > 1)
		MPI_Barrier(MPI_COMM_WORLD);
#endif
	if (m_myPe == 0) {
		sprintf(name, "outputvtk_%05d.pvtr", step);
		sprintf(name, "%s/Hydro_%04d.pvtr", vfrname, step);
		vf = fopen(name, "w");
		if (vf == NULL) {
			fprintf(stderr, "Ouverture du fichier %s impossible\n",
				name);
			abort();
		}
		fprintf(vf, "<?xml version=\"1.0\"?>\n");
		fprintf(vf,
			"<VTKFile type=\"PRectilinearGrid\" byte_order=\"LittleEndian\">\n");
		fprintf(vf,
			"<PRectilinearGrid WholeExtent=\"0 %d 0 %d 0 %d\"  GhostLevel=\"0\" >\n",
			m_globNx, m_globNy, 1);
		fprintf(vf, " <PCellData>\n");
#pragma novector
		for (nv = 0; nv < NB_VAR; nv++) {
			name[0] = '\0';
			if (nv == ID_VAR)
				sprintf(name, "varID");
			if (nv == IU_VAR)
				sprintf(name, "varIU");
			if (nv == IV_VAR)
				sprintf(name, "varIV");
			if (nv == IP_VAR)
				sprintf(name, "varIP");

#if BINARY == 1
			fprintf(vf,
				"  <PDataArray Name=\"%s\" type=\"Float32\" format=\"binary\" encoding=\"base64\" NumberOfComponents=\"1\"/>\n",
				name);
#else
			fprintf(vf,
				"  <PDataArray Name=\"%s\" type=\"Float32\" format=\"ascii\" NumberOfComponents=\"1\"/>\n",
				name);
#endif
		}
		fprintf(vf, " </PCellData>\n");
		fprintf(vf, " <PCoordinates>\n");
		fprintf(vf,
			"  <PDataArray type=\"Float32\" format=\"ascii\" NumberOfComponents=\"1\"/>\n");
		fprintf(vf,
			"  <PDataArray type=\"Float32\" format=\"ascii\" NumberOfComponents=\"1\"/>\n");
		fprintf(vf,
			"  <PDataArray type=\"Float32\" format=\"ascii\" NumberOfComponents=\"1\"/>\n");
		fprintf(vf, " </PCoordinates>\n");
		for (i = 0; i < m_nProc; i++) {
			int box[8];
			memset(box, 0, 8 * sizeof(int));
			CalcSubSurface(0, m_globNx, 0, m_globNy, 0, m_nProc - 1,
				       box, i);
			sprintf(name, "Hydro_%05d_%04d.vtr", i, step);
			fprintf(vf,
				" <Piece Extent=\"%d %d %d %d %d %d\" Source=\"%s\"/>\n",
				box[XMIN_D], box[XMAX_D], box[YMIN_D],
				box[YMAX_D], 0, 1, name);
		}
		fprintf(vf, "</PRectilinearGrid>\n");
		fprintf(vf, "</VTKFile>\n");
		fclose(vf);

		// We make the time step available only now to ensure consistency
		vtkwpvd(step, (const char *) ("Dep"));
	}
}

void Domain::vtkOutput(int n)
{
	vtkfile(n);
}
