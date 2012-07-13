/*
  A simple 2D hydro code
  (C) Romain Teyssier : CEA/IRFU           -- original F90 code
  (C) Pierre-Francois Lavallee : IDRIS      -- original F90 code
  (C) Guillaume Colin de Verdiere : CEA/DAM -- for the C version
*/

/*

This software is governed by the CeCILL license under French law and
abiding by the rules of distribution of free software.  You can  use, 
modify and/ or redistribute the software under the terms of the CeCILL
license as circulated by CEA, CNRS and INRIA at the following URL
"http://www.cecill.info". 

As a counterpart to the access to the source code and  rights to copy,
modify and redistribute granted by the license, users are provided only
with a limited warranty  and the software's author,  the holder of the
economic rights,  and the successive licensors  have only  limited
liability. 

In this respect, the user's attention is drawn to the risks associated
with loading,  using,  modifying and/or developing or reproducing the
software by the user in light of its specific status of free software,
that may mean  that it is complicated to manipulate,  and  that  also
therefore means  that it is reserved for developers  and  experienced
professionals having in-depth computer knowledge. Users are therefore
encouraged to load and test the software's suitability as regards their
requirements in conditions enabling the security of their systems and/or 
data to be ensured and,  more generally, to use and operate it in the 
same conditions as regards security. 

The fact that you are presently reading this means that you have had
knowledge of the CeCILL license and that you accept its terms.

*/

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <strings.h>

#include "parametres.h"
#include "utils.h"
#include "vtkfile.h"
void
vtkfile(long step, const hydroparam_t H, hydrovar_t * Hv)
{
  char name[160];
  FILE *fic;
  long i, j, nv;

  WHERE("vtkfile");
  sprintf(name, "outputvtk_%05ld.vts", step);
  fic = fopen(name, "w");
  if (fic == NULL) {
    fprintf(stderr, "Ouverture du fichier %s impossible\n", name);
    exit(1);
  }
  fprintf(fic, "<?xml version=\"1.0\"?>\n");
  fprintf(fic, "<VTKFile type=\"StructuredGrid\">\n");
  fprintf(fic, "<StructuredGrid WholeExtent=\" %ld %ld %ld %ld %ld %ld\">\n", (long) 0, H.nx, (long) 0, H.ny, (long) 0, (long) 0);
  fprintf(fic, "<Piece Extent=\" %ld %ld %ld %ld %ld %ld\">\n", (long) 0, H.nx, (long) 0, H.ny, (long) 0, (long) 0);
  fprintf(fic, "<Points>\n");
  fprintf(fic, "<DataArray type=\"Float32\" format=\"ascii\" NumberOfComponents=\"3\">\n");
  for (j = 0; j < H.ny + 1; j++) {
    for (i = 0; i < H.nx + 1; i++) {
      fprintf(fic, "%f %f %f\n", i * H.dx, j * H.dx, 0.0);
    }
  }
  fprintf(fic, "</DataArray>\n");
  fprintf(fic, "</Points>\n");
  name[0] = 0;
  for (nv = 0; nv < IP; nv++) {
    if (nv == ID)
      sprintf(name, "%s varID", name);
    if (nv == IU)
      sprintf(name, "%s varIU", name);
    if (nv == IV)
      sprintf(name, "%s varIV", name);
    if (nv == IP)
      sprintf(name, "%s varIP", name);
  }

  // declaration of the variable list
  fprintf(fic, "<CellData Scalars=\"%s\">\n", name);
  name[0] = 0;
  for (nv = 0; nv <= IP; nv++) {
    if (nv == ID)
      sprintf(name, "varID");
    if (nv == IU)
      sprintf(name, "varIU");
    if (nv == IV)
      sprintf(name, "varIV");
    if (nv == IP)
      sprintf(name, "varIP");

    //Definition of the cell values
    fprintf(fic, "<DataArray type=\"Float32\" Name=\"%s\" format=\"ascii\">\n", name);

    // the image is the interior of the computed domain
    for (j = H.jmin + ExtraLayer; j < H.jmax - ExtraLayer; j++) {
      for (i = H.imin + ExtraLayer; i < H.imax - ExtraLayer; i++) {
        fprintf(fic, "%lf ", Hv->uold[IHv(i, j, nv)]);
      }
      fprintf(fic, "\n");
    }
    fprintf(fic, "</DataArray>\n");
  }
  fprintf(fic, "</CellData>\n");
  fprintf(fic, "</Piece>\n");
  fprintf(fic, "</StructuredGrid>\n");
  fprintf(fic, "</VTKFile>\n");
  fclose(fic);
}
