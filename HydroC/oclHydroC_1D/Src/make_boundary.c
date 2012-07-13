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

#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <stdio.h>

#include "parametres.h"
#include "make_boundary.h"
#include "utils.h"
void
make_boundary(long idim, const hydroparam_t H, hydrovar_t * Hv)
{

  // - - - - - - - - - - - - - - - - - - -
  // Cette portion de code est à vérifier
  // détail. J'ai des doutes sur la conversion
  // des index depuis fortran.
  // - - - - - - - - - - - - - - - - - - -
  long i, ivar, i0, j, j0;
  double sign;
  WHERE("make_boundary");
  if (idim == 1) {

    // Left boundary
    for (ivar = 0; ivar < H.nvar; ivar++) {
      for (i = 0; i < ExtraLayer; i++) {
        sign = 1.0;
        if (H.boundary_left == 1) {
          i0 = ExtraLayerTot - i - 1;
          if (ivar == IU) {
            sign = -1.0;
          }
        } else if (H.boundary_left == 2) {
          i0 = 2;
        } else {
          i0 = H.nx + i;
        }
        for (j = H.jmin + ExtraLayer; j < H.jmax - ExtraLayer; j++) {
          Hv->uold[IHv(i, j, ivar)] = Hv->uold[IHv(i0, j, ivar)] * sign;
          MFLOPS(1, 0, 0, 0);
        }
      }
    }

    // Right boundary
    for (ivar = 0; ivar < H.nvar; ivar++) {
      for (i = H.nx + ExtraLayer; i < H.nx + ExtraLayerTot; i++) {
        sign = 1.0;
        if (H.boundary_right == 1) {
          i0 = 2 * H.nx + ExtraLayerTot - i - 1;
          if (ivar == IU) {
            sign = -1.0;
          }
        } else if (H.boundary_right == 2) {
          i0 = H.nx + ExtraLayer;
        } else {
          i0 = i - H.nx;
        }
        for (j = H.jmin + ExtraLayer; j < H.jmax - ExtraLayer; j++) {
          Hv->uold[IHv(i, j, ivar)] = Hv->uold[IHv(i0, j, ivar)] * sign;
          MFLOPS(1, 0, 0, 0);
        }
      }
    }
  } else {

    // Lower boundary
    j0 = 0;
    for (ivar = 0; ivar < H.nvar; ivar++) {
      for (j = 0; j < ExtraLayer; j++) {
        sign = 1.0;
        if (H.boundary_down == 1) {
          j0 = ExtraLayerTot - j - 1;
          if (ivar == IV) {
            sign = -1.0;
          }
        } else if (H.boundary_down == 2) {
          j0 = ExtraLayerTot;
        } else {
          j0 = H.ny + j;
        }
        for (i = H.imin + ExtraLayer; i < H.imax - ExtraLayer; i++) {
          Hv->uold[IHv(i, j, ivar)] = Hv->uold[IHv(i, j0, ivar)] * sign;
          MFLOPS(1, 0, 0, 0);
        }
      }
    }

    // Upper boundary
    for (ivar = 0; ivar < H.nvar; ivar++) {
      for (j = H.ny + ExtraLayer; j < H.ny + ExtraLayerTot; j++) {
        sign = 1.0;
        if (H.boundary_up == 1) {
          j0 = 2 * H.ny + ExtraLayerTot - j - 1;
          if (ivar == IV) {
            sign = -1.0;
          }
        } else if (H.boundary_up == 2) {
          j0 = H.ny + 1;
        } else {
          j0 = j - H.ny;
        }
        for (i = H.imin + ExtraLayer; i < H.imax - ExtraLayer; i++) {
          Hv->uold[IHv(i, j, ivar)] = Hv->uold[IHv(i, j0, ivar)] * sign;
          MFLOPS(1, 0, 0, 0);
        }
      }
    }
  }
}                               // make_boundary


//EOF
