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

#include "utils.h"
#include "hydro_utils.h"
#include "hydro_funcs.h"
void
hydro_init(hydroparam_t * H, hydrovar_t * Hv)
{
  long i, j;
  long x, y;

  // *WARNING* : we will use 0 based arrays everywhere since it is C code!
  H->imin = H->jmin = 0;

  // We add two extra layers left/right/top/bottom
  H->imax = H->nx + ExtraLayerTot;
  H->jmax = H->ny + ExtraLayerTot;
  H->nxt = H->imax - H->imin;   // column size in the array
  H->nyt = H->jmax - H->jmin;   // row size in the array
  // maximum direction size
  H->nxyt = (H->nxt > H->nyt) ? H->nxt : H->nyt;

  H->arSz = (H->nxyt + 2);
  H->arVarSz = (H->nxyt + 2) * H->nvar;
  H->arUoldSz = H->nvar * H->nxt * H->nyt;
  // allocate uold for each conservative variable
  Hv->uold = (real_t *) calloc(H->arUoldSz, sizeof(real_t));

  // wind tunnel with point explosion
  for (j = H->jmin + ExtraLayer; j < H->jmax - ExtraLayer; j++) {
    for (i = H->imin + ExtraLayer; i < H->imax - ExtraLayer; i++) {
      Hv->uold[IHvP(i, j, ID)] = one;
      Hv->uold[IHvP(i, j, IU)] = zero;
      Hv->uold[IHvP(i, j, IV)] = zero;
      Hv->uold[IHvP(i, j, IP)] = 1e-5;
    }
  }
  // Initial shock
  if (H->testCase == 0) {
    if (H->nproc == 1) {
      x = (H->imax - H->imin) / 2 + ExtraLayer * 0;
      y = (H->jmax - H->jmin) / 2 + ExtraLayer * 0;
      Hv->uold[IHvP(x, y, IP)] = one / H->dx / H->dx;
      printf("Centered test case : %ld %ld\n", x, y);
    } else {
      x = ((H->globnx) / 2);
      y = ((H->globny) / 2);
      if ((x >= H->box[XMIN_BOX]) && (x < H->box[XMAX_BOX]) && (y >= H->box[YMIN_BOX]) && (y < H->box[YMAX_BOX])) {
        x = x - H->box[XMIN_BOX] + ExtraLayer;
        y = y - H->box[YMIN_BOX] + ExtraLayer;
        Hv->uold[IHvP(x, y, IP)] = one / H->dx / H->dx;
        printf("Centered test case : [%d] %ld %ld\n", H->mype, x, y);
      }
    }
  }
  if (H->testCase == 1) {
    if (H->nproc == 1) {
      x = ExtraLayer;
      y = ExtraLayer;
      Hv->uold[IHvP(x, y, IP)] = one / H->dx / H->dx;
      printf("Lower corner test case : %ld %ld\n", x, y);
    } else {
      x = ExtraLayer;
      y = ExtraLayer;
      if ((x >= H->box[XMIN_BOX]) && (x < H->box[XMAX_BOX]) && (y >= H->box[YMIN_BOX]) && (y < H->box[YMAX_BOX])) {
        Hv->uold[IHvP(x, y, IP)] = one / H->dx / H->dx;
        printf("Lower corner test case : [%d] %ld %ld\n", H->mype, x, y);
      }
    }
  }
}                               // hydro_init

void
hydro_finish(const hydroparam_t H, hydrovar_t * Hv)
{
  Free(Hv->uold);
}                               // hydro_finish

void
allocate_work_space(const hydroparam_t H, hydrowork_t * Hw, hydrovarwork_t * Hvw)
{
  WHERE("allocate_work_space");

  Hvw->u = DMalloc(H.nxystep * H.arVarSz);
  Hvw->q = DMalloc(H.nxystep * H.arVarSz);
  Hvw->dq = DMalloc(H.nxystep * H.arVarSz);
  Hvw->qxm = DMalloc(H.nxystep * H.arVarSz);
  Hvw->qxp = DMalloc(H.nxystep * H.arVarSz);
  Hvw->qleft = DMalloc(H.nxystep * H.arVarSz);
  Hvw->qright = DMalloc(H.nxystep * H.arVarSz);
  Hvw->qgdnv = DMalloc(H.nxystep * H.arVarSz);
  Hvw->flux = DMalloc(H.nxystep * H.arVarSz);
  Hw->e = DMalloc(H.nxystep * H.arSz);
  Hw->c = DMalloc(H.nxystep * H.arSz);
  Hw->sgnm = IMalloc(H.nxystep * H.arSz);
}                               // allocate_work_space


void
deallocate_work_space(const hydroparam_t H, hydrowork_t * Hw, hydrovarwork_t * Hvw)
{
  WHERE("deallocate_work_space");

  Free(Hw->e);
  Free(Hvw->u);
  Free(Hvw->q);
  Free(Hvw->dq);
  Free(Hvw->qxm);
  Free(Hvw->qxp);
  Free(Hvw->qleft);
  Free(Hvw->qright);
  Free(Hvw->qgdnv);
  Free(Hvw->flux);
  Free(Hw->sgnm);
  Free(Hw->c);
}                               // deallocate_work_space


// EOF
