/*
  A simple 2D hydro code
  (C) Romain Teyssier : CEA/IRFU           -- original F90 code
  (C) Pierre-Francois Lavallee : IDRIS      -- original F90 code
  (C) Guillaume Colin de Verdiere : CEA/DAM -- for the C version
*/

#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <stdio.h>

#include "utils.h"
#include "hydro_utils.h"
#include "hydro_funcs.h"
void
hydro_init (hydroparam_t * H, hydrovar_t * Hv)
{
  int i, j;
  int x, y;

  // *WARNING* : we will use 0 based arrays everywhere since it is C code!
  H->imin = H->jmin = 0;

  // We add two extra layers left/right/top/bottom
  H->imax = H->nx + ExtraLayerTot;
  H->jmax = H->ny + ExtraLayerTot;
  H->nxt = H->imax - H->imin;	// column size in the array
  H->nyt = H->jmax - H->jmin;	// row size in the array
  // maximum direction size
  H->nxyt = (H->nxt > H->nyt) ? H->nxt : H->nyt;

  // allocate uold for each conservative variable
  Hv->uold = (double *) calloc (H->nvar * H->nxt * H->nyt, sizeof (double));

  // wind tunnel with point explosion
  for (j = H->jmin + ExtraLayer; j < H->jmax - ExtraLayer; j++)
    {
      for (i = H->imin + ExtraLayer; i < H->imax - ExtraLayer; i++)
	{
	  Hv->uold[IHvP (i, j, ID)] = one;
	  Hv->uold[IHvP (i, j, IU)] = zero;
	  Hv->uold[IHvP (i, j, IV)] = zero;
	  Hv->uold[IHvP (i, j, IP)] = 1e-5;
	}
    }

  // Initial shock
  if (H->nproc == 1)
    {
      x = (H->imax - H->imin) / 2 + ExtraLayer * 0;
      y = (H->jmax - H->jmin) / 2 + ExtraLayer * 0;
      Hv->uold[IHvP (x, y, IP)] = one / H->dx / H->dx;
      printf ("%d %d\n", x, y);
    }
  else
    {
      x = ((H->globnx + 2 * ExtraLayer) / 2);
      y = ((H->globny + 2 * ExtraLayer) / 2);
      if ((x >= H->box[XMIN_BOX]) && (x < H->box[XMAX_BOX])
	  && (y >= H->box[YMIN_BOX]) && (y < H->box[YMAX_BOX]))
	{
	  x = (H->globnx / 2) - H->box[XMIN_BOX] + ExtraLayer;
	  y = (H->globny / 2) - H->box[YMIN_BOX] + ExtraLayer;
	  Hv->uold[IHvP (x, y, IP)] = one / H->dx / H->dx;
	  printf ("[%d] %d %d\n", H->mype, x, y);
	}
    }
//   // Perturbation of the computation
//   for (i = 0; i < 10; i++) {
//     x = ((H->globnx + 2 * ExtraLayer) / 5) + i;
//     y = ((H->globny + 2 * ExtraLayer) / 4);
//     if (H->nproc == 1) {
//       Hv->uold[IHvP(x, y, ID)] = 1e5;
//       printf("%d %d\n", x, y);
//     } else {
//       if ((x >= H->box[XMIN_BOX]) && (x < H->box[XMAX_BOX]) && (y >= H->box[YMIN_BOX]) && (y < H->box[YMAX_BOX])) {
//         x = x - H->box[XMIN_BOX] + ExtraLayer;
//         y = y - H->box[YMIN_BOX] + ExtraLayer;
//         Hv->uold[IHvP(x, y, ID)] = 1e5;;
//         printf("[%d] %d %d\n", H->mype, x, y);
//       }
//     }
//   }
}				// hydro_init

void
hydro_finish (const hydroparam_t H, hydrovar_t * Hv)
{
  Free (Hv->uold);
}				// hydro_finish

void
allocate_work_space (int ngrid, const hydroparam_t H, hydrowork_t * Hw,
		     hydrovarwork_t * Hvw)
{
  WHERE ("allocate_work_space");
  Hvw->u = DMalloc ((ngrid + 2) * H.nxystep * H.nvar);
  Hvw->q = DMalloc ((ngrid + 2) * H.nxystep * H.nvar);
  Hvw->dq = DMalloc ((ngrid + 2) * H.nxystep * H.nvar);
  Hvw->qxm = DMalloc ((ngrid + 2) * H.nxystep * H.nvar);
  Hvw->qxp = DMalloc ((ngrid + 2) * H.nxystep * H.nvar);
  Hvw->qleft = DMalloc ((ngrid + 2) * H.nxystep * H.nvar);
  Hvw->qright = DMalloc ((ngrid + 2) * H.nxystep * H.nvar);
  Hvw->qgdnv = DMalloc ((ngrid + 2) * H.nxystep * H.nvar);
  Hvw->flux = DMalloc ((ngrid + 2) * H.nxystep * H.nvar);
  Hw->e = DMalloc ((ngrid + 1) * H.nxystep);
  Hw->c = DMalloc ((ngrid + 1) * H.nxystep);
  Hw->sgnm = IMalloc ((ngrid + 0) * H.nxystep);
  //   Hw->rl = DMalloc(ngrid);
  //   Hw->ul = DMalloc(ngrid);
  //   Hw->pl = DMalloc(ngrid);
  //   Hw->cl = DMalloc(ngrid);
  //   Hw->rr = DMalloc(ngrid);
  //   Hw->ur = DMalloc(ngrid);
  //   Hw->pr = DMalloc(ngrid);
  //   Hw->cr = DMalloc(ngrid);
  //   Hw->ro = DMalloc(ngrid);
  //   Hw->uo = DMalloc(ngrid);
  //   Hw->po = DMalloc(ngrid);
  //   Hw->co = DMalloc(ngrid);
  //   Hw->rstar = DMalloc(ngrid);
  //   Hw->ustar = DMalloc(ngrid);
  //   Hw->pstar = DMalloc(ngrid);
  //   Hw->cstar = DMalloc(ngrid);
  //   Hw->wl = DMalloc(ngrid);
  //   Hw->wr = DMalloc(ngrid);
  //   Hw->wo = DMalloc((ngrid));
  //   Hw->spin = DMalloc(ngrid);
  //   Hw->spout = DMalloc(ngrid);
  //   Hw->ushock = DMalloc(ngrid);
  //   Hw->frac = DMalloc(ngrid);
  //   Hw->scr = DMalloc(ngrid);
  //   Hw->delp = DMalloc(ngrid);
  //   Hw->pold = DMalloc(ngrid);
  //   Hw->ind = IMalloc(ngrid);
  //   Hw->ind2 = IMalloc(ngrid);
}				// allocate_work_space

void
deallocate_work_space (const hydroparam_t H, hydrowork_t * Hw,
		       hydrovarwork_t * Hvw)
{
  WHERE ("deallocate_work_space");

  //
  Free (Hw->e);
  Free (Hw->c);
  //
  Free (Hvw->u);
  Free (Hvw->q);
  Free (Hvw->dq);
  Free (Hvw->qxm);
  Free (Hvw->qxp);
  Free (Hvw->qleft);
  Free (Hvw->qright);
  Free (Hvw->qgdnv);
  Free (Hvw->flux);
  Free (Hw->sgnm);

  //
  //   Free(Hw->rl);
  //   Free(Hw->ul);
  //   Free(Hw->pl);
  //   Free(Hw->cl);
  //   Free(Hw->rr);
  //   Free(Hw->ur);
  //   Free(Hw->pr);
  //   Free(Hw->cr);
  //   Free(Hw->ro);
  //   Free(Hw->uo);
  //   Free(Hw->po);
  //   Free(Hw->co);
  //   Free(Hw->rstar);
  //   Free(Hw->ustar);
  //   Free(Hw->pstar);
  //   Free(Hw->cstar);
  //   Free(Hw->wl);
  //   Free(Hw->wr);
  //   Free(Hw->wo);
  //   Free(Hw->spin);
  //   Free(Hw->spout);
  //   Free(Hw->ushock);
  //   Free(Hw->frac);
  //   Free(Hw->scr);
  //   Free(Hw->delp);
  //   Free(Hw->pold);
  //   Free(Hw->ind);
  //   Free(Hw->ind2);
}				// deallocate_work_space


// EOF
