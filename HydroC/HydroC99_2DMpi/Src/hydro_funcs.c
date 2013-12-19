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
#include <stdint.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>

#include "utils.h"
#include "hydro_utils.h"
#include "hydro_funcs.h"

#include "hydro_numa.h"

void
hydro_init(hydroparam_t * H, hydrovar_t * Hv) {
  int i, j;
  int x, y;

  // *WARNING* : we will use 0 based arrays everywhere since it is C code!
  H->imin = H->jmin = 0;

  // We add two extra layers left/right/top/bottom
  H->imax = H->nx + ExtraLayerTot;
  H->jmax = H->ny + ExtraLayerTot;
  H->nxt = H->imax - H->imin;   // column size in the array
  H->nyt = H->jmax - H->jmin;   // row size in the array

  // maximum direction size
  H->nxyt = (H->nxt > H->nyt) ? H->nxt : H->nyt;
  // To make sure that slices are properly aligned, we make the array a
  // multiple of NDBLE double
#define NDBLE 16
  // printf("avant %d ", H->nxyt);
  // H->nxyt = (H->nxyt + NDBLE - 1) / NDBLE;
  // H->nxyt *= NDBLE;
  // printf("apres %d \n", H->nxyt);

  // allocate uold for each conservative variable
  Hv->uold = (real_t *) DMalloc(H->nvar * H->nxt * H->nyt);

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
      printf("Centered test case : %d %d\n", x, y);
    } else {
      x = ((H->globnx) / 2);
      y = ((H->globny) / 2);
      if ((x >= H->box[XMIN_BOX]) && (x < H->box[XMAX_BOX]) && (y >= H->box[YMIN_BOX]) && (y < H->box[YMAX_BOX])) {
        x = x - H->box[XMIN_BOX] + ExtraLayer;
        y = y - H->box[YMIN_BOX] + ExtraLayer;
        Hv->uold[IHvP(x, y, IP)] = one / H->dx / H->dx;
        printf("Centered test case : [%d] %d %d\n", H->mype, x, y);
      }
    }
  }
  if (H->testCase == 1) {
    if (H->nproc == 1) {
      x = ExtraLayer;
      y = ExtraLayer;
      Hv->uold[IHvP(x, y, IP)] = one / H->dx / H->dx;
      printf("Lower corner test case : %d %d\n", x, y);
    } else {
      x = ExtraLayer;
      y = ExtraLayer;
      if ((x >= H->box[XMIN_BOX]) && (x < H->box[XMAX_BOX]) && (y >= H->box[YMIN_BOX]) && (y < H->box[YMAX_BOX])) {
        Hv->uold[IHvP(x, y, IP)] = one / H->dx / H->dx;
        printf("Lower corner test case : [%d] %d %d\n", H->mype, x, y);
      }
    }
  }
  if (H->testCase == 2) {
    if (H->nproc == 1) {
      x = ExtraLayer;
      y = ExtraLayer;
      for (j = y; j < H->jmax; j++) {
        Hv->uold[IHvP(x, j, IP)] = one / H->dx / H->dx;
      }
      printf("SOD tube test case\n");
    } else {
      x = ExtraLayer;
      y = ExtraLayer;
      for (j = 0; j < H->globny; j++) {
        if ((x >= H->box[XMIN_BOX]) && (x < H->box[XMAX_BOX]) && (j >= H->box[YMIN_BOX]) && (j < H->box[YMAX_BOX])) {
          y = j - H->box[YMIN_BOX] + ExtraLayer;
          Hv->uold[IHvP(x, y, IP)] = one / H->dx / H->dx;
        }
      }
      printf("SOD tube test case in //\n");
    }
  }
  if (H->testCase > 2) {
    printf("Test case not implemented -- aborting !\n");
    abort();
  }
  fflush(stdout);
}                               // hydro_init

void
hydro_finish(const hydroparam_t H, hydrovar_t * Hv) {
  DFree(&Hv->uold, H.nvar * H.nxt * H.nyt);
}                               // hydro_finish


static void touchPage(real_t *adr, int lg) {
  int i;
#ifndef NOTOUCHPAGE
#pragma omp parallel for private(i) shared(adr) 
  for(i = 0; i < lg; i++) {
    adr[i] = 0.0l;
  }
#endif
}


void
allocate_work_space(int ngrid, const hydroparam_t H, hydrowork_t * Hw, hydrovarwork_t * Hvw) {
  int domain = ngrid * H.nxystep;
  int domainVar = domain * H.nvar;
  int domainD = domain * sizeof(real_t);
  int domainI = domain * sizeof(int);
  int domainVarD = domainVar * sizeof(real_t);
  int pageM = 1024*1024;

#define ONEBLOCK 1

#ifndef PAGEOFFSET
#define PAGEOFFSET sizeof(double)
#endif

#ifdef ONEBLOCK
#ifndef TAILLEPAGE
#define TAILLEPAGE 1024
#endif
  int oneBlock = 0;
  int domainVarM = 0;
  int domainM = 0;
  int pageMD = TAILLEPAGE / 8 ;
  real_t *blockD = 0; 
#endif

  WHERE("allocate_work_space");

#ifdef MOVETHEPAGES
#ifndef __MIC__
#define MOVEPAGEVAR(t) force_move_pages(t, domainVar, sizeof(real_t), HYDRO_NUMA_SIZED_BLOCK_RR, pageM)
#define MOVEPAGE(t)    force_move_pages(t, domain,    sizeof(real_t), HYDRO_NUMA_SIZED_BLOCK_RR, pageM)
#else
#define MOVEPAGEVAR(t) 
#define MOVEPAGE(t)    
#endif
#else
#define MOVEPAGEVAR(t) 
#define MOVEPAGE(t)    
#endif

#ifdef ONEBLOCK
  if (H.mype == 0) fprintf(stdout, "Page offset %d\n", (int) PAGEOFFSET);
  // determine the right amount of pages to fit all arrays
  domainVarM = (domainVar + pageMD - 1) / pageMD;
  domainVarM *= pageMD + PAGEOFFSET;
  domainM = (domain + pageMD - 1) / pageMD;
  domainM *= pageMD + PAGEOFFSET;

  oneBlock = 9 * domainVarM + 12 * domainM;  // expressed in term of pages of double
  assert(oneBlock >= (9 * domainVar + 12 * domain));
#pragma message "ONE BLOCK option"
  blockD = (real_t *) malloc(oneBlock * sizeof(real_t));
  assert(blockD != NULL);
  if (((uint64_t) (&blockD[0]) & 63) == 0) {
    fprintf(stderr, "ONE block allocated is not aligned \n");
  }
  Hvw->u      = blockD;                   touchPage(Hvw->u, domainVar);     
  Hvw->q      = Hvw->u      + domainVarM; touchPage(Hvw->q, domainVar);     
  Hvw->dq     = Hvw->q      + domainVarM; touchPage(Hvw->dq, domainVar);    
  Hvw->qxm    = Hvw->dq     + domainVarM; touchPage(Hvw->qxm, domainVar);   
  Hvw->qxp    = Hvw->qxm    + domainVarM; touchPage(Hvw->qxp, domainVar);   
  Hvw->qleft  = Hvw->qxp    + domainVarM; touchPage(Hvw->qleft, domainVar); 
  Hvw->qright = Hvw->qleft  + domainVarM; touchPage(Hvw->qright, domainVar);
  Hvw->qgdnv  = Hvw->qright + domainVarM; touchPage(Hvw->qgdnv, domainVar); 
  Hvw->flux   = Hvw->qgdnv  + domainVarM; touchPage(Hvw->flux, domainVar);  
  Hw->e       = Hvw->flux   + domainVarM; touchPage(Hw->e, domain);         
  Hw->c       = Hw->e       + domainM;    touchPage(Hw->c, domain);         
  Hw->pstar   = Hw->c       + domainM;    touchPage(Hw->pstar, domain);     
  Hw->rl      = Hw->pstar   + domainM;    touchPage(Hw->rl, domain);        
  Hw->ul      = Hw->rl      + domainM;    touchPage(Hw->ul, domain);        
  Hw->pl      = Hw->ul      + domainM;    touchPage(Hw->pl, domain);        
  Hw->cl      = Hw->pl      + domainM;    touchPage(Hw->cl, domain);        
  Hw->rr      = Hw->cl      + domainM;    touchPage(Hw->rr, domain);        
  Hw->ur      = Hw->rr      + domainM;    touchPage(Hw->ur, domain);        
  Hw->pr      = Hw->ur      + domainM;    touchPage(Hw->pr, domain);        
  Hw->cr      = Hw->pr      + domainM;    touchPage(Hw->cr, domain);        
  Hw->ro      = Hw->cr      + domainM;    touchPage(Hw->ro, domain);        
#else
  /*
    force_move_pages(Hvw->u, domainVar, sizeof(double), HYDRO_NUMA_SIZED_BLOCK_RR, pageM);
  */
  fprintf(stderr, "Page malloc\n");
  Hvw->u      = DMalloc(domainVar); MOVEPAGEVAR(Hvw->u);
  Hvw->q      = DMalloc(domainVar); MOVEPAGEVAR(Hvw->q);
  Hvw->dq     = DMalloc(domainVar); MOVEPAGEVAR(Hvw->dq);
  Hvw->qxm    = DMalloc(domainVar); MOVEPAGEVAR(Hvw->qxm);
  Hvw->qxp    = DMalloc(domainVar); MOVEPAGEVAR(Hvw->qxp);
  Hvw->qleft  = DMalloc(domainVar); MOVEPAGEVAR(Hvw->qleft);
  Hvw->qright = DMalloc(domainVar); MOVEPAGEVAR(Hvw->qright);
  Hvw->qgdnv  = DMalloc(domainVar); MOVEPAGEVAR(Hvw->qgdnv);
  Hvw->flux   = DMalloc(domainVar); MOVEPAGEVAR(Hvw->flux);
  //
  Hw->e       = DMalloc(domain); MOVEPAGE(Hw->e);
  Hw->c       = DMalloc(domain); MOVEPAGE(Hw->c);
  // 
  Hw->pstar = DMalloc(domain); MOVEPAGE(Hw->pstar);
  Hw->rl    = DMalloc(domain); MOVEPAGE(Hw->rl);
  Hw->ul    = DMalloc(domain); MOVEPAGE(Hw->ul);
  Hw->pl    = DMalloc(domain); MOVEPAGE(Hw->pl);
  Hw->cl    = DMalloc(domain); MOVEPAGE(Hw->cl);
  Hw->rr    = DMalloc(domain); MOVEPAGE(Hw->rr);
  Hw->ur    = DMalloc(domain); MOVEPAGE(Hw->ur);
  Hw->pr    = DMalloc(domain); MOVEPAGE(Hw->pr);
  Hw->cr    = DMalloc(domain); MOVEPAGE(Hw->cr);
  Hw->ro    = DMalloc(domain); MOVEPAGE(Hw->ro);
#endif
  Hw->goon  = IMalloc(domain);
  Hw->sgnm  = IMalloc(domain);

  //   Hw->uo = DMalloc(ngrid);
  //   Hw->po = DMalloc(ngrid);
  //   Hw->co = DMalloc(ngrid);
  //   Hw->rstar = DMalloc(ngrid);
  //   Hw->ustar = DMalloc(ngrid);
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
}                               // allocate_work_space

void
deallocate_work_space(int ngrid, const hydroparam_t H, hydrowork_t * Hw, hydrovarwork_t * Hvw) {
  int domain = ngrid * H.nxystep;
  int domainVar = domain * H.nvar;
  int domainD = domain * sizeof(real_t);
  int domainI = domain * sizeof(int);
  int domainVarD = domainVar * sizeof(real_t);

  WHERE("deallocate_work_space");
#ifdef ONEBLOCK
  int oneBlock = 0;
  int domainVarM = 0;
  int domainM = 0;
  int pageM = 1024*1024;
  int pageMD = TAILLEPAGE / 8;
  real_t *blockD = 0; 
#endif

  //
#ifdef ONEBLOCK
  // determine the right amount of pages to fit all arrays
  domainVarM = (domainVar + pageMD - 1) / pageMD;
  domainVarM *= pageMD + PAGEOFFSET;
  domainM = (domain + pageMD - 1) / pageMD;
  domainM *= pageMD + PAGEOFFSET;

  oneBlock = 9 * domainVarM + 12 * domainM;  // expressed in term of pages of double
  DFree(&Hvw->u, oneBlock);
  Hvw->q = Hvw->dq = Hvw->qxm = Hvw->qxp = 0;
  Hvw->qleft = Hvw->qright = Hvw->qgdnv = Hvw->flux = Hw->e = Hw->c =0;
  Hw->pstar = Hw->rl = Hw->ul = Hw->pl = Hw->cl = Hw->rr = Hw->ur =  Hw->pr = Hw->cr = Hw->ro = 0;
#else
  DFree(&Hvw->u, domainVar);
  DFree(&Hvw->q, domainVar);
  DFree(&Hvw->dq, domainVar);
  DFree(&Hvw->qxm, domainVar);
  DFree(&Hvw->qxp, domainVar);
  DFree(&Hvw->qleft, domainVar);
  DFree(&Hvw->qright, domainVar);
  DFree(&Hvw->qgdnv, domainVar);
  DFree(&Hvw->flux, domainVar);
  DFree(&Hw->e, domain);
  DFree(&Hw->c, domain);
  DFree(&Hw->pstar, domain);
  DFree(&Hw->rl, domain);
  DFree(&Hw->ul, domain);
  DFree(&Hw->pl, domain);
  DFree(&Hw->cl, domain);
  DFree(&Hw->rr, domain);
  DFree(&Hw->ur, domain);
  DFree(&Hw->pr, domain);
  DFree(&Hw->cr, domain);
  DFree(&Hw->ro, domain);
#endif

  IFree(&Hw->sgnm, domainVar);
  IFree(&Hw->goon, domain);
  //   Free(Hw->uo);
  //   Free(Hw->po);
  //   Free(Hw->co);
  //   Free(Hw->rstar);
  //   Free(Hw->ustar);
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
}                               // deallocate_work_space


// EOF
