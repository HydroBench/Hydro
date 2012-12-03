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
#include "utils.h"
#include "oclRiemann.h"
#include "oclInit.h"
#include "ocltools.h"

typedef struct _Args {
  cl_mem qleft;
  cl_mem qright;
  cl_mem qgdnv;
  cl_mem rl;
  cl_mem ul;
  cl_mem pl;
  cl_mem cl;
  cl_mem wl;
  cl_mem rr;
  cl_mem ur;
  cl_mem pr;
  cl_mem cr;
  cl_mem wr;
  cl_mem ro;
  cl_mem uo;
  cl_mem po;
  cl_mem co;
  cl_mem wo;
  cl_mem rstar;
  cl_mem ustar;
  cl_mem pstar;
  cl_mem cstar;
  cl_mem sgnm;
  cl_mem spin;
  cl_mem spout;
  cl_mem ushock;
  cl_mem frac;
  cl_mem scr;
  cl_mem delp;
  cl_mem pold;
  cl_mem ind;
  cl_mem ind2;
  long narray;
  double Hsmallr;
  double Hsmallc;
  double Hgamma;
  long Hniter_riemann;
  long Hnvar;
  long Hnxyt;
} Args_t;

// Args_t est une structure que l'on place en memoire de constante sur
// le device et qui va contenir les arguments de riemann. on les
// transmets en bloc en une fois et les differents kernels pourront y
// acceder.
void
oclRiemann(cl_mem qleft, cl_mem qright,
           cl_mem qgdnv, cl_mem rl,
           cl_mem ul, cl_mem pl, cl_mem cl,
           cl_mem wl, cl_mem rr, cl_mem ur,
           cl_mem pr, cl_mem cr, cl_mem wr,
           cl_mem ro, cl_mem uo, cl_mem po,
           cl_mem co, cl_mem wo,
           cl_mem rstar, cl_mem ustar,
           cl_mem pstar, cl_mem cstar,
           cl_mem sgnm, cl_mem spin,
           cl_mem spout, cl_mem ushock,
           cl_mem frac, cl_mem scr,
           cl_mem delp, cl_mem pold,
           cl_mem ind, cl_mem ind2,
           const long narray,
           const double Hsmallr,
           const double Hsmallc, const double Hgamma, const long Hniter_riemann, const long Hnvar, const long Hnxyt)
{
  // Local variables
  Args_t k;
  cl_mem K;
  cl_int err = 0;

  WHERE("riemann");
  k.qleft = qleft;
  k.qright = qright;
  k.qgdnv = qgdnv;
  k.rl = rl;
  k.ul = ul;
  k.pl = pl;
  k.cl = cl;
  k.wl = wl;
  k.rr = rr;
  k.ur = ur;
  k.pr = pr;
  k.cr = cr;
  k.wr = wr;
  k.ro = ro;
  k.uo = uo;
  k.po = po;
  k.co = co;
  k.wo = wo;
  k.rstar = rstar;
  k.ustar = ustar;
  k.pstar = pstar;
  k.cstar = cstar;
  k.sgnm = sgnm;
  k.spin = spin;
  k.spout = spout;
  k.ushock = ushock;
  k.frac = frac;
  k.scr = scr;
  k.delp = delp;
  k.pold = pold;
  k.ind = ind;
  k.ind2 = ind2;
  k.narray = narray;
  k.Hsmallr = Hsmallr;
  k.Hsmallc = Hsmallc;
  k.Hgamma = Hgamma;
  k.Hniter_riemann = Hniter_riemann;
  k.Hnvar = Hnvar;
  k.Hnxyt = Hnxyt;

//   cudaMemcpyToSymbol(K, &k, sizeof(Args_t), 0, cudaMemcpyHostToDevice);
//   CheckErr("cudaMemcpyToSymbol");
//   // 64 threads donnent le meilleur rendement compte-tenu de la complexite du kernel
//   SetBlockDims(narray, 64, block, grid);
//   // Pressure, density and velocity
//   Loop1KcuRiemann <<< grid, block >>> ();
//   CheckErr("Avant synchronize Loop1KcuRiemann");
//   cudaThreadSynchronize();
//   CheckErr("After synchronize Loop1KcuRiemann");
//   // other passive variables
//   if (Hnvar > IP + 1) {
//     Loop10KcuRiemann <<< grid, block >>> ();
//     cudaThreadSynchronize();
//     CheckErr("After synchronize Loop10KcuRiemann");
//   }

  // Ici la creation se fait en dupliquant directement la
  // structure. Pas besoin de faire un write ensuite.
  K = clCreateBuffer(ctx, CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY, sizeof(k), &k, &err);
  oclCheckErr(err, "clCreateBuffer");

  OCLINITARG;

//   OCLRESETARG;
//   OCLSETARG(ker[Init1KcuRiemann], K);
//   OCLSETARG(ker[Init1KcuRiemann], k.qleft);
//   OCLSETARG(ker[Init1KcuRiemann], k.qright);
//   OCLSETARG(ker[Init1KcuRiemann], k.qgdnv);
//   OCLSETARG(ker[Init1KcuRiemann], k.rl);
//   OCLSETARG(ker[Init1KcuRiemann], k.ul);
//   OCLSETARG(ker[Init1KcuRiemann], k.pl);
//   OCLSETARG(ker[Init1KcuRiemann], k.cl);
//   OCLSETARG(ker[Init1KcuRiemann], k.wl);
//   OCLSETARG(ker[Init1KcuRiemann], k.rr);
//   OCLSETARG(ker[Init1KcuRiemann], k.ur);
//   oclLaunchKernel(ker[Init1KcuRiemann], cqueue, 1, THREADSSZ);

//   OCLRESETARG;
//   OCLSETARG(ker[Init2KcuRiemann], K);
//   OCLSETARG(ker[Init2KcuRiemann], k.pr);
//   OCLSETARG(ker[Init2KcuRiemann], k.cr);
//   OCLSETARG(ker[Init2KcuRiemann], k.wr);
//   OCLSETARG(ker[Init2KcuRiemann], k.ro);
//   OCLSETARG(ker[Init2KcuRiemann], k.uo);
//   OCLSETARG(ker[Init2KcuRiemann], k.po);
//   OCLSETARG(ker[Init2KcuRiemann], k.co);
//   OCLSETARG(ker[Init2KcuRiemann], k.wo);
//   OCLSETARG(ker[Init2KcuRiemann], k.rstar);
//   OCLSETARG(ker[Init2KcuRiemann], k.ustar);
//   oclLaunchKernel(ker[Init2KcuRiemann], cqueue, 1, THREADSSZ);

//   OCLRESETARG;
//   OCLSETARG(ker[Init3KcuRiemann], K);
//   OCLSETARG(ker[Init3KcuRiemann], k.pstar);
//   OCLSETARG(ker[Init3KcuRiemann], k.cstar);
//   OCLSETARG(ker[Init3KcuRiemann], k.sgnm);
//   OCLSETARG(ker[Init3KcuRiemann], k.spin);
//   OCLSETARG(ker[Init3KcuRiemann], k.spout);
//   OCLSETARG(ker[Init3KcuRiemann], k.ushock);
//   OCLSETARG(ker[Init3KcuRiemann], k.frac);
//   OCLSETARG(ker[Init3KcuRiemann], k.scr);
//   OCLSETARG(ker[Init3KcuRiemann], k.delp);
//   OCLSETARG(ker[Init3KcuRiemann], k.pold);
//   oclLaunchKernel(ker[Init3KcuRiemann], cqueue, 1, THREADSSZ);

//   OCLRESETARG;
//   OCLSETARG(ker[Init4KcuRiemann], K);
//   OCLSETARG(ker[Init4KcuRiemann], k.ind);
//   OCLSETARG(ker[Init4KcuRiemann], k.ind2);
//   oclLaunchKernel(ker[Init4KcuRiemann], cqueue, 1, THREADSSZ);

//   OCLRESETARG;
//   OCLSETARG(ker[Loop1KcuRiemann], k.qleft);
//   OCLSETARG(ker[Loop1KcuRiemann], k.qright);
//   OCLSETARG(ker[Loop1KcuRiemann], k.sgnm);
//   OCLSETARG(ker[Loop1KcuRiemann], k.qgdnv);
//   OCLSETARG(ker[Loop1KcuRiemann], k.Hnxyt);
//   OCLSETARG(ker[Loop1KcuRiemann], k.narray);
//   OCLSETARG(ker[Loop1KcuRiemann], k.Hsmallc);
//   OCLSETARG(ker[Loop1KcuRiemann], k.Hgamma);
//   OCLSETARG(ker[Loop1KcuRiemann], k.Hsmallr);
//   OCLSETARG(ker[Loop1KcuRiemann], k.Hniter_riemann);
  // fprintf(stderr, "Lancement de Loop1KcuRiemann\n");
  OCLSETARG10(ker[Loop1KcuRiemann], k.qleft, k.qright, k.sgnm, k.qgdnv, k.Hnxyt, k.narray, k.Hsmallc, k.Hgamma, k.Hsmallr,
              k.Hniter_riemann);
  oclLaunchKernel(ker[Loop1KcuRiemann], cqueue, narray, THREADSSZ, __FILE__, __LINE__);
  // exit(123);
  if (Hnvar > IP + 1) {
    OCLRESETARG;
    OCLSETARG(ker[Loop10KcuRiemann], K);
    OCLSETARG(ker[Loop10KcuRiemann], k.qleft);
    OCLSETARG(ker[Loop10KcuRiemann], k.qright);
    OCLSETARG(ker[Loop10KcuRiemann], k.sgnm);
    OCLSETARG(ker[Loop10KcuRiemann], k.qgdnv);
    OCLSETARG(ker[Loop10KcuRiemann], k.narray);
    OCLSETARG(ker[Loop10KcuRiemann], k.Hnvar);
    OCLSETARG(ker[Loop10KcuRiemann], k.Hnxyt);
    // fprintf(stderr, "Lancement de Loop10KcuRiemann\n");
    oclLaunchKernel(ker[Loop10KcuRiemann], cqueue, narray, THREADSSZ, __FILE__, __LINE__);
  }
  err = clReleaseMemObject(K);
  oclCheckErr(err, "clReleaseMemObject");
}                               // riemann


//EOF
