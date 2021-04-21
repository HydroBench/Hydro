# Explanation of the DataFlow on Hydro based on the code

For each routine, we list the field input variables and the field output variables. This should help to see how each routine depend from the others

Each variables are member variables of tiles, except uOld which is global to the tiles.

- boundary: uOld* <- uOld*
- gather: u <- uOld*
- constprim: q(*), e <- u
- eos: q(p), c <- q(d), e
- Slope: dq <- q
- trace: qxp,qxm <- q,d
- qleftright: qleft, qright <- qxp, qxm
- riemann: qgdnv <- qleft, qright
- compflx: flux <- qgdnv
- update: uOld* <- u, flux

- computedt1: q(*),e <- uOld*
- eos: q(p),c <- q(d), e
- computedt2: <- q(*), c

I put eos twice since it is really called twice times.

So the wall loop takes uOld* and return an update uOld* and each step depends from the previous one
(So for sycl, we need to wait!)


