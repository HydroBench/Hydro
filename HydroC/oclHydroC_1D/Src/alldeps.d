cmpflx.o: cmpflx.c parametres.h utils.h cmpflx.h 
compute_deltat.o: compute_deltat.c parametres.h compute_deltat.h utils.h equation_of_state.h 
conservar.o: conservar.c parametres.h utils.h conservar.h 
constoprim.o: constoprim.c parametres.h constoprim.h utils.h 
equation_of_state.o: equation_of_state.c equation_of_state.h utils.h parametres.h 
hydro_funcs.o: hydro_funcs.c utils.h parametres.h hydro_utils.h hydro_funcs.h 
hydro_godunov.o: hydro_godunov.c parametres.h hydro_godunov.h hydro_funcs.h utils.h make_boundary.h cmpflx.h conservar.h equation_of_state.h qleftright.h constoprim.h riemann.h trace.h slope.h 
hydro_utils.o: hydro_utils.c hydro_utils.h 
main.o: main.c parametres.h hydro_funcs.h vtkfile.h oclComputeDeltat.h hydro_godunov.h oclHydroGodunov.h utils.h oclInit.h 
make_boundary.o: make_boundary.c parametres.h make_boundary.h utils.h 
oclCmpflx.o: oclCmpflx.c parametres.h utils.h oclCmpflx.h oclInit.h ocltools.h oclerror.h 
oclComputeDeltat.o: oclComputeDeltat.c parametres.h oclComputeDeltat.h oclHydroGodunov.h utils.h oclEquationOfState.h oclInit.h ocltools.h oclerror.h oclReduce.h 
oclConservar.o: oclConservar.c parametres.h utils.h oclConservar.h oclInit.h ocltools.h oclerror.h 
oclConstoprim.o: oclConstoprim.c parametres.h utils.h oclConstoprim.h oclInit.h ocltools.h oclerror.h 
oclEquationOfState.o: oclEquationOfState.c oclEquationOfState.h utils.h parametres.h oclInit.h ocltools.h oclerror.h 
oclHydroGodunov.o: oclHydroGodunov.c parametres.h hydro_funcs.h utils.h make_boundary.h cmpflx.h conservar.h equation_of_state.h qleftright.h constoprim.h riemann.h trace.h slope.h oclInit.h ocltools.h oclerror.h oclHydroGodunov.h oclConservar.h oclConstoprim.h oclSlope.h oclTrace.h oclEquationOfState.h oclQleftright.h oclRiemann.h oclCmpflx.h oclMakeBoundary.h 
oclInit.o: oclInit.c oclInit.h ocltools.h oclerror.h 
oclMakeBoundary.o: oclMakeBoundary.c parametres.h oclMakeBoundary.h utils.h oclInit.h ocltools.h oclerror.h 
oclQleftright.o: oclQleftright.c parametres.h utils.h oclQleftright.h oclInit.h ocltools.h oclerror.h 
oclReduce.o: oclReduce.c oclReduce.h oclInit.h ocltools.h oclerror.h 
oclRiemann.o: oclRiemann.c parametres.h utils.h oclRiemann.h oclInit.h ocltools.h oclerror.h 
oclSlope.o: oclSlope.c parametres.h utils.h oclSlope.h oclInit.h ocltools.h oclerror.h 
oclTrace.o: oclTrace.c parametres.h utils.h oclTrace.h oclInit.h ocltools.h oclerror.h 
oclerror.o: oclerror.c oclerror.h 
ocltools.o: ocltools.c ocltools.h oclerror.h 
parametres.o: parametres.c parametres.h 
qleftright.o: qleftright.c parametres.h utils.h qleftright.h 
riemann.o: riemann.c parametres.h utils.h riemann.h 
slope.o: slope.c parametres.h utils.h slope.h 
trace.o: trace.c parametres.h utils.h trace.h 
utils.o: utils.c utils.h parametres.h 
vtkfile.o: vtkfile.c parametres.h utils.h vtkfile.h 
