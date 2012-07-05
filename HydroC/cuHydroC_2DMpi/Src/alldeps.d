SplitSurface.o: SplitSurface.c SplitSurface.h 
cmpflx.o: cmpflx.c parametres.h utils.h cmpflx.h 
compute_deltat.o: compute_deltat.c parametres.h compute_deltat.h utils.h equation_of_state.h 
conservar.o: conservar.c parametres.h utils.h conservar.h 
constoprim.o: constoprim.c parametres.h constoprim.h utils.h 
equation_of_state.o: equation_of_state.c equation_of_state.h utils.h parametres.h 
hydro_funcs.o: hydro_funcs.c utils.h parametres.h hydro_utils.h hydro_funcs.h 
hydro_godunov.o: hydro_godunov.c parametres.h hydro_godunov.h hydro_funcs.h utils.h make_boundary.h cmpflx.h conservar.h equation_of_state.h qleftright.h constoprim.h riemann.h trace.h slope.h 
hydro_utils.o: hydro_utils.c hydro_utils.h 
main.o: main.c parametres.h hydro_funcs.h vtkfile.h cuComputeDeltat.h hydro_godunov.h cuHydroGodunov.h utils.h GetDevice.h perfcnt.h 
make_boundary.o: make_boundary.c parametres.h make_boundary.h utils.h 
parametres.o: parametres.c SplitSurface.h parametres.h 
qleftright.o: qleftright.c parametres.h utils.h qleftright.h 
riemann.o: riemann.c parametres.h utils.h riemann.h 
slope.o: slope.c parametres.h utils.h slope.h 
trace.o: trace.c parametres.h utils.h trace.h 
utils.o: utils.c utils.h parametres.h 
vtkfile.o: vtkfile.c parametres.h utils.h vtkfile.h SplitSurface.h 
GetDevice.o : GetDevice.cu gridfuncs.h GetDevice.h 
cuCmpflx.o : cuCmpflx.cu cuCmpflx.h utils.h gridfuncs.h perfcnt.h parametres.h 
cuComputeDeltat.o : cuComputeDeltat.cu utils.h gridfuncs.h perfcnt.h cuComputeDeltat.h cuHydroGodunov.h cuEquationOfState.h parametres.h 
cuConservar.o : cuConservar.cu utils.h gridfuncs.h perfcnt.h cuConservar.h parametres.h 
cuConstoprim.o : cuConstoprim.cu utils.h gridfuncs.h perfcnt.h parametres.h cuConstoprim.h 
cuEquationOfState.o : cuEquationOfState.cu utils.h gridfuncs.h perfcnt.h cuEquationOfState.h parametres.h 
cuHydroGodunov.o : cuHydroGodunov.cu trace.h cuCmpflx.h utils.h cmpflx.h slope.h gridfuncs.h riemann.h cuMakeBoundary.h cuHydroGodunov.h equation_of_state.h cuEquationOfState.h cuConservar.h cuQleftright.h cuTrace.h parametres.h qleftright.h cuConstoprim.h hydro_funcs.h make_boundary.h cuSlope.h constoprim.h cuRiemann.h conservar.h 
cuMakeBoundary.o : cuMakeBoundary.cu utils.h gridfuncs.h perfcnt.h cuMakeBoundary.h parametres.h 
cuQleftright.o : cuQleftright.cu utils.h gridfuncs.h cuQleftright.h parametres.h 
cuRiemann.o : cuRiemann.cu utils.h gridfuncs.h perfcnt.h parametres.h cuRiemann.h 
cuSlope.o : cuSlope.cu utils.h gridfuncs.h perfcnt.h parametres.h cuSlope.h 
cuTrace.o : cuTrace.cu utils.h gridfuncs.h perfcnt.h cuTrace.h parametres.h 
gridfuncs.o : gridfuncs.cu utils.h gridfuncs.h parametres.h 
perfcnt.o : perfcnt.cu gridfuncs.h perfcnt.h 
