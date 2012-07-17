SplitSurface.o: SplitSurface.c SplitSurface.h 
hydro_funcs.o: hydro_funcs.c utils.h parametres.h hydro_utils.h hydro_funcs.h 
hydro_utils.o: hydro_utils.c hydro_utils.h 
main.o: main.c parametres.h hydro_funcs.h vtkfile.h oclComputeDeltat.h oclHydroGodunov.h utils.h oclInit.h 
oclCmpflx.o: oclCmpflx.c parametres.h utils.h oclCmpflx.h oclInit.h ocltools.h oclerror.h 
oclComputeDeltat.o: oclComputeDeltat.c parametres.h oclComputeDeltat.h oclHydroGodunov.h utils.h oclEquationOfState.h oclInit.h ocltools.h oclerror.h oclReduce.h 
oclConservar.o: oclConservar.c parametres.h utils.h oclConservar.h oclInit.h ocltools.h oclerror.h 
oclConstoprim.o: oclConstoprim.c parametres.h utils.h oclConstoprim.h oclInit.h ocltools.h oclerror.h 
oclEquationOfState.o: oclEquationOfState.c oclEquationOfState.h utils.h parametres.h oclInit.h ocltools.h oclerror.h 
oclHydroGodunov.o: oclHydroGodunov.c parametres.h hydro_funcs.h utils.h oclInit.h ocltools.h oclerror.h oclHydroGodunov.h oclConservar.h oclConstoprim.h oclSlope.h oclTrace.h oclEquationOfState.h oclQleftright.h oclRiemann.h oclCmpflx.h oclMakeBoundary.h 
oclInit.o: oclInit.c oclInit.h ocltools.h oclerror.h 
oclMakeBoundary.o: oclMakeBoundary.c parametres.h oclMakeBoundary.h utils.h oclInit.h ocltools.h oclerror.h 
oclQleftright.o: oclQleftright.c parametres.h utils.h oclQleftright.h oclInit.h ocltools.h oclerror.h 
oclReduce.o: oclReduce.c oclReduce.h oclInit.h ocltools.h oclerror.h 
oclRiemann.o: oclRiemann.c parametres.h utils.h oclRiemann.h oclInit.h ocltools.h oclerror.h 
oclSlope.o: oclSlope.c parametres.h utils.h oclSlope.h oclInit.h ocltools.h oclerror.h 
oclTrace.o: oclTrace.c parametres.h utils.h oclTrace.h oclInit.h ocltools.h oclerror.h 
oclerror.o: oclerror.c oclerror.h 
ocltools.o: ocltools.c ocltools.h oclerror.h 
parametres.o: parametres.c SplitSurface.h parametres.h 
utils.o: utils.c utils.h parametres.h 
vtkfile.o: vtkfile.c parametres.h utils.h vtkfile.h SplitSurface.h 
GetDevice.o : GetDevice.cu gridfuncs.h GetDevice.h 
gridfuncs.o : gridfuncs.cu utils.h gridfuncs.h parametres.h 
perfcnt.o : perfcnt.cu gridfuncs.h perfcnt.h 
