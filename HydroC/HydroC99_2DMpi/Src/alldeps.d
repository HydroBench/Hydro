SplitSurface.o: SplitSurface.c SplitSurface.h 
cmpflx.o: cmpflx.c parametres.h utils.h cmpflx.h perfcnt.h 
compute_deltat.o: compute_deltat.c parametres.h compute_deltat.h utils.h perfcnt.h equation_of_state.h 
conservar.o: conservar.c parametres.h utils.h conservar.h perfcnt.h 
constoprim.o: constoprim.c parametres.h constoprim.h utils.h perfcnt.h 
equation_of_state.o: equation_of_state.c equation_of_state.h utils.h parametres.h perfcnt.h 
hydro_funcs.o: hydro_funcs.c utils.h parametres.h hydro_utils.h hydro_funcs.h 
hydro_godunov.o: hydro_godunov.c hmpp.h constoprim.h utils.h parametres.h equation_of_state.h slope.h trace.h qleftright.h riemann.h cmpflx.h conservar.h hydro_godunov.h hydro_funcs.h make_boundary.h 
hydro_utils.o: hydro_utils.c hydro_utils.h 
main.o: main.c parametres.h hydro_funcs.h vtkfile.h compute_deltat.h hydro_godunov.h perfcnt.h utils.h 
make_boundary.o: make_boundary.c parametres.h make_boundary.h perfcnt.h utils.h 
parametres.o: parametres.c parametres.h SplitSurface.h 
perfcnt.o: perfcnt.c perfcnt.h 
qleftright.o: qleftright.c parametres.h utils.h qleftright.h hmpp.h constoprim.h equation_of_state.h slope.h trace.h riemann.h cmpflx.h conservar.h 
riemann.o: riemann.c parametres.h perfcnt.h utils.h riemann.h hmpp.h constoprim.h equation_of_state.h slope.h trace.h qleftright.h cmpflx.h conservar.h 
slope.o: slope.c parametres.h utils.h slope.h hmpp.h constoprim.h equation_of_state.h trace.h qleftright.h riemann.h cmpflx.h conservar.h perfcnt.h 
trace.o: trace.c parametres.h utils.h trace.h hmpp.h constoprim.h equation_of_state.h slope.h qleftright.h riemann.h cmpflx.h conservar.h perfcnt.h 
utils.o: utils.c utils.h parametres.h 
vtkfile.o: vtkfile.c parametres.h utils.h vtkfile.h SplitSurface.h 
