//
// (C) Guillaume.Colin-de-Verdiere at CEA.Fr
//
#ifndef PERFCNT_H
#define PERFCNT_H
//

// number of floatting ops per operation type
#define FLOPSARI  1 
// + - * -x

#define FLOPSSQR  2 
// / sqrt 

#define FLOPSMIN  1 
// min, max, sign, abs 

#define FLOPSTRA  5 
// exp, sin, cos 

extern long flopsAri, flopsSqr, flopsMin, flopsTra;
extern double MflopsSUM;
extern long nbFLOPS;

#define FLOPS(a, b, c, d) do { flopsAri+=(a); flopsSqr+=(b); flopsMin+=(c); flopsTra+=(d); } while (0)

//
#endif
//EOF
