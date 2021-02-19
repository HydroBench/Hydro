//
// (C) Guillaume.Colin-de-Verdiere at CEA.Fr
//
#ifndef IMAGE_H
#define IMAGE_H
//
#include "parametres.h"

#if WITHPNG > 0
void pngWriteFile(char *name, hydroparam_t *H);
void pngProcess(hydroparam_t *H, hydrovar_t *Hv);
void pngCloseFile(hydroparam_t *H);
void pngWriteFile(char *name, hydroparam_t *H);
void pngProcess(hydroparam_t *H, hydrovar_t *Hv);
#endif

void ppmWriteFile(char *name, hydroparam_t *H);
void ppmProcess(hydroparam_t *H, hydrovar_t *Hv);
void ppmCloseFile(hydroparam_t *H);
void ppmWriteFile(char *name, hydroparam_t *H);
void ppmProcess(hydroparam_t *H, hydrovar_t *Hv);
 
void imgCloseFile(hydroparam_t *H);
void imgFillGap(int curx, int cury, int nx, int ny, int Iptr[4], int *cpt, int32_t imgSizeX,
                int32_t imgSizeY);

real_t reduceMaxAndBcast(real_t dt);
void getMaxVarValues(real_t *mxP, real_t *mxD, real_t *mxUV, hydroparam_t *H, hydrovar_t *Hv);
void DumpImage(int n, hydroparam_t *H, hydrovar_t *Hv);
//
#define PIXRGBA 4

#endif // IMAGE_H
// EOF
