//
// (C) Guillaume.Colin-de-Verdiere at CEA.Fr
//
#ifndef IMAGE_H
#define IMAGE_H
//
#include "parametres.h"

void pngWriteFile(char *name, hydroparam_t *H);
void pngProcess(hydroparam_t *H, hydrovar_t *Hv);
void pngCloseFile(hydroparam_t *H);
void getMaxVarValues(real_t *mxP, real_t *mxD, real_t *mxUV, hydroparam_t *H, hydrovar_t *Hv);
void pngFillGap(int curx, int cury, int nx, int ny, int Iptr[4], int *cpt, int32_t imgSizeX,
                int32_t imgSizeY);
void DumpImage(int n, hydroparam_t *H, hydrovar_t *Hv);
//
#endif
// EOF
