//
// (C) Guillaume.Colin-de-Verdiere at CEA.Fr
//
#ifndef SPLITSURFACE_H
#define SPLITSURFACE_H
//
typedef enum
{ XMIN_D, XMAX_D, YMIN_D, YMAX_D, UP_D, DOWN_D, LEFT_D, RIGHT_D,
  MAXBOX_D
} dir_t;


void
CalcSubSurface (int xmin, int xmax,
		int ymin, int ymax, int pmin, int pmax, int level,
		int box[MAXBOX_D], int mype, int pass);
//
#endif
//EOF
