
#ifndef ENUMDEFS_HPP
#define ENUMDEFS_HPP

typedef enum _tileNeighbour {
	UP_TILE = 0, DOWN_TILE, LEFT_TILE, RIGHT_TILE, NEIGHBOUR_TILE
} tileNeighbour_t;

typedef enum _boxBoundary {
	XMIN_D = 0, XMAX_D, YMIN_D, YMAX_D, UP_D, DOWN_D, LEFT_D, RIGHT_D, MAXBOX_D
} boxBoundary_t;

typedef enum _tileSpan {
	TILE_FULL, TILE_INTERIOR
} tileSpan_t;

typedef enum _godunovDir {
	X_SCAN = 0, Y_SCAN
} godunovDir_t;

typedef enum _godunovVars {
	IP_VAR = 0, ID_VAR, IU_VAR, IV_VAR, NB_VAR
} godunovVars_t;

typedef enum _godunovScheme {
	SCHEME_MUSCL, SCHEME_PLMDE, SCHEME_COLLELA
} godunovScheme_t;

typedef enum _protectionMode {
	PROT_LENGTH = 1,
	PROT_READ, PROT_WRITE
} protectionMode_t;

typedef enum _funcNames {
	FNM_TILE_GATCON = 0,
	FNM_END
} funcNames_t;

typedef enum _loopNames {
	LOOP_GODUNOV = 0,
	LOOP_UPDATE,
	LOOP_END
} loopNames_t;

#endif
