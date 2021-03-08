
#ifndef ENUMDEFS_HPP
#define ENUMDEFS_HPP

enum tileNeighbour_t { UP_TILE = 0, DOWN_TILE, LEFT_TILE, RIGHT_TILE, NEIGHBOUR_TILE };

enum boxBoundary_t { XMIN_D = 0, XMAX_D, YMIN_D, YMAX_D, UP_D, DOWN_D, LEFT_D, RIGHT_D, MAXBOX_D };

enum tileSpan_t { TILE_FULL, TILE_INTERIOR };

enum godunovDir_t { X_SCAN = 0, Y_SCAN };

enum godunovVars_t { IP_VAR = 0, ID_VAR, IU_VAR, IV_VAR, NB_VAR };

enum godunovScheme_t { SCHEME_MUSCL, SCHEME_PLMDE, SCHEME_COLLELA };

enum protectionMode_t { PROT_LENGTH = 1, PROT_READ, PROT_WRITE };

enum funcNames_t { FNM_TILE_GATCON = 0, FNM_END };

enum loopNames_t { LOOP_GODUNOV = 0, LOOP_UPDATE, LOOP_END };

#endif
