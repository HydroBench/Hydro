//
// (C) Guillaume.Colin-de-Verdiere at CEA.Fr
//
#ifndef OPTIONS_H
#define OPTIONS_H

// This file controls compilation options inside the code

#define RIEMANNINREGS 1
#define USEMKL 0
#define ALIGNED 1

#define TILEUSER 1

#ifdef __MIC__
// #undef TILEUSER
// #define TILEUSER 0
#define TILEMIN 5
#define TILEAVG 32
#define TILESIZ 28
#else
#define TILEMIN 5
#define TILEAVG 128
#define TILESIZ 124
#endif

// #ifdef __INTEL_COMPILER
// #define USEINTRINSICS 1
// #else
// #define USEINTRINSICS 0
// #endif

#ifdef __MIC__
#define KNC 1
#undef WITHPNG
#else
#ifndef AVX
#define SSE 1
#endif
#endif

#define DOUBLE 1

#ifdef __MIC__
#define SCHEDULE schedule(dynamic)
static char * Schedule = "schedule(dynamic)";
#else
#define SCHEDULE schedule(guided)
static char * Schedule = "schedule(guided)";
#endif

#define WITH_TIMERS 0

#define TILE_PER_THREAD 4

#endif
//EOF
