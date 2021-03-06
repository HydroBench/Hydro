#ifndef PRECISION_H
#define PRECISION_H

#ifndef PREC_SP
typedef double real_t;

constexpr real_t one = 1.0L;
constexpr real_t two = 2.0L;
constexpr real_t half = 0.5L;
constexpr real_t zero = 0.0L;
constexpr real_t hundred = 100.0L;
constexpr real_t precision = 1.e-6;

#else
typedef float real_t;

constexpr real_t one = 1.0f;
constexpr real_t two = 2.0f;
constexpr real_t half = 0.5f;
constexpr real_t zero = 0.0f;
constexpr real_t hundred = 100.0f;
constexpr real_t PRECISION = 1.e-6f;

#endif

typedef real_t *__restrict__ Preal_t;

#endif
