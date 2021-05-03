#ifndef PRECISION_H
#define PRECISION_H

#ifndef PREC_SP
using real_t = double;

constexpr real_t one = 1.0L;
constexpr real_t two = 2.0L;
constexpr real_t my_half = 0.5L;
constexpr real_t zero = 0.0L;
constexpr real_t hundred = 100.0L;
constexpr real_t precision = 1.e-6;

#else
using real_t = float;

constexpr real_t one = 1.0f;
constexpr real_t two = 2.0f;
constexpr real_t my_half = 0.5f;
constexpr real_t zero = 0.0f;
constexpr real_t hundred = 100.0f;
constexpr real_t precision = 1.e-6f;

#endif

using Preal_t = real_t *__restrict__;

#endif
