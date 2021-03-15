#ifndef MORTON_HPP
#define MORTON_HPP

#include <cassert>
#include <cstdint>

static inline int32_t morton1(int32_t x_) {
    int32_t x = x_;
    assert(x <= 0xFFFF);
    x = (x | (x << 8)) & 0x00FF00FF;
    x = (x | (x << 4)) & 0x0F0F0F0F;
    x = (x | (x << 2)) & 0x33333333;
    x = (x | (x << 1)) & 0x55555555;
    return x;
};

static inline int32_t umorton1(int32_t x) {
    x = x & 0x55555555;
    x = (x | (x >> 1)) & 0x33333333;
    x = (x | (x >> 2)) & 0x0F0F0F0F;
    x = (x | (x >> 4)) & 0x00FF00FF;
    x = (x | (x >> 8)) & 0x0000FFFF;
    return x;
};

// morton2 - extract odd and even bits

static inline void umorton2(int32_t &x, int32_t &y, int32_t m) {
    int32_t z1;
    x = umorton1(m);
    z1 = m >> 1;
    y = umorton1(z1);
};

static inline int32_t morton2(int32_t x, int32_t y) { return morton1(x) | (morton1(y) << 1); };

#endif
