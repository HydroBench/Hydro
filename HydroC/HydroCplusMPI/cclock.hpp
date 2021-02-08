#ifndef CCLOCK_H
#define CCLOCK_H

#include <chrono>

// Do it C++ way

class Custom_Timer {

    static Custom_Timer *timer_singleton;

    typedef std::chrono::nanoseconds ns;
    typedef std::chrono::high_resolution_clock::time_point my_time_t;
    my_time_t m_time_point_start;

     Custom_Timer()
    :m_time_point_start(std::chrono::high_resolution_clock::now()) {
    } static Custom_Timer *get();

 public:
    static double dcclock();
    static my_time_t cclock();
    static double ccelaps(const my_time_t &, const my_time_t &);
    static void convertToHuman(char *s, double t);
};

#endif				// CCLOCK_H
