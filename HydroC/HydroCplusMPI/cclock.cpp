#include "cclock.hpp"
#include <cstdio>

Custom_Timer *Custom_Timer::timer_singleton = 0;

Custom_Timer *Custom_Timer::get()
{
    if (timer_singleton == 0) {
	timer_singleton = new Custom_Timer();
    }
    return timer_singleton;
}

double Custom_Timer::dcclock()
{
    auto actual = std::chrono::high_resolution_clock::now();
    return std::chrono::duration < double >(actual - get()->m_time_point_start)
    .count();
}

Custom_Timer::my_time_t Custom_Timer::cclock()
{
    return std::chrono::high_resolution_clock::now();
}

void Custom_Timer::convertToHuman(char *s, double t)
{
    int subsec;
    int days, hours, minutes, secondes;
    double curt = t;

    s[0] = 0;
    days = (int)(curt / (3600 * 24));
    curt -= (days * 3600 * 24);
    hours = (int)(curt / 3600);
    curt -= (hours * 3600);
    minutes = (int)(curt / 60);
    curt -= (minutes * 60);
    secondes = (int)(curt);
    subsec = (int)(((float)(curt) - (float)(secondes)) * 100);
    if (days)
	sprintf(s, "[%d:]", days);
    sprintf(s, "%s%02d:%02d:%02d.%d", s, hours, minutes, secondes, subsec);
}

//
