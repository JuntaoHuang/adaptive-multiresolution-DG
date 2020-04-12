#include "Timer.h"

void Timer::time(const char* msg)
{
	auto t2 = std::chrono::high_resolution_clock::now();
	std::cout << msg << ": " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - time_start).count()/1000. << " s" << std::endl;
}

void Timer::reset()
{
    time_start = std::chrono::high_resolution_clock::now();
}