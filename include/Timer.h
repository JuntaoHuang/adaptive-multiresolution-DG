#pragma once

#include "libs.h"

class Timer 
{
public:
    Timer() { time_start = std::chrono::high_resolution_clock::now(); };
    ~Timer() {};

    void time(const char* msg);

    // reset start time to be now
    void reset();

protected:
    std::chrono::high_resolution_clock::time_point time_start;
    std::string message_;
};