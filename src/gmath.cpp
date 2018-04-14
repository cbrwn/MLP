#include "gmath.h"

#include <cstdlib>

float randBetween(float min, float max)
{
    const float precision = 10000.0f;

    float percentage = (rand() % (int)precision) / precision;

    return min + (max - min) * percentage;
}

float map(float v, float vMin, float vMax, float oMin, float oMax)
{
    // get the difference between the minimums and maximums
    float vDif = vMax - vMin;
    float oDif = oMax - oMin;

    // get the percentage of the value between minimum and maximum
    float perc = (v - vMin) / vDif;

    // apply that percentage to the new range
    return oMin + (perc * oDif);
}
