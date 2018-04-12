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
    float vDif = vMax - vMin;
    float oDif = oMax - oMin;

    float perc = (v - vMin) / vDif;

    return oMin + (perc * oDif);
}
