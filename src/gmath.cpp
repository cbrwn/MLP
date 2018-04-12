#include "gmath.h"

float map(float v, float vMin, float vMax, float oMin, float oMax)
{
    float vDif = vMax - vMin;
    float oDif = oMax - oMin;

    float perc = (v - vMin) / vDif;

    return oMin + (perc * oDif);
}
