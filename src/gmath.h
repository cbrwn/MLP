#pragma once

#define PI 3.141592653589793238f

inline int sign(float n) { return n < 0 ? -1 : 1; }
inline int sign(int n) { return n < 0 ? -1 : 1; }

inline float abs(float n) { return n < 0 ? -n : n; }

float map(float v, float vMin, float vMax, float oMin, float oMax);

