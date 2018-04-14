#pragma once

#define PI 3.141592653589793238f

inline int sign(float n) { return n < 0 ? -1 : 1; }
inline int sign(int n) { return n < 0 ? -1 : 1; }

inline float absf(float n) { return n < 0 ? -n : n; }

/***
 * @brief Returns a random float in a specified range
 * @param min Minimum value of the random number
 * @param max Maximum value of the random number
 * @return A random number within the range
 */
float randBetween(float min, float max);

/***
 * @brief Takes a value and its range and then scales it to fit another range
 *          Works like Processing's map function:
 *          https://processing.org/reference/map_.html
 * @param v Value to scale
 * @param vMin Original minimum value
 * @param vMax Original maxmimum value
 * @param oMin New minimum value
 * @param oMax New maximum value
 * @return The scaled number
 */
float map(float v, float vMin, float vMax, float oMin, float oMax);

