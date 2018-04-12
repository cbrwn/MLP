#include "perceptron.hpp"

#include "gmath.h"
#include "random.h"

Perceptron::Perceptron(int count)
    : m_inputCount(count)
{
    m_learningRate = 0.1f;
    m_weights = new float[count];

    // randomly set weights
    for(int i = 0; i < count; ++i)
    {
        m_weights[i] = randBetween(-1.0f, 1.0f);
    }
}

Perceptron::~Perceptron()
{
    delete[] m_weights;
}

int Perceptron::guess(float* inputs)
{
    float sum = 0.0f;
    for(int i = 0; i < m_inputCount; ++i)
    {
        sum += inputs[i] * m_weights[i];
    }

    return sign(sum);
}

void Perceptron::teach(float* inputs, int goal)
{
    int g = guess(inputs);
    int error = goal - g;

    for(int i = 0; i < m_inputCount; ++i)
        m_weights[i] += error * inputs[i] * m_learningRate;
}
