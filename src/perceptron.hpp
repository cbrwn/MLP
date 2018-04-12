#pragma once

class Perceptron
{
public:
    explicit Perceptron(int count);
    ~Perceptron();

    int guess(float* inputs);
    void teach(float* inputs, int goal);

private:
    int m_inputCount;

    float m_learningRate;
    float* m_weights;
};
