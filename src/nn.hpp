#pragma once

class Matrix;

float sigmoid(float x);

class NeuralNetwork {
public:
    NeuralNetwork(int in, int hid, int out);
    ~NeuralNetwork();

    float* feedForward(float* input);
    float* feedForward(Matrix& input);

private:
    int m_inputNodes;
    int m_hiddenNodes;
    int m_outputNodes;

    Matrix* m_inputWeights;
    Matrix* m_hiddenWeights;

    Matrix* m_inputBias;
    Matrix* m_hiddenBias;
};
