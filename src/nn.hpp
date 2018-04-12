#pragma once

class Matrix;

float sigmoid(float x);
float isigmoid(float n);

class NeuralNetwork {
public:
    NeuralNetwork(int in, int hid, int out);
    ~NeuralNetwork();

    float* guess(float* input);
    float* feedForward(Matrix& input);

    void teach(float* inputs, float* targets);

    float getLearningRate() { return m_learningRate; }
    void setLearningRate(float rate) { m_learningRate = rate; }

private:
    int m_inputNodes;
    int m_hiddenNodes;
    int m_outputNodes;

    float m_learningRate;

    Matrix* m_inputWeights;
    Matrix* m_hiddenWeights;

    Matrix* m_inputBias;
    Matrix* m_hiddenBias;
};
