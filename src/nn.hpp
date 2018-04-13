#pragma once

class Matrix;

/***
 * @brief Sigmoid function used to 'normalize' the outputs of each neuron
 * @param x Value to normalize
 * @return Normalized value
 */
float sigmoid(float x);
/**
 * @brief Inverse of sigmoid, assuming the number has already been normalized
 * @param n Normalized value
 * @return Result of the derivative function
 */
float isigmoid(float n);

class NeuralNetwork {
public:
    /***
     * @brief Creates a neural network with a specified number of inputs,
     *          hidden nodes and outputs
     * @param in Number of inputs
     * @param hid Number of nodes in the hidden layer
     * @param out Number of outputs
     */
    NeuralNetwork(int in, int hid, int out);
    ~NeuralNetwork();

    /***
     * @brief Uses the current weights to get a result from inputs
     * @param input Inputs to use to get the result
     * @return Array of floats containing the outputs
     */
    float* guess(float const* input);
    /***
     * @brief Takes a Matrix and uses the feed forward algorithm to get a result
     * @param input Matrix of inputs
     * @return Array of floats containing the outputs
     */
    float* feedForward(Matrix& input);

    /***
     * @brief Takes a single set of inputs and targets and uses these to
     *          adjust weights in order to "learn"
     * @param inputs Inputs which should result in targets
     * @param targets Desired output from inputs
     */
    void teach(float const* inputs, float const* targets);

    // learning rate getter/setter
    float getLearningRate() { return m_learningRate; }
    void setLearningRate(float rate) { m_learningRate = rate; }

    Matrix* getInputWeights() { return m_inputWeights; }
    Matrix* getHiddenWeights() { return m_hiddenWeights; }
    Matrix* getInputBias() { return m_inputBias; }
    Matrix* getHiddenBias() { return m_hiddenBias; }

    int getHiddenNodes() { return m_hiddenNodes; }

private:
    // numbers of nodes
    int m_inputNodes;
    int m_hiddenNodes;
    int m_outputNodes;

    // how the weight changes are scaled
    float m_learningRate;

    // weights
    Matrix* m_inputWeights;
    Matrix* m_hiddenWeights;

    // biases
    Matrix* m_inputBias;
    Matrix* m_hiddenBias;
};
