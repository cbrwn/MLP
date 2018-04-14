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
     * @brief Creates a neural network with a set amount of input nodes,
     *          output nodes, hidden layers, and nodes in each hidden layer
     * @param in Number of inputs
     * @param hid Number of hidden layers
     * @param nodes Array representing how many nodes in each hidden layer
     * @param out Number of outputs
     */
    NeuralNetwork(int in, int hid, int const* nodes, int out);
    ~NeuralNetwork();

    /***
     * @brief Uses the current weights to get a result from inputs
     * @param input Inputs to use to get the result
     * @return Array of floats containing the outputs
     */
    void guess(float const* input, float* output);

    /***
     * @brief Takes a single set of inputs and targets and uses these to
     *          adjust weights in order to "learn"
     * @param inputs Inputs which should result in targets
     * @param targets Desired output from inputs
     */
    void propagate(float const* inputs, float const* targets);

    // learning rate getter/setter
    float getLearningRate() { return m_learningRate; }
    void setLearningRate(float rate) { m_learningRate = rate; }

private:
    int m_inputNodes;
    int m_outputNodes;

    int m_hiddenLayers;
    int* m_hiddenNodeCount;

    // how the weight changes are scaled
    float m_learningRate;

    // arrays to matrix pointers where these values are stored
    Matrix** m_weights;
    Matrix** m_biases;
};
