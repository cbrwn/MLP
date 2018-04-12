#include "nn.hpp"

#include <cmath>
#include <iostream>

#include "matrix.hpp"

NeuralNetwork::NeuralNetwork(int in, int hid, int out)
    : m_inputNodes(in), m_hiddenNodes(hid), m_outputNodes(out)
{

    m_inputWeights = new Matrix(hid, in);
    m_hiddenWeights = new Matrix(out, hid);

    // start with random weights
    m_inputWeights->randomize();
    m_hiddenWeights->randomize();

    printf("Input Weights:\n");
    m_inputWeights->print("%.2f ");
    printf("Hidden Weights:\n");
    m_hiddenWeights->print("%.2f ");

    m_inputBias = new Matrix(hid, 1);
    m_hiddenBias = new Matrix(out, 1);

    // set the biases to 1
    for(int i = 0; i < hid; ++i)
        (*m_inputBias)[i][0] = 1.0f;
    for(int i = 0; i < out; ++i)
        (*m_hiddenBias)[i][0] = 1.0f;
}

NeuralNetwork::~NeuralNetwork()
{
    delete m_inputWeights;
    delete m_hiddenWeights;

    delete m_inputBias;
    delete m_hiddenBias;
}

float* NeuralNetwork::feedForward(Matrix& input)
{
    // get outputs of hidden nodes
    auto hiddenLayer = m_inputWeights->product(input);
    hiddenLayer += *m_inputBias;
    // apply activation function
    hiddenLayer.map(&sigmoid);

    // get outputs for output nodes
    auto outputLayer = m_hiddenWeights->product(hiddenLayer);
    outputLayer += *m_hiddenBias;
    // apply activation function
    outputLayer.map(&sigmoid);

    // let's turn this matrix into a float array
    auto* result = new float[m_outputNodes];
    for(int i = 0; i < m_outputNodes; ++i)
        result[i] = outputLayer[i][0];

    return result;
}

float* NeuralNetwork::feedForward(float* input)
{
    // make a matrix from this array
    Matrix m(m_inputNodes, 1);

    for(int i = 0; i < m_inputNodes; ++i)
        m[i][0] = input[i];

    return feedForward(m);
}

float sigmoid(float x)
{
    float ex = expf(x);
    return ex / (ex + 1);
}
