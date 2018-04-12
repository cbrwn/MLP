#include "nn.hpp"

#include <cmath>

#include "matrix.hpp"

NeuralNetwork::NeuralNetwork(int in, int hid, int out)
    : m_inputNodes(in), m_hiddenNodes(hid), m_outputNodes(out)
{
    m_inputWeights = new Matrix(hid, in);
    m_hiddenWeights = new Matrix(out, hid);

    // start with random weights
    m_inputWeights->randomize();
    m_hiddenWeights->randomize();

    m_inputBias = new Matrix(hid, 1);
    m_hiddenBias = new Matrix(out, 1);

    // set the biases to 1
    for(int i = 0; i < hid; ++i)
        (*m_inputBias)[i][0] = 1.0f;
    for(int i = 0; i < out; ++i)
        (*m_hiddenBias)[i][0] = 1.0f;

    m_learningRate = 0.1f;
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

float* NeuralNetwork::guess(float const* input)
{
    // make a matrix from this array
    Matrix m(m_inputNodes, 1);

    for(int i = 0; i < m_inputNodes; ++i)
        m[i][0] = input[i];

    return feedForward(m);
}

void NeuralNetwork::teach(float const* inputs, float const* targets)
{
    // turn targets into a matrix
    Matrix targetMatrix(m_outputNodes, 1);
    for(int i = 0; i < m_outputNodes; ++i)
        targetMatrix[i][0] = targets[i];

    // turn input into matrix
    Matrix inputMatrix(m_inputNodes, 1);
    for(int i = 0; i < m_inputNodes; ++i)
        inputMatrix[i][0] = inputs[i];

    // grab the current output of the input
    // not just calling guess because we want to keep track of
    // the matrices along the way
    Matrix hiddenLayer = m_inputWeights->product(inputMatrix);
    hiddenLayer += *m_inputBias;
    // apply activation function
    hiddenLayer.map(&sigmoid);

    // get outputs for output nodes
    Matrix outputLayer = m_hiddenWeights->product(hiddenLayer);
    outputLayer += *m_hiddenBias;
    // apply activation function
    outputLayer.map(&sigmoid);

    // start back propagation

    // get the difference between these
    Matrix outputError = targetMatrix - outputLayer;

    // get the gradient for use in delta calculations
    Matrix outputGradient = outputLayer;
    outputGradient.map(&isigmoid);
    outputGradient *= outputError;
    outputGradient *= m_learningRate;

    // update bias before we change gradient any more
    (*m_hiddenBias) += outputGradient;

    // get the deltas to adjust the weights
    Matrix hiddenLayerT = hiddenLayer.transposed();
    Matrix hiddenDelta = outputGradient.product(hiddenLayerT);
    (*m_hiddenWeights) += hiddenDelta;

    // grab hidden layer error
    Matrix hiddenTrans = m_hiddenWeights->transposed();
    Matrix hiddenError = hiddenTrans.product(outputError);

    // get hidden gradient
    Matrix hiddenGradient = hiddenLayer;
    hiddenGradient.map(&isigmoid);
    hiddenGradient *= hiddenError;
    hiddenGradient *= m_learningRate;

    // adjust bias
    (*m_inputBias) += hiddenGradient;

    // get the deltas for these weights
    Matrix inputLayerT = inputMatrix.transposed();
    Matrix inputDelta = hiddenGradient.product(inputLayerT);
    (*m_inputWeights) += inputDelta;
}

float sigmoid(float x)
{
    float ex = expf(x);
    return ex / (ex + 1);
}

// inverse of sigmoid
// but we're assuming n has already been mapped using sigmoid
float isigmoid(float n)
{
    return n * (1 - n);
}
