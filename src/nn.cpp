#include "nn.hpp"

#include <cmath>

#include "matrix.hpp"

NeuralNetwork::NeuralNetwork(int in, int hid, int* nodes, int out)
{
    m_inputNodes = in;
    m_outputNodes = out;

    m_hiddenLayers = hid;
    m_hiddenNodeCount = new int[hid];

    for(int i = 0; i < hid; ++i)
        m_hiddenNodeCount[i] = nodes[i];

    m_learningRate = 0.1f;

    // the number of matrices for weights and biases
    const int matrixCount = hid+1;

    m_weights = new Matrix*[matrixCount];
    m_biases = new Matrix*[matrixCount];

    // make matrices based on the number of nodes in each layer
    int lastNodes = in;
    for(int i = 0; i < hid; ++i)
    {
        int numNodes = nodes[i];
        m_weights[i] = new Matrix(numNodes, lastNodes);
        m_biases[i] = new Matrix(numNodes, 1);
        lastNodes = numNodes;
    }

    // make last weight and bias matrices
    m_weights[matrixCount-1] = new Matrix(out, lastNodes);
    m_biases[matrixCount-1] = new Matrix(out, 1);

    // randomly set weights and biases
    for(int i = 0; i < matrixCount; ++i)
    {
        m_weights[i]->randomize();
        m_biases[i]->randomize();
    }
}

NeuralNetwork::~NeuralNetwork()
{
    for(int i = 0; i < m_hiddenLayers+1; ++i)
    {
        delete m_weights[i];
        delete m_biases[i];
    }
    delete[] m_weights;
    delete[] m_biases;

    delete[] m_hiddenNodeCount;
}

void NeuralNetwork::guess(float const* input, float* output)
{
    // make a matrix from the input
    auto lastLayer = new Matrix(m_inputNodes, 1);
    for(int i = 0; i < m_inputNodes; ++i)
        (*lastLayer)[i][0] = input[i];

    for(int i = 0; i < m_hiddenLayers+1; ++i)
    {
        auto layer = m_weights[i]->product(*lastLayer);
        layer += *(m_biases[i]);
        layer.map(&sigmoid);

        delete lastLayer;
        lastLayer = new Matrix(layer);
    }

    // turn output into float array
    for(int i = 0; i < m_outputNodes; ++i)
        output[i] = (*lastLayer)[i][0];

    delete lastLayer;
}

void NeuralNetwork::propagate(float const* inputs, float const* targets)
{
    // turn the inputs and targets into 1 column matrices
    auto inputMatrix = new Matrix(m_inputNodes, 1);
    auto targetMatrix = new Matrix(m_outputNodes, 1);

    for(int i = 0; i < m_inputNodes; ++i)
        (*inputMatrix)[i][0] = inputs[i];
    for(int i = 0; i < m_outputNodes; ++i)
        (*targetMatrix)[i][0] = targets[i];

    // get the results of each layer
    auto allLayers = new Matrix*[m_hiddenLayers+1];
    auto lastLayer = new Matrix(*inputMatrix);
    for(int i = 0; i < m_hiddenLayers+1; ++i)
    {
        auto layer = m_weights[i]->product(*lastLayer);
        layer += *(m_biases[i]);
        layer.map(&sigmoid);

        if(i == 0)
            delete lastLayer;
        lastLayer = new Matrix(layer);
        allLayers[i] = lastLayer;
    }

    // work backwards
    auto error = new Matrix(*targetMatrix - *(allLayers[m_hiddenLayers]));
    for(int i = m_hiddenLayers; i >= 0; --i)
    {
        auto gradient = new Matrix(*allLayers[i]);
        gradient->map(&isigmoid);
        (*gradient) *= *error;
        (*gradient) *= m_learningRate;

        (*m_biases[i]) += (*gradient);

        Matrix* pLayer;
        if(i == 0)
            pLayer = inputMatrix;
        else
            pLayer = allLayers[i-1];
        Matrix pTrans = pLayer->transposed();
        Matrix wDelta = gradient->product(pTrans);

        (*m_weights[i]) += wDelta;
        Matrix weightTrans = m_weights[i]->transposed();

        Matrix* m = error;
        error = new Matrix(weightTrans.product(*error));
        delete m;
        delete gradient;
    }
    delete error;

    // delete everything
    for(int i = 0; i < m_hiddenLayers+1; ++i)
    {
        delete allLayers[i];
    }
    delete[] allLayers;
    delete targetMatrix;
    delete inputMatrix;
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
