#include "nn.hpp"

#include <cmath>

#include "matrix.hpp"

NeuralNetwork::NeuralNetwork(int in, int hid, int const* nodes, int out)
{
    m_inputNodes = in;
    m_outputNodes = out;

    m_hiddenLayers = hid;

    // copy values from the nodes array
    m_hiddenNodeCount = new int[hid];
    for(int i = 0; i < hid; ++i)
        m_hiddenNodeCount[i] = nodes[i];

    // default learning rate, same as Daniel Shiffman's
    // (i don't know what a good learning rate is)
    m_learningRate = 0.1f;

    // the number of matrices for weights and biases
    // hidden layers + 1 for the output layer
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
        // take the input to these neurons and multiply them by the weights
        auto layer = m_weights[i]->product(*lastLayer);
        // add biases separately, could also just be another weight
        layer += *(m_biases[i]);
        // scale between 0 and 1 using activation function
        layer.map(&sigmoid);

        delete lastLayer;
        lastLayer = new Matrix(layer);
    }
    // now lastLayer is the matrix representing the output

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
    // just feedforward (like the guess function) but keep track of layers
    // I should probably put this feedforward stuff into a function and
    //  use that for both this and guessing
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

    // actual backpropagation part
    auto error = new Matrix(*targetMatrix - *(allLayers[m_hiddenLayers]));
    for(int i = m_hiddenLayers; i >= 0; --i)
    {
        // get the gradient - the derivative of the results of this layer
        auto gradient = new Matrix(*allLayers[i]);
        gradient->map(&isigmoid);
        // adjust based on the difference between the target and the result
        (*gradient) *= *error;
        // and adjust for our learning rate
        (*gradient) *= m_learningRate;

        // adjust bias with this value before calculating the weight delta
        (*m_biases[i]) += (*gradient);

        Matrix* pLayer; // previous layer
        if(i == 0)
            pLayer = inputMatrix;
        else
            pLayer = allLayers[i-1];
        // transpose to get it in the correct layout to multiply
        Matrix pTrans = pLayer->transposed();
        // multiply the last layer's results by the gradient to get the
        //  amount we should adjust the weights by
        Matrix wDelta = gradient->product(pTrans);

        // adjust the weights!
        (*m_weights[i]) += wDelta;

        Matrix weightTrans = m_weights[i]->transposed();
        Matrix* m = error; // save this here to delete it after reassigning
        // base the next layer's error on this layer's error
        error = new Matrix(weightTrans.product(*error));
        delete m;
        delete gradient;
    }
    delete error;

    // delete everything
    for(int i = 0; i < m_hiddenLayers+1; ++i)
        delete allLayers[i];
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
