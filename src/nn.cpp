#include "nn.hpp"

#include <iostream>
#include <cmath>
#include <fstream>

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
    //auto lastLayer = new Matrix(m_inputNodes, 1);
    Matrix lastLayer = Matrix(m_inputNodes, 1);
    for(int i = 0; i < m_inputNodes; ++i)
        lastLayer[i][0] = input[i];

    for(int i = 0; i < m_hiddenLayers+1; ++i)
    {
        // take the input to these neurons and multiply them by the weights
        auto layer = m_weights[i]->product(lastLayer);
        // add biases separately, could also just be another weight
        layer += *(m_biases[i]);
        // scale between 0 and 1 using activation function
        layer.map(&activtan);

        lastLayer = layer;
    }
    // now lastLayer is the matrix representing the output

    // turn output into float array
    for(int i = 0; i < m_outputNodes; ++i)
        output[i] = lastLayer[i][0];
}

void NeuralNetwork::propagate(float const* inputs, float const* targets)
{
    // turn the inputs and targets into 1 column matrices
    Matrix inputMatrix(m_inputNodes, 1);
    Matrix targetMatrix(m_outputNodes, 1);

    for(int i = 0; i < m_inputNodes; ++i)
        inputMatrix[i][0] = inputs[i];
    for(int i = 0; i < m_outputNodes; ++i)
        targetMatrix[i][0] = targets[i];

    // get the results of each layer
    // just feedforward (like the guess function) but keep track of layers
    // I should probably put this feedforward stuff into a function and
    //  use that for both this and guessing
    Matrix* allLayers = new Matrix[m_hiddenLayers+1];
    Matrix lastLayer = inputMatrix;
    for(int i = 0; i < m_hiddenLayers+1; ++i)
    {
        auto layer = m_weights[i]->product(lastLayer);
        layer += *(m_biases[i]);
        layer.map(&activtan);

        lastLayer = layer;
        allLayers[i] = lastLayer;
    }

    // actual backpropagation part
    Matrix error = targetMatrix - allLayers[m_hiddenLayers];
    for(int i = m_hiddenLayers; i >= 0; --i)
    {
        // get the gradient - the derivative of the results of this layer
        Matrix gradient = allLayers[i];
        gradient.map(&derivtan);
        // adjust based on the difference between the target and the result
        gradient *= error;
        // and adjust for our learning rate
        gradient *= m_learningRate;

        // adjust bias with this value before calculating the weight delta
        (*m_biases[i]) += gradient;

        Matrix pLayer; // previous layer
        if(i == 0)
            pLayer = inputMatrix;
        else
            pLayer = allLayers[i-1];
        // transpose to get it in the correct layout to multiply
        Matrix pTrans = pLayer.transposed();
        // multiply the last layer's results by the gradient to get the
        //  amount we should adjust the weights by
        Matrix wDelta = gradient.product(pTrans);

        // adjust the weights!
        (*m_weights[i]) += wDelta;

        Matrix weightTrans = m_weights[i]->transposed();
        Matrix m = error; // save this here to delete it after reassigning
        // base the next layer's error on this layer's error
        error = weightTrans.product(error);
    }

	delete[] allLayers;
}

bool NeuralNetwork::save(const char* filename)
{
    /*
     * Format of file:
     *
     * 8 bytes - identifying the file
     *
     * 4 bytes - input nodes
     * 4 bytes - output nodes
     *
     * 4 bytes - learning rate
     *
     * 4 bytes - number of hidden layers
     * n * 4 bytes - number of neurons in each hidden layer
     *
     * 4 bytes - number of matrices (hidden layers+1)
     * however many bytes - all weights
     * however many bytes - all biases
     */
    std::fstream file;
    file.open(filename, std::ios::out | std::ios::binary);

    if(!file.is_open())
        return false;

    // stick the ID values into a char array
    char fileId[] = NN_FILE_ID;

    // write file ID
    file.write(fileId, NN_FILE_ID_SIZE);

    // write input/output node numbers
    file.write((char*)&m_inputNodes, 4);
    file.write((char*)&m_outputNodes, 4);

    // write learning rate
    file.write((char*)&m_learningRate, 4);

    // write number of hidden layers
    file.write((char*)&m_hiddenLayers, 4);
    // write number of hidden layer neurons
    for(int i = 0; i < m_hiddenLayers; ++i)
        file.write((char*)&m_hiddenNodeCount[i], 4);

    // now we can start writing matrix data

    // write the number of matrices, will always be hiddenLayers+1
    int matrixCount = m_hiddenLayers+1;
    file.write((char*)&matrixCount, 4);

    // write weight matrices
    for(int i = 0; i < matrixCount; ++i)
    {
        Matrix* m = m_weights[i];

        int rows = m->getRows();
        int cols = m->getColumns();

        // write the actual values
        for(int y = 0; y < rows; ++y)
        {
            for(int x = 0; x < cols; ++x)
            {
                float val = (*m)[y][x];

                file.write((char*)&val, 4);
            }
        }
    }

    // write bias matrices
    for(int i = 0; i < matrixCount; ++i)
    {
        Matrix* m = m_biases[i];

        int rows = m->getRows();
        int cols = m->getColumns();

        // write the actual values
        for(int y = 0; y < rows; ++y)
        {
            for(int x = 0; x < cols; ++x)
            {
                float val = (*m)[y][x];

                file.write((char*)&val, 4);
            }
        }
    }

    return true;
}

NeuralNetwork* NeuralNetwork::load(const char* filename)
{
    std::fstream file;
    file.open(filename, std::ios::in | std::ios::binary);

    if(!file.is_open())
        return nullptr;

    char fileId[NN_FILE_ID_SIZE+1];
    char expectedId[] = NN_FILE_ID;

    // check 'header' to make sure we're opening a valid file
    file.read(fileId, NN_FILE_ID_SIZE);
    for(int i = 0; i < NN_FILE_ID_SIZE; ++i)
        if(fileId[i] != expectedId[i])
            return nullptr;

    int input;
    int output;
    int hLayers;
    float learningRate;

    // get input nodes
    file.read((char*)&input, 4);
    // and output nodes
    file.read((char*)&output, 4);

    // get learning rate
    file.read((char*)&learningRate, 4);

    // get number of hidden layers
    file.read((char*)&hLayers, 4);

    // get the node count for each hidden layer
    auto hNodes = new int[hLayers];
    for(int i = 0; i < hLayers; ++i)
    {
        int read;
        file.read((char*)&read, 4);
        hNodes[i] = read;
    }

    // we can make our NN object now
    auto result = new NeuralNetwork(input, hLayers, hNodes, output);
    result->setLearningRate(learningRate);

    // and delete that dynamically allocated array
    delete[] hNodes;

    // get number of matrices
    int matrixCount;
    file.read((char*)&matrixCount, 4);

    // read weight matrices
    for(int i = 0; i < matrixCount; ++i)
    {
        // why can I access this private member???
        Matrix* m = result->m_weights[i];

        int rows = m->getRows();
        int cols = m->getColumns();

        for(int y = 0; y < rows; ++y)
        {
            for(int x = 0; x < cols; ++x)
            {
                float val;
                file.read((char*)&val, 4);

                printf("%.2f, ", val);

                (*m)[y][x] = val;
            }
            printf("\n");
        }
        printf("\n");
    }

    // read bias matrices
    for(int i = 0; i < matrixCount; ++i)
    {
        Matrix* m = result->m_biases[i];

        int rows = m->getRows();
        int cols = m->getColumns();

        for(int y = 0; y < rows; ++y)
        {
            for(int x = 0; x < cols; ++x)
            {
                float val;
                file.read((char*)&val, 4);

                (*m)[y][x] = val;
            }
        }
    }

    return result;
}

NeuralNetwork* NeuralNetwork::copy()
{
    auto result = new NeuralNetwork(m_inputNodes, m_hiddenLayers,
                                    m_hiddenNodeCount, m_outputNodes);

    result->setLearningRate(this->getLearningRate());

    // copy matrices
    for(int i = 0; i < m_hiddenLayers+1; ++i)
    {
        delete result->m_weights[i];
        result->m_weights[i] = new Matrix(*m_weights[i]);
        delete result->m_biases[i];
        result->m_biases[i] = new Matrix(*m_weights[i]);
    }

    return result;
}

void NeuralNetwork::mutate(float rate)
{
    for(int i = 0; i < m_hiddenLayers+1; ++i)
    {
        m_weights[i]->mutate(rate);
        m_biases[i]->mutate(rate);
    }
}

void NeuralNetwork::breed(NeuralNetwork* other)
{
    for(int i = 0; i < m_hiddenLayers+1; ++i)
    {
    }
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

float activtan(float x)
{
    return tanhf(x);
}

float derivtan(float x)
{
    return 1 - (x*x);
}
