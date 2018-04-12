#pragma once

class NeuralNetwork {
public:
    NeuralNetwork(int in, int hid, int out);

private:
    int m_inputNodes;
    int m_hiddenNodes;
    int m_outputNodes;
};
