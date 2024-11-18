#include "MlpNetwork.h"

/*
 * Constructor for MlpNetwork.
 * @param weights - array of matrices representing the weights of the network.
 * @param biases - array of matrices representing the biases of the network.
 * @return a new MlpNetwork object.
 */
MlpNetwork::MlpNetwork(Matrix *weights, Matrix *biases)
{
    for (int i = 0; i < MLP_SIZE - 1; i++)
    {
        _layers[i] = new Dense(weights[i], biases[i], activation::relu);
    }
    _layers[MLP_SIZE - 1] = new Dense(
        weights[MLP_SIZE - 1], biases[MLP_SIZE - 1], activation::softmax
        );
}

/*
 * Destructor for MlpNetwork.
 */
MlpNetwork::~MlpNetwork()
{
    for (int i = 0; i < MLP_SIZE; i++)
    {
        delete _layers[i]; // Free the allocated memory for each Dense object
    }
}

/*
 * Overloaded operator() for MlpNetwork.
 * @param input - a matrix representing the input to the network.
 * @return a digit object representing the network's prediction.
 */
digit MlpNetwork::operator() (const Matrix& input) const
{
    Matrix returns(input);
    // Feed the input through the different layers in the network
    for (int i = 0; i < MLP_SIZE; i++)
    {
        returns = (*_layers[i])(returns);
    }
    digit res;
    res.probability = 0;
    // Find the highest probability in the output layer
    for (int i = 0; i < returns.get_rows(); i++)
    {
        if (returns(i, 0) > res.probability)
        {
            res.probability = returns(i, 0);
            res.value = i;
        }
    }
    return res;
}