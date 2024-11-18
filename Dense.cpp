#include "Dense.h"

/*
 * Dense constructor.
 * @param weights - the weight matrix of the layer.
 * @param bias - the bias matrix of the layer.
 * @param act_func - the activation function of the layer.
 * @return a new Dense object.
 */
Dense::Dense(Matrix& weights, Matrix& bias, activation_func act_func) :
    _weights(weights), _bias(bias), _act_func(act_func)
    {}

/*
 * getter function.
 * @return the weight matrix of the layer.
 */
Matrix Dense::get_weights () const
{
    return _weights;
}

/*
 * getter function.
 * @return the bias matrix of the layer.
 */
Matrix Dense::get_bias () const
{
    return _bias;
}

/*
 * getter function.
 * @return the activation function of the layer.
 */
activation_func Dense::get_activation () const
{
    return _act_func;
}

/*
 * operator() overload.
 * @param input - the input matrix to the layer.
 * @return the output matrix of the layer.
 */
Matrix Dense::operator()(const Matrix &input)
{
    return Matrix(_act_func(_weights * input + _bias));
}
