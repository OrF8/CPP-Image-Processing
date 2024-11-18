// Dense.h
#ifndef DENSE_H
#define DENSE_H

#include "Activation.h"

// Insert Dense class here...
class Dense {
    public:
        Dense(Matrix& weights, Matrix& bias, activation_func);
        Matrix get_weights() const;
        Matrix get_bias() const;
        activation_func get_activation() const;
        Matrix operator()(const Matrix &input);

    private:
        Matrix _weights;
        Matrix _bias;
        activation_func _act_func;
};








#endif //DENSE_H