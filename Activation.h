// Activation.h
#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "Matrix.h"
#include <cmath>

typedef Matrix (*activation_func) (const Matrix& mat);

// Insert Activation namespace here...
namespace activation {
    extern activation_func relu;
    extern activation_func softmax;
}

#endif //ACTIVATION_H
