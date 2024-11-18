#include "Activation.h"
#include <cfloat>

/*
 * ReLu function implementation.
 * @param mat matrix to calculate ReLu on.
 * @return Matrix with ReLu entries.
 */
Matrix relu_impl(const Matrix &mat)
{
    int rows = mat.get_rows();
    int cols = mat.get_cols();
    Matrix res(rows, cols);
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            // Calculate ReLu according to the given formula
            float x = mat(i, j);
            x >= 0 ? res(i, j) = x : res(i, j) = 0;
        }
    }
    return res;
}

/*
 * softmax function for a row implementation.
 * @param mat matrix to calculate softmax on.
 * @return Matrix with softmax entries.
 */
Matrix softmax_row(const Matrix& mat)
{
    int rows = mat.get_rows();
    int cols = mat.get_cols();
    Matrix res(rows, cols);
    for (int i = 0; i < rows; i++)
    {
        // Find the max value for numerical stability
        float max_val = -FLT_MAX;
        for (int j = 0; j < cols; j++)
        {
            if (mat(i, j) > max_val)
            {
                max_val = mat(i, j);
            }
        }
        // Calculate the sum of exponentials
        float exp_sum = 0.0;
        for (int j = 0; j < cols; j++)
        {
            exp_sum += std::exp(mat(i, j) - max_val);
        }
        // Calculate softmax probabilities
        for (int j = 0; j < cols; j++)
        {
            res(i, j) = std::exp(mat(i, j) - max_val) / exp_sum;
        }
    }

    return res;
}

/*
 * softmax function implementation.
 * @param mat matrix to calculate softmax on.
 * @return Matrix with softmax entries.
 */
Matrix softmax_impl(const Matrix &mat)
{
    int rows = mat.get_rows();
    int cols = mat.get_cols();
    if (rows == 1)
    {
        return softmax_row(mat);
    }
    Matrix res(rows, cols);
    for (int i = 0; i < cols; i++) // Iterate over columns
    {
        float exp_coordinate_sum = 0.0;
        for (int j = 0; j < rows; j++) // Sum exponentials in a column
        {
            float x = mat(j, i);
            exp_coordinate_sum += std::exp(x);
        }
        for (int j = 0; j < rows; j++) // Apply softmax formula per column
        {
            float x = mat(j, i);
            res(j, i) = std::exp(x) / exp_coordinate_sum;
        }
    }
    return res;
}

namespace activation {
    activation_func relu = relu_impl;
    activation_func softmax = softmax_impl;
}
