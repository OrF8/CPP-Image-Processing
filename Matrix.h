// Matrix.h
#ifndef MATRIX_H
#define MATRIX_H

#include <iostream>
#include <cmath>
#include <cstring>

#define PPRINT_VALUE 0.1
#define DIM_ERROR "Invalid matrix dimension."
#define INDEX_ERROR "Index out of bounds."

// You don't have to use the struct. Can help you with MlpNetwork.h
struct matrix_dims {
    int rows;
    int cols;
};


// Insert Matrix class here...

class Matrix {
    public:
        Matrix(int rows, int cols);
        Matrix();
        Matrix(const Matrix& m);
        ~Matrix();
        int get_rows() const;
        int get_cols() const;
        Matrix& transpose();
        Matrix& vectorize();
        void plain_print() const;
        Matrix dot(const Matrix& rhs) const;
        float norm() const;
        Matrix rref() const;
        int argmax() const;
        float sum() const;
        Matrix& operator+= (const Matrix& rhs);
        Matrix operator+ (const Matrix& rhs);
        Matrix& operator= (const Matrix& rhs);
        Matrix operator* (const Matrix& rhs) const;
        Matrix operator* (float scalar) const;
        float operator() (int row, int col) const;
        float& operator()(int row, int col);
        float operator[] (int index) const;
        float& operator[] (int index);
        friend std::istream& operator>> (std::istream& is, Matrix& m);
        friend std::ostream& operator<< (std::ostream& os, const Matrix& m);

    private:
        matrix_dims _dims{};
        float *_data;
        void swap(Matrix& other) noexcept;
};

Matrix operator*(float scalar, const Matrix& rhs);

#endif //MATRIX_H