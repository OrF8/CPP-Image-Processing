#include "Matrix.h"

/*
 * Constructs matrix of size rows x cols.
 * Inits all values to 0.
 * @param rows Number of rows in the matrix.
 * @param cols Number of columns in the matrix.
 * @return Matrix object.
 */
Matrix::Matrix(int rows, int cols)
{
    if (rows <= 0 || cols <= 0)
    {
        throw::std::invalid_argument(DIM_ERROR);
    }
    _dims = matrix_dims();
    _dims.rows = rows;
    _dims.cols = cols;
    _data = new float[rows * cols];
    for (int i = 0; i < rows * cols; i++)
    {
        // Initializes all entries to 0.
        _data[i] = 0;
    }
}

/*
 * Default constructor.
 * Constructs matrix of size 1 x 1.
 * Inits all values to 0.
 * @return Matrix object.
 */
Matrix::Matrix() : Matrix(1, 1)
{
    // Uses constructor delegation.
}

/*
 * Copy constructor.
 * Constructs a matrix from another matrix m.
 * @param m Matrix to copy.
 * @return Matrix object.
 */
Matrix::Matrix(const Matrix& m) :
    _dims(matrix_dims())
{
    _dims.rows = m._dims.rows;
    _dims.cols = m._dims.cols;
    _data = new float[_dims.rows * _dims.cols];
    for (int i = 0; i < _dims.rows * _dims.cols; i++)
    {
        _data[i] = m[i];
    }
}

/*
 * Destructor.
 * Frees memory allocated for the matrix.
 */
Matrix::~Matrix()
{
    delete[] _data;
}

/*
 * Returns the number of rows in the matrix.
 * @return Number of rows in the matrix.
 */
int Matrix::get_rows() const
{
    return _dims.rows;
}

/*
 * Returns the number of columns in the matrix.
 * @return Number of columns in the matrix.
 */
int Matrix::get_cols() const
{
    return _dims.cols;
}

/*
 * Transposes the matrix.
 * @return Transposed matrix.
 */
Matrix& Matrix::transpose()
{
    // Alternate rows and cols
    Matrix transposed(_dims.cols, _dims.rows);
    for (int i = 0; i < _dims.rows; i++)
    {
        for (int j = 0; j < _dims.cols; j++)
        {
            // (A^t)_(i, j) = A_(j, i)
            transposed(j, i) = (*this)(i, j);
        }
    }
    *this = transposed;
    return *this;
}

/*
 * Vectorizes the matrix.
 * @return Vectorized matrix.
 */
Matrix& Matrix::vectorize()
{
    Matrix vectorized(_dims.rows * _dims.cols, 1);
    for (int i = 0; i < _dims.rows; i++)
    {
        for (int j = 0; j < _dims.cols; j++)
        {
            vectorized(i * _dims.cols + j, 0) = (*this)(i, j);
        }
    }
    *this = vectorized;
    return *this;
}

/*
 * Prints the matrix in plain format -
 * matrix elements with spaces between them, and newlines between rows.
 */
void Matrix::plain_print() const
{
    for (int i = 0; i < _dims.rows; i++)
    {
        for (int j = 0; j < _dims.cols; j++)
        {
            std::cout << (*this)(i, j) << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

/*
 * Computes the dot product (hadamard product) of two matrices.
 * @param rhs Matrix to multiply with.
 * @return Dot product of the two matrices.
 */
Matrix Matrix::dot(const Matrix& rhs) const
{
    if (_dims.cols != rhs._dims.cols || _dims.rows != rhs._dims.rows)
    {
        throw::std::invalid_argument(DIM_ERROR);
    }
    int rows = _dims.rows;
    int cols = _dims.cols;
    Matrix result(rows, cols);
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            result(i, j) = (*this)(i, j) * rhs(i, j);
        }
    }
    return result;
}

/*
 * Computes the Frobenius norm of the matrix.
 * @return Frobenius norm of the matrix.
 */
float Matrix::norm() const
{
    float norm = 0.0;
    for (int i = 0; i < _dims.rows; i++)
    {
        for (int j = 0; j < _dims.cols; j++)
        {
            norm += (*this)(i, j) * (*this)(i, j);
        }
    }
    return std::sqrt(norm);
}

/*
 * Computes the reduced row echelon form of the matrix.
 * @return Reduced row echelon form of the matrix.
 */
Matrix Matrix::rref() const {
    Matrix result(*this);
    int lead = 0;
    for (int r = 0; r < result._dims.rows; ++r, ++lead)
    {
        if (lead >= result._dims.cols)
        {
            break;
        }
        int i = r;
        while (result._data[i * result._dims.cols + lead] == 0)
        {
            ++i;
            if (i == result._dims.rows)
            {
                i = r;
                ++lead;
                if (lead == result._dims.cols)
                {
                    return result;
                }
            }
        }
        for (int k = 0; k < result._dims.cols; ++k) // Swap rows i and r
        {
            std::swap(
                result._data[r * result._dims.cols + k],
                result._data[i * result._dims.cols + k]
                );
        }
        float pivot = result._data[r * result._dims.cols + lead];
        for (int k = 0; k < result._dims.cols; ++k) // Normalize row r
        {
            result._data[r * result._dims.cols + k] /= pivot;
        }
        for (int j = 0; j < result._dims.rows; j++)
        {
            if (j != r) // Eliminate all other elements in column lead
            {
                float lv = result._data[j * result._dims.cols + lead];
                for (int k = 0; k < result._dims.cols; ++k)
                {
                    result._data[j * result._dims.cols + k] -=
                        lv * result._data[r * result._dims.cols + k];
                }
            }
        }
    }
    return result;
}

/*
 * Computes the index of the maximum element in the matrix.
 * @return Index of the maximum element in the matrix.
 */
int Matrix::argmax() const
{
    int argmax = 0;
    for (int i = 0; i < _dims.rows; i++)
    {
        for (int j = 0; j < _dims.cols; j++)
        {
            if ((*this)(i, j) > (*this)[argmax])
            {
                argmax = i * _dims.cols + j;
            }
        }
    }
    return argmax;
}

/*
 * Computes the sum of all elements in the matrix.
 * @return Sum of all elements in the matrix.
 */
float Matrix::sum() const
{
    float sum = 0.0;
    for (int i = 0; i < _dims.rows; i++)
    {
        for (int j = 0; j < _dims.cols; j++)
        {
            sum += (*this)(i, j);
        }
    }
    return sum;
}

/*
 * Adds the matrix rhs to the current matrix.
 * @param rhs Matrix to add.
 * @return Reference to the current matrix.
 */
Matrix& Matrix::operator+= (const Matrix& rhs)
{
    if (_dims.rows != rhs._dims.rows || _dims.cols != rhs._dims.cols)
    {
        throw::std::invalid_argument(DIM_ERROR);
    }
    for (int i = 0; i < _dims.rows; i++)
    {
        for (int j = 0; j < _dims.cols; j++)
        {
            (*this)(i, j) += rhs(i, j);
        }
    }
    return *this;
}

/*
 * Adds the matrix rhs to the current matrix.
 * @param rhs Matrix to add.
 * @return Sum of the two matrices.
 */
Matrix Matrix::operator+ (const Matrix& rhs)
{
    if (_dims.rows != rhs._dims.rows || _dims.cols != rhs._dims.cols)
    {
        throw::std::invalid_argument(DIM_ERROR);
    }
    Matrix sum(*this);
    sum += rhs;
    return sum;
}

void Matrix::swap(Matrix& other) noexcept
{
    std::swap(_dims, other._dims);
    std::swap(_data, other._data);
}

/*
 * Assigns the matrix rhs to the current matrix.
 * @param rhs Matrix to assign.
 * @return Reference to the current matrix.
 */
Matrix& Matrix::operator= (const Matrix& rhs)
{
    Matrix temp(rhs);
    this->swap (temp);
    return *this;
}

/*
 * Computes the product of the matrix with another matrix rhs.
 * Works from the right and from the left.
 * @param rhs Matrix to multiply with.
 * @return Product of the two matrices.
 */
Matrix Matrix::operator* (const Matrix& rhs) const
{
    if (_dims.cols != rhs._dims.rows)
    {
        throw::std::invalid_argument(DIM_ERROR);
    }
    Matrix product(_dims.rows, rhs._dims.cols);
    for (int i = 0; i < _dims.rows; i++)
    {
        for (int j = 0; j < rhs._dims.cols; j++)
        {
            for (int k = 0; k < _dims.cols; k++)
            {
                product(i, j) += (*this)(i, k) * rhs(k, j);
            }
        }
    }
    return product;
}

/*
 * Computes the product of the matrix with a scalar from the right.
 * @param scalar Scalar to multiply with.
 * @return Product of the matrix with the scalar.
 */
Matrix Matrix::operator* (const float scalar) const
{
    Matrix product(*this);
    for (int i = 0; i < _dims.rows; i++)
    {
        for (int j = 0; j < _dims.cols; j++)
        {
            product(i, j) *= scalar;
        }
    }
    return product;
}

/*
 * Computes the product of the matrix with a scalar from the left.
 * @param scalar Scalar to multiply with.
 * @param rhs Matrix to multiply with.
 * @return Product of the matrix with the scalar.
 */
Matrix operator*(const float scalar, const Matrix& rhs)
{
    Matrix product(rhs.get_rows(), rhs.get_cols());
    for (int i = 0; i < rhs.get_rows(); i++)
    {
        for (int j = 0; j < rhs.get_cols(); j++)
        {
            product(i, j) = scalar * rhs(i, j);
        }
    }
    return product;
}

/*
 * Read access operator for the matrix.
 * @param row Row index.
 * @param col Column index.
 * @return Element at the specified row and column.
 */
float Matrix::operator() (int row, int col) const
{
    if (_dims.rows <= row || _dims.cols <= col)
    {
        throw::std::invalid_argument(INDEX_ERROR);
    }
    return _data[row * _dims.cols + col];
}

/*
 * Write access operator for the matrix.
 * @param row Row index.
 * @param col Column index.
 * @return Reference to the element at the specified row and column.
 */
float& Matrix::operator()(int row, int col)
{
    if (_dims.rows <= row || row < 0 || _dims.cols <= col || col < 0)
    {
        throw::std::invalid_argument(INDEX_ERROR);
    }
    return _data[row * _dims.cols + col];
}

/*
 * Read access operator for the matrix.
 * @param index Index of the element.
 * @return Element at the specified index.
 */
float Matrix::operator[] (int index) const
{
    return _data[index];
}

/*
 * Write access operator for the matrix.
 * @param index Index of te element.
 * @return Reference to the element at the specified index.
 */
float& Matrix::operator[] (int index)
{
    return _data[index];
}

/*
 * Output stream operator for the matrix.
 * @param os Output stream.
 * @param m Matrix to output.
 * @return Reference to the output stream.
 */
std::ostream& operator<< (std::ostream& os, const Matrix& m)
{
    for (int i = 0; i < m._dims.rows; i++)
    {
        for (int j = 0; j < m._dims.cols; j++)
        {
            if (m(i, j) > PPRINT_VALUE)
            {
                os << "**";
            }
            else
            {
                os << "  ";
            }
        }
        os << std::endl;
    }
    return os;
}

/*
 * Input stream operator for the matrix.
 * @param is Input stream.
 * @param m Matrix to input.
 * @return Reference to the input stream.
 */
std::istream& operator>> (std::istream& is, Matrix& m)
{
    is.seekg(0, std::ios::end);
    long length = is.tellg();
    long m_dims = (long)(m._dims.cols * m._dims.rows * sizeof(float));
    if (length < m_dims)
    {
        throw::std::invalid_argument(DIM_ERROR);
    }
    is.seekg(0, std::ios::beg);
    char *buffer = new char[m_dims];
    is.read(buffer, (long)m_dims);
    if (!is.good())
    {
        delete[] buffer;
        throw::std::invalid_argument("Error opening file.");
    }
    else
    {
        std::memcpy(m._data, buffer, m_dims);
        delete[] buffer;
    }
    return is;
}
