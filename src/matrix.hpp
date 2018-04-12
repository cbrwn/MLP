#pragma once

class Matrix
{
public:
    /***
     * @brief Makes a new matrix of size rows*cols
     * @param rows Number of rows
     * @param cols Number of columns
     */
    Matrix(int rows, int cols);
    ~Matrix();

    /***
     * @brief Allows indexing the matrix's values directly
     * @param index Index of the desired row
     * @return Array containing the row of the matrix
     */
    float*  operator[](int index);

    /***
     * @return The number of rows in the matrix
     */
    int getRows() { return m_rowCount; }
    /***
     * @return The number of columns in the matrix
     */
    int getColumns() { return m_colCount; }
    /***
     * @brief Gets the size of the matrix
     * @param rows Pointer to the location to put the row value
     * @param cols Pointer to the location to put the column value
     */
    void getSize(int* rows, int* cols) {*rows = m_rowCount; *cols = m_colCount;}

    Matrix product(Matrix& mat);

    // scalar operations
    /***
     * @brief Adds a scalar value to each element of a matrix
     * @param num Scalar value to add
     * @return Resulting matrix after addition
     */
    Matrix  operator+ (float num);
    /***
     * @brief Adds a scalar value to each value of this matrix
     * @param num Scalar value to add to this matrix
     * @return Reference to this which has had the scalar value added to it
     */
    Matrix& operator+=(float num);
    /***
     * @brief Multiplies a scalar value by each element of a matrix
     * @param mul Scalar value to multiply this matrix by
     * @return Resulting matrix after multiplication
     */
    Matrix  operator* (float mul);
    /***
     * @brief Multiplies this matrix by a scalar value
     * @param mul Scalar value to multiply this matrix by
     * @return Reference to this after scalar multiplication
     */
    Matrix& operator*=(float mul);

    // element-wise operations
    /***
     * @brief Adds this matrix to another matrix and returns the result
     * @param mat Other matrix to add this matrix to
     * @return A matrix containing the result of the addition
     */
    Matrix  operator+ (Matrix& mat);
    /***
     * @brief Adds a matrix to this matrix and returns a reference to this
     * @param mat Matrix to add to this matrix
     * @return Reference to this which has been added to the other matrix
     */
    Matrix& operator+=(Matrix& mat);
    /***
     * @brief Multiplies this matrix by another matrix element-wise
     *          Element-wise meaning multiplying each element by its
     *          corresponding element
     * @param mat Other matrix to multiply
     * @return A matrix containing the result of the elemnt-wise multiplication
     */
    Matrix  operator* (Matrix& mat);
    /***
     * @brief Multiplies a matrix by this matrix element-wise
     * @param mat Other matrix to multiply by
     * @return Reference to this which has been multiplied element-wise
     */
    Matrix& operator*=(Matrix& mat);

    /***
     * @brief Checks if this matrix is equal to another one.
     * @param mat Matrix to test against
     * @return Whether or not the matrices are equal
     */
    bool operator==(Matrix& mat);
    /***
     * @brief Checks if this matrix is about equal to another one.
     *          Useful when accounting for float inaccuracies
     * @param mat Matrix to test against
     * @param err The amount of error allowed
     * @return Whether or not the matrices are about equal
     */
    bool equal(Matrix& mat, float err);

private:
    // values to keep track of the size
    int m_rowCount;
    int m_colCount;

    // the elements of the matrix
    float** m_values;
};
