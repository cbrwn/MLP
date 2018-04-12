#include "matrix.hpp"

#include <iostream>

#include "gmath.h"

Matrix::Matrix(int rows, int cols)
        : m_rowCount(rows), m_colCount(cols)
{
    m_values = new float*[rows];

    // initialize all elements to 0
    for(int i = 0; i < rows; ++i)
    {
        m_values[i] = new float[cols];
        for(int j = 0; j < cols; ++j)
            m_values[i][j] = 0.0f;
    }
}

Matrix::~Matrix()
{
    for(int i = 0; i < m_rowCount; ++i)
        delete[] m_values[i];
    delete[] m_values;
}

// Copy constructor
Matrix::Matrix(Matrix& mat)
{
    mat.getSize(&m_rowCount, &m_colCount);

    m_values = new float*[m_rowCount];
    for(int i = 0; i < m_rowCount; ++i)
    {
        m_values[i] = new float[m_colCount];
        for(int j = 0; j < m_colCount; ++j)
            m_values[i][j] = mat[i][j];
    }
}

// Copy assignment operator
Matrix& Matrix::operator=(Matrix& mat)
{
    mat.getSize(&m_rowCount, &m_colCount);

    m_values = new float*[m_rowCount];
    for(int i = 0; i < m_rowCount; ++i)
    {
        m_values[i] = new float[m_colCount];
        for(int j = 0; j < m_colCount; ++j)
            m_values[i][j] = mat[i][j];
    }

    return *this;
}

// Move constructor
Matrix::Matrix(Matrix&& mat)
{
    mat.getSize(&m_rowCount, &m_colCount);

    m_values = new float*[m_rowCount];
    for(int i = 0; i < m_rowCount; ++i)
    {
        m_values[i] = new float[m_colCount];
        for(int j = 0; j < m_colCount; ++j)
        {
            m_values[i][j] = mat[i][j];
            mat[i][j] = 0.0f;
        }
    }

}

// Move assignment operator
Matrix& Matrix::operator=(Matrix&& mat)
{
    mat.getSize(&m_rowCount, &m_colCount);

    m_values = new float*[m_rowCount];
    for(int i = 0; i < m_rowCount; ++i)
    {
        m_values[i] = new float[m_colCount];
        for(int j = 0; j < m_colCount; ++j)
        {
            m_values[i][j] = mat[i][j];
            mat[i][j] = 0.0f;
        }
    }

    return *this;
}

float* Matrix::operator[](int index)
{
    return m_values[index];
}

Matrix Matrix::transposed()
{
    Matrix m(m_colCount, m_rowCount);
    for(int y = 0; y < m_rowCount; ++y)
        for(int x = 0; x < m_colCount; ++x)
            m[x][y] = m_values[y][x];
    return m;
}

Matrix Matrix::product(Matrix& mat)
{
    // matrices must match rows/columns and columns/rows
    if(mat.getRows() != m_colCount)
    {
        printf("PRODUCT SIZE MISMATCH!!!\n");
        return *this;
    }

    Matrix m(m_rowCount, mat.getColumns());
    for(int i = 0; i < m.getRows(); ++i)
    {
        for(int j = 0; j < m.getColumns(); ++j)
        {
            float sum = 0.0f;
            for(int k = 0; k < m_colCount; ++k)
                sum += (m_values[i][k] * mat[k][j]);
            m[i][j] = sum;
        }
    }
    return m;
/*
    Matrix m(m_rowCount, m_rowCount);
    for(int a = 0; a < m_rowCount; ++a)
    {
        for(int b = 0; b < m_rowCount; ++b)
        {
            float sum = 0.0f;
            for(int i = 0; i < m_colCount; ++i)
                sum += m_values[a][i] * mat[i][b];
            m[a][b] = sum;
        }
    }

    return m;*/
}

void Matrix::map(ModifyFunction func)
{
    for(int y = 0; y < m_rowCount; ++y)
        for(int x = 0; x < m_colCount; ++x)
            m_values[y][x] = func(m_values[y][x]);
}

Matrix Matrix::operator*(float mul)
{
    Matrix m(m_rowCount, m_colCount);

    for(int y = 0; y < m_rowCount; ++y)
        for(int x = 0; x < m_colCount; ++x)
            m[y][x] = m_values[y][x] * mul;

    return m;
}

Matrix& Matrix::operator*=(float mul)
{
    for(int y = 0; y < m_rowCount; ++y)
        for(int x = 0; x < m_colCount; ++x)
            m_values[y][x] *= mul;
    return *this;
}

Matrix Matrix::operator+(float num)
{
    Matrix m(m_rowCount, m_colCount);

    for(int y = 0; y < m_rowCount; ++y)
        for(int x = 0; x < m_colCount; ++x)
            m[y][x] = m_values[y][x] + num;

    return m;
}

Matrix& Matrix::operator+=(float num)
{
    for(int y = 0; y < m_rowCount; ++y)
        for(int x = 0; x < m_colCount; ++x)
            m_values[y][x] += num;
    return *this;
}

Matrix Matrix::operator+(Matrix& mat)
{
    if(mat.getRows() != m_rowCount
            || mat.getColumns() != m_colCount)
    {
        // columns and rows don't match
        return *this;
    }

    Matrix m(m_rowCount, m_colCount);
    for(int y = 0; y < m_rowCount; ++y)
        for(int x = 0; x < m_colCount; ++x)
            m[y][x] = m_values[y][x] + mat[y][x];

    return m;
}

Matrix& Matrix::operator+=(Matrix& mat)
{
    if(mat.getRows() != m_rowCount
       || mat.getColumns() != m_colCount)
    {
        // columns and rows don't match
        return *this;
    }

    for(int y = 0; y < m_rowCount; ++y)
        for(int x = 0; x < m_colCount; ++x)
            m_values[y][x] += mat[y][x];

    return *this;
}

Matrix Matrix::operator-(Matrix& mat)
{
    if(mat.getRows() != m_rowCount
       || mat.getColumns() != m_colCount)
    {
        // columns and rows don't match
        return *this;
    }

    Matrix m(m_rowCount, m_colCount);
    for(int y = 0; y < m_rowCount; ++y)
        for(int x = 0; x < m_colCount; ++x)
            m[y][x] = m_values[y][x] - mat[y][x];

    return m;
}

Matrix& Matrix::operator-=(Matrix& mat)
{
    if(mat.getRows() != m_rowCount
       || mat.getColumns() != m_colCount)
    {
        // columns and rows don't match
        return *this;
    }

    for(int y = 0; y < m_rowCount; ++y)
        for(int x = 0; x < m_colCount; ++x)
            m_values[y][x] -= mat[y][x];

    return *this;
}

Matrix Matrix::operator*(Matrix& mat)
{
    if(mat.getRows() != m_rowCount
       || mat.getColumns() != m_colCount)
    {
        // columns and rows don't match
        return *this;
    }

    Matrix m(m_rowCount, m_colCount);
    for(int y = 0; y < m_rowCount; ++y)
        for(int x = 0; x < m_colCount; ++x)
            m[y][x] = m_values[y][x] * mat[y][x];

    return m;
}

Matrix& Matrix::operator*=(Matrix& mat)
{
    if(mat.getRows() != m_rowCount
       || mat.getColumns() != m_colCount)
    {
        // columns and rows don't match
        return *this;
    }

    for(int y = 0; y < m_rowCount; ++y)
        for(int x = 0; x < m_colCount; ++x)
            m_values[y][x] *= mat[y][x];

    return *this;
}

bool Matrix::operator==(Matrix& mat)
{
    if(mat.getRows() != m_rowCount
       || mat.getColumns() != m_colCount)
    {
        // columns and rows don't match
        return false;
    }

    for(int y = 0; y < m_rowCount; ++y)
        for(int x = 0; x < m_colCount; ++x)
            if(m_values[y][x] != mat[y][x])
                return false;

    return true;
}

bool Matrix::equal(Matrix& mat, float err)
{
    if(mat.getRows() != m_rowCount
       || mat.getColumns() != m_colCount)
    {
        // columns and rows don't match
        return false;
    }

    for(int y = 0; y < m_rowCount; ++y)
        for(int x = 0; x < m_colCount; ++x)
            if(absf(m_values[y][x] - mat[y][x]) > err)
                return false;

    return true;
}

void Matrix::randomize()
{
    for(int y = 0; y < m_rowCount; ++y)
        for(int x = 0; x < m_colCount; ++x)
            m_values[y][x] = randBetween(-1.0f, 1.0f);
}

void Matrix::print(const char* fmt)
{
    for(int y = 0; y < m_rowCount; ++y)
    {
        printf("[ ");
        for(int x = 0; x < m_colCount; ++x)
        {
            printf(fmt, m_values[y][x]);
        }
        printf("]\n");
    }
    printf("\n");
}
