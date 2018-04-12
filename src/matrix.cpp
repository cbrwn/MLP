#include "matrix.hpp"

#include "math.h"

Matrix::Matrix(int rows, int cols)
        : m_rowCount(rows), m_colCount(cols)
{
    m_values = new float*[rows];
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

float* Matrix::operator[](int index)
{
    return m_values[index];
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
            if(abs(m_values[y][x] - mat[y][x]) > err)
                return false;

    return true;
}
