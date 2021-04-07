#pragma once
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define DEBUG_MODE

// vector allocation
inline double* vector(int16_t row) {
  int16_t i;
  double* ret;
  ret = (double*)malloc(sizeof(double) * row);
#ifdef DEBUG_MODE
  if (NULL == ret) {
    printf("Vector malloc() failure.");
  }
#endif
  for (i = 0; i < row; i++) {
    *(ret + i) = 0.0;
  }

  return ret;
}

// vector allocation
inline double* matrix(int16_t row, int16_t col) {
  int16_t i;
  double* ret;
  ret = (double*)malloc(sizeof(double) * row * col);
#ifdef DEBUG_MODE
  if (NULL == ret) {
    printf("Matrix malloc() failure.");
  }
#endif
  for (i = 0; i < row * col; i++) {
    *(ret + i) = 0.0;
  }

  return ret;
}

// ret = vec
inline void mov(double* ret, const double* vec, const int16_t row) {
  int16_t i;
  for (i = 0; i < row; i++) {
    ret[i] = vec[i];
  }
}

// ret = mat
inline void mov(double* ret, const double* mat, const int16_t row, const int16_t col) {
  int16_t i;
  for (i = 0; i < row * col; i++) {
    ret[i] = mat[i];
  }
}

// ret = vec1 + vec2
inline void add(double* ret, const double* vec1, const double* vec2, const int16_t row) {
  int16_t i;
  for (i = 0; i < row; i++) {
    ret[i] = vec1[i] + vec2[i];
  }
}

// ret = mat1 + mat2
inline void add(double* ret, const double* mat1, const double* mat2, const int16_t row, const int16_t col) {
  int16_t i;
  for (i = 0; i < row * col; i++) {
    ret[i] = mat1[i] + mat2[i];
  }
}

// ret = vec1 - vec2
inline void sub(double* ret, const double* vec1, const double* vec2, const int16_t row) {
  int16_t i;
  for (i = 0; i < row; i++) {
    ret[i] = vec1[i] - vec2[i];
  }
}

// ret = mat1 - mat2
inline void sub(double* ret, const double* mat1, const double* mat2, const int16_t row, const int16_t col) {
  int16_t i;
  for (i = 0; i < row * col; i++) {
    ret[i] = mat1[i] - mat2[i];
  }
}

// ret = vec * c
inline void mul(double* ret, const double* vec, const double c, const int16_t row) {
  int16_t i;
  for (i = 0; i < row; i++) {
    ret[i] = vec[i] * c;
  }
}

// ret = mat * c
inline void mul(double* ret, const double* mat, const double c, const int16_t row, const int16_t col) {
  int16_t i;
  for (i = 0; i < row * col; i++) {
    ret[i] = mat[i] * c;
  }
}

// ret = mat * vec
inline void mul(double* ret, const double* mat, const double* vec, const int16_t row, const int16_t col) {
  int16_t i, j, idx;
#ifdef DEBUG_MODE
  if (ret == vec) {
    printf("%s pointer error !\n", __func__);
    exit(-1);
  }
#endif
  for (i = 0; i < row; i++) {
    ret[i] = 0.0;
  }

  for (j = 0; j < col; j++) {
    for (i = 0; i < row; i++) {
      idx = row * j + i;
      ret[i] += mat[idx] * vec[j];
    }
  }
}

// ret = mat * mat
inline void mul(double* ret, const double* mat1, const double* mat2, const int16_t l, const int16_t row, const int16_t col) {
  int16_t i, j, k, idx1, idx2, idx3;
#ifdef DEBUG_MODE
  if (ret == mat1 || ret == mat2) {
    printf("%s pointer error !\n", __func__);
    exit(-1);
  }
#endif

  for (i = 0; i < row; i++) {
    for (j = 0; j < col; j++) {
      idx1 = col * i + j;
      ret[idx1] = 0;
    }

    for (k = 0; k < l; k++) {
      idx2 = col * i + k;
      for (j = 0; j < col; j++) {
        idx1 = col * i + j;
        idx3 = col * k + j;
        ret[idx1] += mat1[idx2] * mat2[idx3];
      }
    }
  }
}

// ret = vec / c
inline void div(double* ret, const double* vec, const double c, const int16_t row) {
  int16_t i;
  double inv_c = 1.0 / c;
  for (i = 0; i < row; i++) {
    ret[i] = vec[i] * inv_c;
  }
}

// ret = mat / c
inline void div(double* ret, const double* mat, const double c, const int16_t row, const int16_t col) {
  int16_t i;
  double inv_c = 1.0 / c;
  for (i = 0; i < row * col; i++) {
    ret[i] = mat[i] * inv_c;
  }
}

// ret = norm(vec)
inline double norm(const double* vec, int16_t n) {
  int16_t i;
  double ret = 0;
  for (i = 0; i < n; i++) {
    ret += vec[i] * vec[i];
  }

  return sqrt(ret);
}

// ret = vec1' * vec2
inline double dot(const double* vec1, const double* vec2, const int16_t n) {
  int16_t i;
  double ret = 0;
  for (i = 0; i < n; i++) {
    ret += vec1[i] * vec2[i];
  }

  return ret;
}

// ret = sign(x)
inline double sign(const double x) { return (x < 0.0) ? -1.0 : 1.0; }
