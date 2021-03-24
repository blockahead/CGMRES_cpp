#pragma once
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

class Simulator {
 public:
  static double* vector(int16_t row) {
    int16_t i;
    double* ret;
    ret = (double*)malloc(sizeof(double) * row);
    if (NULL == ret) {
      printf("Vector malloc() failure.");
    } else {
      for (i = 0; i < row; i++) {
        *(ret + i) = 0.0;
      }
    }
    return ret;
  }

  static void mov(double* ret, const double* vec, const int16_t row) {
    int16_t i;
    for (i = 0; i < row; i++) {
      ret[i] = vec[i];
    }
  }
  static void add(double* ret, const double* vec1, const double* vec2, const int16_t row) {
    int16_t i;
    for (i = 0; i < row; i++) {
      ret[i] = vec1[i] + vec2[i];
    }
  }

  static void mul(double* ret, const double* vec, const double c, const int16_t row) {
    int16_t i;
    for (i = 0; i < row; i++) {
      ret[i] = vec[i] * c;
    }
  }

  static void dxdt(double* ret, const double* x, const double* u) {
    ret[0] = x[2];
    ret[1] = x[3];
    ret[2] = -(k1 * k2) / m1 * x[0] + k2 / m1 * x[1] - (d1 + d2) / m1 * x[2] + d2 / m1 * x[3] + u[0] / m1;
    ret[3] = k2 / m2 * x[0] - k2 / m2 * x[1] + d2 / m2 * x[2] - d2 / m2 * x[3] + u[1] / m2;
  }

 private:
  // For internal system
  // For internal system
  static constexpr double m1 = 1.0, m2 = 1.0, d1 = 1.0, d2 = 1.0, k1 = 1.0, k2 = 1.0;
};