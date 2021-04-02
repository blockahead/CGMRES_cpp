#pragma once

class Simulator {
 public:
  static constexpr double t_end = 20;
  static constexpr double dt = 0.001;
  static constexpr uint16_t dim_x = 2;
  static constexpr uint16_t dim_u = 3;

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
    ret[0] = x[1];
    ret[1] = a * x[0] + b * u[0] * x[1];
  }

 private:
  // For internal system
  static constexpr double a = -1.0;
  static constexpr double b = -1.0;
};