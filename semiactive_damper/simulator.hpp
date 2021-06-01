#pragma once
#include <stdint.h>

class Simulator {
 public:
  static constexpr double t_end = 20;
  static constexpr double dt = 0.001;
  static constexpr uint16_t dim_x = 2;
  static constexpr uint16_t dim_u = 3;
  static constexpr uint16_t dim_p = 0;

  static void dxdt(double* ret, const double* x, const double* u) {
    ret[0] = x[1];
    ret[1] = a * x[0] + b * u[0] * x[1];
  }

 private:
  // For internal system
  static constexpr double a = -1.0;
  static constexpr double b = -1.0;
};