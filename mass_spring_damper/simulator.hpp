#pragma once
#include <stdint.h>

class Simulator {
 public:
  static constexpr double t_end = 20;
  static constexpr double dt = 0.001;
  static constexpr uint16_t dim_x = 4;
  static constexpr uint16_t dim_u = 6;

  static void dxdt(double* ret, const double* x, const double* u) {
    ret[0] = x[2];
    ret[1] = x[3];
    ret[2] = -(k1 * k2) / m1 * x[0] + k2 / m1 * x[1] - (d1 + d2) / m1 * x[2] + d2 / m1 * x[3] + u[0] / m1;
    ret[3] = k2 / m2 * x[0] - k2 / m2 * x[1] + d2 / m2 * x[2] - d2 / m2 * x[3] + u[1] / m2;
  }

 private:
  // For internal system
  static constexpr double m1 = 1.0, m2 = 1.0, d1 = 1.0, d2 = 1.0, k1 = 1.0, k2 = 1.0;
};