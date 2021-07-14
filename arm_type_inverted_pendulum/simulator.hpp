#pragma once
#include <math.h>
#include <stdint.h>

class Simulator {
 public:
  static constexpr double t_end = 10;
  static constexpr double dt = 0.001;
  static constexpr uint16_t dim_x = 4;
  static constexpr uint16_t dim_u = 3;
  static constexpr uint16_t dim_p = 2;
  static constexpr uint16_t dv = 25;

  static void dxdt(double* ret, const double* x, const double* u) {
    ret[0] = x[2];
    ret[1] = x[3];
    ret[2] = -As * x[2] + Bs * u[0];
    ret[3] = A32 * x[2] * x[2] * sin(x[0] - x[1]) + A52 * sin(x[1]) - A32b * cos(x[0] - x[1]) * u[0] + A32a * cos(x[0] - x[1]) * x[2] + C22 * (x[2] - x[3]);
  }

 private:
  // For internal system
  static constexpr double As = 6.25;
  static constexpr double Bs = 15.6;
  static constexpr double A52 = 39.1111;
  static constexpr double C22 = 0.0407448;
  static constexpr double A32a = 5.65635;
  static constexpr double A32 = 0.905016;
  static constexpr double A32b = 14.1183;
};