#pragma once
#include <math.h>
#include <stdint.h>

class Model2 {
 protected:
  // Number of state
  static constexpr uint16_t dim_x = 4;
  // Number of control input
  static constexpr uint16_t control_input = 1;
  // Number of constraint
  static constexpr uint16_t constraint = 1;
  // Number of dummy variable
  static constexpr uint16_t dummy = 1;
  // Number of variable
  static constexpr uint16_t dim_u = control_input + constraint + dummy;
  // Number of parameter
  static constexpr uint16_t dim_p = 2;

  // Sampling period (s)
  static constexpr double dt = 0.001;
  // Forward difference period (s)
  static constexpr double h = 0.002;
  // Stabilize gain for control input (-)
  static constexpr double zeta = 1000.0;
  // Number of evaluation point (-)
  static constexpr uint16_t dv = 25;
  // Prediction horizon (s)
  static constexpr double Tf = 0.5;
  // Rise rate for prediction horizon (-)
  static constexpr double alpha = 0.5;
  // Tolerance of convergence
  static constexpr double tol = 1e-6;
  // Maximum iteration of GMRES
  static constexpr uint16_t k_max = 5;

  static void dxdt(double* ret, const double* x, const double* u, const double* p) {
    ret[0] = x[2];
    ret[1] = x[3];
    ret[2] = -As * x[2] + Bs * u[0];
    ret[3] = A32 * x[2] * x[2] * sin(x[0] - x[1]) + A52 * sin(x[1]) - A32b * cos(x[0] - x[1]) * u[0] + A32a * cos(x[0] - x[1]) * x[2] + C22 * (x[2] - x[3]);
  }

  static void dPhidx(double* ret, const double* x, const double* p) {
    ret[0] = (x[0] - p[0]) * sf0;
    ret[1] = (x[1] - p[1]) * sf1;
    ret[2] = x[2] * sf2;
    ret[3] = x[3] * sf3;
  }

  static void dHdx(double* ret, const double* x, const double* u, const double* p, const double* lmd) {
    ret[0] = (x[0] - p[0]) * q0 + lmd[3] * (A32 * x[2] * x[2] * cos(x[0] - x[1]) + A32b * sin(x[0] - x[1]) * u[0] - A32a * sin(x[0] - x[1]) * x[2]);
    ret[1] = (x[1] - p[1]) * q1 + lmd[3] * (-A32 * x[2] * x[2] * cos(x[0] - x[1]) + A52 * cos(x[1]) - A32b * sin(x[0] - x[1]) * u[0] + A32a * sin(x[0] - x[1]) * x[2]);
    ret[2] = x[2] * q2 + lmd[0] - lmd[2] * As + lmd[3] * (0.2e1 * A32 * x[2] * sin(x[0] - x[1]) + A32a * cos(x[0] - x[1]) + C22);
    ret[3] = x[3] * q3 + lmd[1] - lmd[3] * C22;
  }

  static void dHdu(double* ret, const double* x, const double* u, const double* p, const double* lmd) {
    ret[0] = (r0 * u[0]) + lmd[2] * Bs - lmd[3] * A32b * cos(x[0] - x[1]) + (double)(u[2] * (2.0 * u[0] - 2.0 * uc));
    ret[1] = -0.5 * r1 + (2.0 * u[2] * u[1]);
    ret[2] = (u[0] - uc) * (u[0] - uc) + u[1] * u[1] - ur * ur;
  }

  static void ddHduu(double* ret, const double* x, const double* u, const double* p, const double* lmd) {
    ret[0] = r0 + 2 * u[2];
    ret[1] = 0;
    ret[2] = 2 * u[0] - 2 * uc;

    ret[3] = 0;
    ret[4] = 2 * u[2];
    ret[5] = 2 * u[1];

    ret[6] = 2 * u[0] - 2 * uc;
    ret[7] = 2 * u[1];
    ret[8] = 0;
  }

 private:
  // For objective function
  static constexpr double xf0 = 0.0, xf1 = 0.0, xf2 = 0.0, xf3 = 0.0;
  static constexpr double sf0 = 3.0, sf1 = 1.0, sf2 = 0.0, sf3 = 0.0;
  static constexpr double q0 = 1.0, q1 = 1.0, q2 = 0.0, q3 = 0.0;
  static constexpr double r0 = 1.0, r1 = 0.1;

  // For constraints
  static constexpr double umin = -3.0;
  static constexpr double umax = 3.0;
  static constexpr double uc = (umax + umin) / 2.0;
  static constexpr double ur = (umax - umin) / 2.0;

  // For internal system
  static constexpr double As = 6.25;
  static constexpr double Bs = 15.6;
  static constexpr double A52 = 39.1111;
  static constexpr double C22 = 0.0407448;
  static constexpr double A32a = 5.65635;
  static constexpr double A32 = 0.905016;
  static constexpr double A32b = 14.1183;
};