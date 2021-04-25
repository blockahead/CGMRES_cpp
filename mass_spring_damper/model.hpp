#pragma once
#include <stdint.h>

class Model {
 protected:
  // Number of state
  static constexpr uint16_t dim_x = 4;
  // Number of control input
  static constexpr uint16_t control_input = 2;
  // Number of constraint
  static constexpr uint16_t constraint = 2;
  // Number of dummy variable
  static constexpr uint16_t dummy = 2;
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
  static constexpr uint16_t dv = 50;
  // Prediction horizon (s)
  static constexpr double Tf = 1.0;
  // Rise rate for prediction horizon (-)
  static constexpr double alpha = 0.5;
  // Tolerance of convergence
  static constexpr double tol = 1e-6;
  // Maximum iteration of GMRES
  static constexpr uint16_t k_max = 5;

  static void dxdt(double* ret, const double* x, const double* u, const double* p) {
    ret[0] = x[2];
    ret[1] = x[3];
    ret[2] = -(k1 * k2) / m1 * x[0] + k2 / m1 * x[1] - (d1 + d2) / m1 * x[2] + d2 / m1 * x[3] + u[0] / m1;
    ret[3] = k2 / m2 * x[0] - k2 / m2 * x[1] + d2 / m2 * x[2] - d2 / m2 * x[3] + u[1] / m2;
  }

  static void dPhidx(double* ret, const double* x, const double* p) {
    ret[0] = -(p[0] - x[0]) * sf0;
    ret[1] = -(p[1] - x[1]) * sf1;
    ret[2] = x[2] * sf2;
    ret[3] = x[3] * sf3;
  }

  static void dHdx(double* ret, const double* x, const double* u, const double* p, const double* lmd) {
    ret[0] = -(p[0] - x[0]) * q0 - (k1 + k2) / m1 * lmd[2] + k2 / m2 * lmd[3];
    ret[1] = -(p[1] - x[1]) * q1 + k2 / m1 * lmd[2] - k2 / m2 * lmd[3];
    ret[2] = x[2] * q2 + lmd[0] - (d1 + d2) / m1 * lmd[2] + d2 / m2 * lmd[3];
    ret[3] = x[3] * q3 + lmd[1] + d2 / m1 * lmd[2] - d2 / m2 * lmd[3];
  }

  static void dHdu(double* ret, const double* x, const double* u, const double* p, const double* lmd) {
    ret[0] = r0 * u[0] + lmd[2] / m1 + 2.0 * u[4] * (u[0] - uc);
    ret[1] = r1 * u[1] + lmd[3] / m2 + 2.0 * u[5] * (u[1] - uc);
    ret[2] = -r2 + 2.0 * u[4] * u[2];
    ret[3] = -r3 + 2.0 * u[5] * u[3];
    ret[4] = (u[0] - uc) * (u[0] - uc) + u[2] * u[2] - ur * ur;
    ret[5] = (u[1] - uc) * (u[1] - uc) + u[3] * u[3] - ur * ur;
  }

  static void ddHduu(double* ret, const double* x, const double* u, const double* p, const double* lmd) {
    ret[0] = r0 + 2 * u[4];
    ret[1] = 0;
    ret[2] = 0;
    ret[3] = 0;
    ret[4] = 2 * (u[0] - uc);
    ret[5] = 0;

    ret[6] = 0;
    ret[7] = r1 + 2 * u[5];
    ret[8] = 0;
    ret[9] = 0;
    ret[10] = 0;
    ret[11] = 2 * (u[1] - uc);

    ret[12] = 0;
    ret[13] = 0;
    ret[14] = 2 * u[4];
    ret[15] = 0;
    ret[16] = 2 * u[2];
    ret[17] = 0;

    ret[18] = 0;
    ret[19] = 0;
    ret[20] = 0;
    ret[21] = 2 * u[5];
    ret[22] = 0;
    ret[23] = 2 * u[3];

    ret[24] = 2 * (u[0] - uc);
    ret[25] = 0;
    ret[26] = 2 * u[2];
    ret[27] = 0;
    ret[28] = 0;
    ret[29] = 0;

    ret[30] = 0;
    ret[31] = 2 * (u[1] - uc);
    ret[32] = 0;
    ret[33] = 2 * u[3];
    ret[34] = 0;
    ret[35] = 0;
  }

 private:
  // For objective function
  static constexpr double xf0 = 0.0, xf1 = 0.0;
  static constexpr double sf0 = 10.0, sf1 = 10.0, sf2 = 1.0, sf3 = 1.0;
  static constexpr double q0 = 1.0, q1 = 1.0, q2 = 10.0, q3 = 10.0;
  static constexpr double r0 = 0.1, r1 = 0.1, r2 = 0.01, r3 = 0.01;

  // For constraints
  static constexpr double umin = -10.0;
  static constexpr double umax = 10.0;
  static constexpr double uc = (umax + umin) / 2.0;
  static constexpr double ur = (umax - umin) / 2.0;

  // For internal system
  static constexpr double m1 = 1.0, m2 = 1.0, d1 = 1.0, d2 = 1.0, k1 = 1.0, k2 = 1.0;
};