#pragma once
#include <stdint.h>

class Model {
 public:
  // Number of state
  static constexpr uint16_t dim_x = 2;
  // Number of control input
  static constexpr uint16_t control_input = 1;
  // Number of constraint
  static constexpr uint16_t constraint = 1;
  // Number of dummy variable
  static constexpr uint16_t dummy = 1;
  // Number of variable
  static constexpr uint16_t dim_u = control_input + constraint + dummy;

  // private:
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

  // For objective function
  static constexpr double xf0 = 0.0, xf1 = 0.0;
  static constexpr double sf0 = 1.0, sf1 = 10.0;
  static constexpr double q0 = 1.0, q1 = 10.0;
  static constexpr double r0 = 1.0, r1 = 0.01;

  // For constraints
  static constexpr double umin = 0.0;
  static constexpr double umax = 1.0;
  static constexpr double uc = (umax + umin) / 2.0;
  static constexpr double ur = (umax - umin) / 2.0;

  // For internal system
  static constexpr double a = -1.0;
  static constexpr double b = -1.0;

  static void dxdt(double* ret, const double* x, const double* u) {
    ret[0] = x[1];
    ret[1] = a * x[0] + b * u[0] * x[1];
  }

  static void dPhidx(double* ret, const double* x) {
    ret[0] = x[0] * sf0;
    ret[1] = x[1] * sf1;
  }

  static void dHdx(double* ret, const double* x, const double* u,
                   const double* lmd) {
    ret[0] = x[0] * q0 + a * lmd[1];
    ret[1] = x[1] * q1 + lmd[0] + b * u[0] * lmd[1];
  }

  static void dHdu(double* ret, const double* x, const double* u,
                   const double* lmd) {
    ret[0] = r0 * u[0] + b * x[1] * lmd[1] + 2 * u[2] * (u[0] - uc);
    ret[1] = -r1 + 2 * u[1] * u[2];
    ret[2] = (u[0] - uc) * (u[0] - uc) + u[1] * u[1] - ur * ur;
  }

  static void ddHduu(double* ret, const double* x, const double* u,
                     const double* lmd) {
    ret[0] = r0 + 2 * u[2];
    ret[1] = 0;
    ret[2] = 2 * (u[0] - uc);

    ret[3] = 0;
    ret[4] = 2 * u[2];
    ret[5] = 2 * u[1];

    ret[6] = 2 * (u[0] - uc);
    ret[7] = 2 * u[1];
    ret[8] = 0;
  }
};