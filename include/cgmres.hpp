#pragma once
#include <float.h>
#include <stdint.h>
#include <stdio.h>

#include "matrix.hpp"
#include "model.hpp"

class Cgmres : public Model {
 public:
  Cgmres(double* u0);
  ~Cgmres(void);
  void u0_newton(double* u0);
  void set_p(const double* pt);
  void control(double* u, const double* x);

 private:
  // Constants
  static constexpr uint16_t g_vec_len = 3;

  // Variables
  double t;
  double* U;
  double* dUdt;

  double* x_dxh;
  double* xtau;
  double* ltau;
  double* ptau;

  double* F_dxh_h;
  double* F_dUh_dxh_h;
  double* b_vec;

  double* v_mat;
  double* h_mat;
  double* rho_e_vec;
  double* g_vec;

  double* U_buf;

  void F_func(double* ret, const double* U_vec, const double* x, const double t);
  void gmres();
};