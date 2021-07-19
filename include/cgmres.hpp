#pragma once
#include <stdint.h>
#include <stdio.h>

#include "gmres.hpp"
#include "matrix.hpp"

template <class Model>
class Cgmres : public Gmres {
 public:
  Cgmres(void) : Gmres(len, Model::k_max, Model::tol) {
    t = 0.0;
    U = new double[dim_u * dv];
    dUdt = new double[dim_u * dv];

    x_dxh = new double[dim_x];
    ptau = new double[dim_p * (dv + 1)];

    F_dxh_h = new double[dim_u * dv];
  }

  ~Cgmres(void) {
    delete[] U;
    delete[] dUdt;

    delete[] x_dxh;
    delete[] ptau;

    delete[] F_dxh_h;
  }

  double get_dtau(const double t) const {
    return Tf * (1 - exp(-alpha * t)) / (double)dv;
  }

  void set_ptau(const double* ptau_buf) {
    // ptau = [ p(t), p(t + dtau), ..., p(t + dv * dtau) ]
    mov(ptau, ptau_buf, dim_p * (dv + 1));
  }

  void set_ptau_repeat(const double* p_buf) {
    uint16_t idx;

    // ptau = [ p(t), p(t), ..., p(t) ]
    for (uint16_t i = 0; i < dv + 1; i++) {
      idx = dim_p * i;
      mov(&ptau[idx], p_buf, dim_p);
    }
  }

  void init_u0(const double* u0) {
    uint16_t idx;

    // U(i) = u0
    for (uint16_t i = 0; i < dv; i++) {
      idx = dim_u * i;
      mov(&U[idx], u0, dim_u);
    }
  }

  void init_u0_newton(double* u0, const double* x0, const double* p0, const uint16_t n_loop) {
    double lmd0[dim_x], vec[dim_u], mat[dim_u * dim_u];

    Model::dPhidx(lmd0, x0, p0);

    // u0 = u0 - dHduu \ dHdu
    for (uint16_t i = 0; i < n_loop; i++) {
      Model::dHdu(vec, x0, u0, p0, lmd0);
      Model::ddHduu(mat, x0, u0, p0, lmd0);
      linsolve(vec, mat, dim_u);

      sub(u0, u0, vec, dim_u);
    }

    init_u0(u0);
  }

  void control(double* u, const double* x) {
    double b_vec[len];
    double U_buf[len];

    // x + dxdt * h
    Model::dxdt(x_dxh, x, &U[0], &ptau[0]);
    mul(x_dxh, x_dxh, h, dim_x);
    add(x_dxh, x_dxh, x, dim_x);

    // F(U, x + dxdt * h, t + h)
    F_func(F_dxh_h, U, x_dxh, t + h);

    // F(U, x, t)
    F_func(b_vec, U, x, t);

    // (F(U, x, t) * (1 - zeta * h) - F(U, x + dxdt * h, t + h)) / h
    mul(b_vec, b_vec, (1 - zeta * h), len);
    sub(b_vec, b_vec, F_dxh_h, len);
    div(b_vec, b_vec, h, len);

    // GMRES
    gmres(dUdt, b_vec);

    // U = U + dUdt * dt
    mul(U_buf, dUdt, dt, len);
    add(U, U, U_buf, len);

    // t = t + dt
    // This value may not overflow due to loss of trailing digits
    t = t + dt;

    mov(u, &U[0], dim_u);
  }

 private:
  void F_func(double* ret, const double* U, const double* x, const double t) const {
    uint16_t idx_x, idx_u, idx_p;
    double dtau;
    double xtau[dim_x * (dv + 1)];
    double ltau[dim_x * (dv + 1)];

#ifdef DEBUG_MODE
    if (ret == U) {
      printf("%s pointer error ! (U_vec_tmp is overwritten due to the same address of ret)\n", __func__);
      exit(-1);
    }
#endif

    // Prediction horizon
    dtau = get_dtau(t);

    // State equation
    // x(0) = x
    // x(i + 1) = x(i) + dxdt(x(i), u(i), p(i)) * dtau
    mov(&xtau[0], x, dim_x);
    for (uint16_t i = 0; i < dv; i++) {
      idx_x = dim_x * i;
      idx_u = dim_u * i;
      idx_p = dim_p * i;
      Model::dxdt(&xtau[idx_x + dim_x], &xtau[idx_x], &U[idx_u], &ptau[idx_p]);
      mul(&xtau[idx_x + dim_x], &xtau[idx_x + dim_x], dtau, dim_x);
      add(&xtau[idx_x + dim_x], &xtau[idx_x + dim_x], &xtau[idx_x], dim_x);
    }

    // Adjoint equation
    // lmd(N) = dPhidx(x(N), p(N))
    // lmd(i) = lmd(i + 1) + dHdx(x(i), u(i), p(i), lmd(i + 1)) * dtau
    Model::dPhidx(&ltau[dim_x * dv], &xtau[dim_x * dv], &ptau[dim_p * dv]);
    for (int16_t i = dv - 1; i >= 0; i--) {
      idx_x = dim_x * i;
      idx_u = dim_u * i;
      idx_p = dim_p * i;
      Model::dHdx(&ltau[idx_x], &xtau[idx_x], &U[idx_u], &ptau[idx_p], &ltau[idx_x + dim_x]);
      mul(&ltau[idx_x], &ltau[idx_x], dtau, dim_x);
      add(&ltau[idx_x], &ltau[idx_x], &ltau[idx_x + dim_x], dim_x);
    }

    // F(i) = dHdU(x(i), u(i), lmd(i + 1))
    for (uint16_t i = 0; i < dv; i++) {
      idx_x = dim_x * i;
      idx_u = dim_u * i;
      idx_p = dim_p * i;
      Model::dHdu(&ret[idx_u], &xtau[idx_x], &U[idx_u], &ptau[idx_p], &ltau[idx_x + dim_x]);
    }
  }

  void Ax_func(double* Ax, const double* dUdt) override {
    double U_buf[len];

    // F(U + dUdt * h, x + dxdt * h, t + h)
    mul(U_buf, dUdt, h, len);
    add(U_buf, U_buf, U, len);
    F_func(Ax, U_buf, x_dxh, t + h);

    // Ax = (F(U + dUdt * h, x + dxdt * h, t + h) - F(U, x + dxdt * h, t + h)) / h
    sub(Ax, Ax, F_dxh_h, len);
    div(Ax, Ax, h, len);
  }

 public:
  // Parameters
  static constexpr uint16_t dim_x = Model::dim_x;
  static constexpr uint16_t dim_u = Model::dim_u;
  static constexpr uint16_t dim_p = Model::dim_p;

  static constexpr double dt = Model::dt;
  static constexpr double h = Model::h;
  static constexpr double zeta = Model::zeta;
  static constexpr uint16_t dv = Model::dv;
  static constexpr double Tf = Model::Tf;
  static constexpr double alpha = Model::alpha;

 private:
  // Constants
  static constexpr uint16_t len = dim_u * dv;

  // Variables
  double t;
  double* U;
  double* dUdt;

  double* x_dxh;
  double* ptau;

  double* F_dxh_h;

  // Copy constructor
  Cgmres(const Cgmres&);
  Cgmres& operator=(const Cgmres&);
};