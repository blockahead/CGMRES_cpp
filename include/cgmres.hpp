#pragma once
#include <float.h>
#include <stdint.h>
#include <stdio.h>

#include "matrix.hpp"

template <class Model>
class Cgmres : public Model {
 public:
  Cgmres(void) {
    t = 0.0;
    U = new double[dim_u * dv];
    dUdt = new double[dim_u * dv];

    x_dxh = new double[dim_x];
    xtau = new double[dim_x * (dv + 1)];
    ltau = new double[dim_x * (dv + 1)];
    ptau = new double[dim_p * (dv + 1)];

    F_dxh_h = new double[dim_u * dv];
    b_vec = new double[dim_u * dv];

    U_buf = new double[dim_u * dv];
  }

  ~Cgmres(void) {
    delete[] U;
    delete[] dUdt;

    delete[] x_dxh;
    delete[] xtau;
    delete[] ltau;
    delete[] ptau;

    delete[] F_dxh_h;
    delete[] b_vec;

    delete[] U_buf;
  }

  double get_dtau(const double t) {
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
    gmres(dUdt, b_vec, len, k_max, tol);

    // U = U + dUdt * dt
    mul(U_buf, dUdt, dt, len);
    add(U, U, U_buf, len);

    // t = t + dt
    // This value may not overflow due to loss of trailing digits
    t = t + dt;

    mov(u, &U[0], dim_u);
  }

 private:
  void F_func(double* ret, const double* U, const double* x, const double t) {
    uint16_t idx_x, idx_u, idx_p;
    double dtau;

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

  void Ax_func(double* Ax, const double* dUdt) {
    // F(U + dUdt * h, x + dxdt * h, t + h)
    mul(U_buf, dUdt, h, len);
    add(U_buf, U_buf, U, len);
    F_func(Ax, U_buf, x_dxh, t + h);

    // Ax = (F(U + dUdt * h, x + dxdt * h, t + h) - F(U, x + dxdt * h, t + h)) / h
    sub(Ax, Ax, F_dxh_h, len);
    div(Ax, Ax, h, len);
  }

  void gmres(double* dUdt, const double* b_vec, const uint16_t len, const uint16_t k_max, const double tol) {
    uint16_t k, idx_v1, idx_v2, idx_h, idx_g;
    double v_mat[(len) * (k_max + 1)];
    double h_mat[(k_max + 1) * (k_max + 1)];
    double rho_e_vec[k_max + 1];
    double g_vec[(g_vec_len) * (k_max)];
    double buf;
    double U_buf[len];

    // Ax = (F(U + dUdt * h, x + dxdt * h, t + h) - F(U, x + dxdt * h, t + h)) / h
    Ax_func(&v_mat[0], dUdt);

    // r0 = b - Ax
    sub(&v_mat[0], b_vec, &v_mat[0], len);

    // rho = sqrt(r0' * r0)
    rho_e_vec[0] = norm(&v_mat[0], len);

    if (rho_e_vec[0] < tol) {
      return;
    }

    // v(0) = r0 / rho
    div(&v_mat[0], &v_mat[0], rho_e_vec[0], len);

    for (k = 0; k < k_max; k++) {
      // v(k + 1) = (F(U + v(k) * h, x + dxdt * h, t + h) - F(U, x + dxdt * h, t + h)) / h
      Ax_func(&v_mat[len * (k + 1)], &v_mat[len * k]);

      idx_v1 = len * (k + 1);
      // Modified Gram-Schmidt
      for (uint16_t i = 0; i < k + 1; i++) {
        idx_v2 = len * i;
        idx_h = (k_max + 1) * k + i;
        h_mat[idx_h] = dot(&v_mat[idx_v2], &v_mat[idx_v1], len);
        mul(U_buf, &v_mat[idx_v2], h_mat[idx_h], len);
        sub(&v_mat[idx_v1], &v_mat[idx_v1], U_buf, len);
      }
      idx_h = (k_max + 1) * k + (k + 1);
      h_mat[idx_h] = norm(&v_mat[idx_v1], len);

      // Check breakdown
      if (fabs(h_mat[idx_h]) < DBL_EPSILON) {
        printf("Breakdown\n");
        return;
      } else {
        div(&v_mat[idx_v1], &v_mat[idx_v1], h_mat[idx_h], len);
      }

      // Transformation h_mat to upper triangular matrix by Householder transformation
      for (uint16_t i = 0; i < k; i++) {
        idx_h = (k_max + 1) * k + i;
        idx_g = g_vec_len * i;
        buf = (g_vec[idx_g + 0] * h_mat[idx_h + 0] + g_vec[idx_g + 1] * h_mat[idx_h + 1]) * g_vec[idx_g + 2];
        h_mat[idx_h + 0] = h_mat[idx_h + 0] - buf * g_vec[idx_g + 0];
        h_mat[idx_h + 1] = h_mat[idx_h + 1] - buf * g_vec[idx_g + 1];
      }
      idx_h = (k_max + 1) * k + k;
      idx_g = g_vec_len * k;
      buf = -sign(h_mat[idx_h]) * norm(&h_mat[idx_h], 2);  // Vector length
      g_vec[idx_g + 0] = h_mat[idx_h + 0] - buf;
      g_vec[idx_g + 1] = h_mat[idx_h + 1];
      g_vec[idx_g + 2] = 2.0 / dot(&g_vec[idx_g], &g_vec[idx_g], 2);
      h_mat[idx_h + 0] = buf;
      h_mat[idx_h + 1] = 0.0;

      // Update residual
      buf = g_vec[idx_g + 0] * rho_e_vec[k + 0] * g_vec[idx_g + 2];
      rho_e_vec[k + 0] = rho_e_vec[k + 0] - buf * g_vec[idx_g + 0];
      rho_e_vec[k + 1] = -buf * g_vec[idx_g + 1];

      // Check convergence
      if (fabs(rho_e_vec[k + 1]) < tol) {
        break;
      }
    }

    // Solve h_mat * y = rho_e_vec
    // h_mat is upper triangle matrix
    for (int16_t i = k - 1; i >= 0; i--) {
      for (int16_t j = k - 1; j > i; j--) {
        idx_h = (k_max + 1) * j + i;
        rho_e_vec[i] -= h_mat[idx_h] * rho_e_vec[j];
      }
      idx_h = (k_max + 1) * i + i;
      rho_e_vec[i] /= h_mat[idx_h];
    }

    // dUdt = dUdt + v_mat * y
    mul(&v_mat[len * k_max], v_mat, rho_e_vec, len, k);
    add(dUdt, dUdt, &v_mat[len * k_max], len);
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
  static constexpr double tol = Model::tol;
  static constexpr uint16_t k_max = Model::k_max;

 private:
  // Constants
  static constexpr uint16_t len = dim_u * dv;
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
  double* b_vec;

  double* U_buf;

  // Copy constructor
  Cgmres(const Cgmres&);
  Cgmres& operator=(const Cgmres&);
};