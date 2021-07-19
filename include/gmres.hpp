#pragma once

#include <float.h>
#include <stdint.h>

#include "matrix.hpp"

class Gmres {
 protected:
  Gmres(const uint16_t len, const uint16_t k_max, const double tol) : len(len), k_max(k_max), tol(tol) {
    v_mat = new double[(len) * (k_max + 1)];
    h_mat = new double[(k_max + 1) * (k_max + 1)];
    rho_e_vec = new double[k_max + 1];
    g_vec = new double[(g_vec_len) * (k_max)];
    U_buf = new double[len];
  }

  ~Gmres() {
    delete[] v_mat;
    delete[] h_mat;
    delete[] rho_e_vec;
    delete[] g_vec;
    delete[] U_buf;
  }

  virtual void Ax_func(double *Ax, const double *x) = 0;

  void gmres(double *x, const double *b_vec) {
    uint16_t k, idx_v1, idx_v2, idx_h, idx_g;
    double buf;

    // r0 = b - Ax(x0)
    Ax_func(&v_mat[0], x);
    sub(&v_mat[0], b_vec, &v_mat[0], len);

    // rho = sqrt(r0' * r0)
    rho_e_vec[0] = norm(&v_mat[0], len);

    if (rho_e_vec[0] < tol) {
      return;
    }

    // v(0) = r0 / rho
    div(&v_mat[0], &v_mat[0], rho_e_vec[0], len);

    for (k = 0; k < k_max; k++) {
      // v(k + 1) = Ax(v(k))
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

    // x = x + v_mat * y
    mul(&v_mat[len * k_max], v_mat, rho_e_vec, len, k);
    add(x, x, &v_mat[len * k_max], len);
  }

 private:
  static constexpr uint16_t g_vec_len = 3;
  const uint16_t len;
  const uint16_t k_max;
  const double tol;

  double *v_mat;
  double *h_mat;
  double *rho_e_vec;
  double *g_vec;
  double *U_buf;

  // Copy constructor
  Gmres(const Gmres &);
  Gmres &operator=(const Gmres &);
};