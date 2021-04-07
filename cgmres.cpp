#include "cgmres.hpp"

Cgmres::Cgmres(double* u0) {
  t = 0.0;
  U = vector(dim_u * dv);
  dUdt = vector(dim_u * dv);

  x_dxh = vector(dim_x);
  xtau = vector(dim_x * (dv + 1));
  ltau = vector(dim_x * (dv + 1));

  F_dxh_h = vector(dim_u * dv);
  F_dUh_dxh_h = vector(dim_u * dv);
  b_vec = vector(dim_u * dv);

  v_mat = matrix(dim_u * dv, k_max + 1);
  h_mat = matrix(k_max + 1, k_max + 1);
  rho_e_vec = vector(k_max + 1);
  g_vec = matrix(g_vec_len, k_max);

  U_buf = vector(dim_u * dv);

  for (int16_t i = 0; i < dv; i++) {
    int16_t idx = dim_u * i;
    mov(&U[idx], u0, dim_u);
  }
}

Cgmres::~Cgmres(void) {
  free(U);
  free(dUdt);

  free(x_dxh);
  free(xtau);
  free(ltau);

  free(F_dxh_h);
  free(F_dUh_dxh_h);
  free(b_vec);

  free(v_mat);
  free(h_mat);
  free(rho_e_vec);
  free(g_vec);

  free(U_buf);
}

void Cgmres::u0_newton(double* u0) {
  // TODO: add Newton method using ddHduu

  for (int16_t i = 0; i < dv; i++) {
    int16_t idx = dim_u * i;
    mov(&U[idx], u0, dim_u);
  }
}

void Cgmres::F_func(double* ret, const double* U, const double* x, const double t) {
  int16_t i, idx_x, idx_u;
  double dtau;

#ifdef DEBUG_MODE
  if (ret == U) {
    printf(
        "%s pointer error ! (U_vec_tmp is overwritten due to the same address "
        "of ret)\n",
        __func__);
    exit(-1);
  }
#endif

  // Prediction horizon
  dtau = Tf * (1 - exp(-alpha * t)) / (double)dv;

  // State equation
  // x(0) = x
  // x(i+1) = x(i) + dxdt(x(i),u(i)) * dtau
  mov(xtau, x, dim_x);
  for (i = 0; i < dv; i++) {
    idx_x = dim_x * i;
    idx_u = dim_u * i;
    Model::dxdt(&xtau[idx_x + dim_x], &xtau[idx_x], &U[idx_u]);
    mul(&xtau[idx_x + dim_x], &xtau[idx_x + dim_x], dtau, dim_x);
    add(&xtau[idx_x + dim_x], &xtau[idx_x + dim_x], &xtau[idx_x], dim_x);
  }

  // Adjoint equation
  // lmd(N) = dPhidx(x(N))
  // lmd(i) = lmd(i+1) + dHdx(x(i),u(i),lmd(i+1)) * dtau
  Model::dPhidx(&ltau[dim_x * dv], &xtau[dim_x * dv]);
  for (i = dv - 1; i >= 0; i--) {
    idx_x = dim_x * i;
    idx_u = dim_u * i;
    Model::dHdx(&ltau[idx_x], &xtau[idx_x], &U[idx_u], &ltau[idx_x + dim_x]);
    mul(&ltau[idx_x], &ltau[idx_x], dtau, dim_x);
    add(&ltau[idx_x], &ltau[idx_x], &ltau[idx_x + dim_x], dim_x);
  }

  // F(i) = dHdU(x(i),u(i),lmd(i+1))
  for (i = 0; i < dv; i++) {
    idx_x = dim_x * i;
    idx_u = dim_u * i;
    Model::dHdu(&ret[idx_u], &xtau[idx_x], &U[idx_u], &ltau[idx_x + dim_x]);
  }
}

void Cgmres::gmres() {
  int16_t len, i, j, k, idx_v1, idx_v2, idx_h, idx_g;
  double buf;

  // F_dUh_dxh_h = F(U + dUdt * h, x + dxdt * h, t + h)
  len = dim_u * dv;
  mul(U_buf, dUdt, h, len);
  add(U_buf, U_buf, U, len);
  F_func(F_dUh_dxh_h, U_buf, x_dxh, t + h);

  // Ax = (F(U + dUdt * h, x + dxdt * h, t + h) - F(U, x + dxdt * h, t + h)) / h
  sub(U_buf, F_dUh_dxh_h, F_dxh_h, len);
  div(U_buf, U_buf, h, len);

  // r0 = b - Ax
  sub(U_buf, b_vec, U_buf, len);

  // rho = sqrt(r0' * r0)
  rho_e_vec[0] = norm(U_buf, len);

  if (rho_e_vec[0] < tol) {
    k = 0;
    return;
  }

  // v_mat[:][0] = r0 / rho
  div(&v_mat[0], U_buf, rho_e_vec[0], len);

  for (k = 0; k < k_max; k++) {
    // F_dUh_dxh_h = F(U + v[k] * h, x + dxdt * h, t + h)
    idx_v1 = len * k;
    mul(U_buf, &v_mat[idx_v1], h, len);
    add(U_buf, U, U_buf, len);
    F_func(F_dUh_dxh_h, U_buf, x_dxh, t + h);

    // v[k + 1] = (F(U + v[k] * h, x + dxdt * h, t + h) - F(U, x + dxdt * h, t + h)) / h
    idx_v1 = len * (k + 1);
    sub(U_buf, F_dUh_dxh_h, F_dxh_h, len);
    div(&v_mat[idx_v1], U_buf, h, len);

    // Modified Gram-Schmidt
    for (i = 0; i < k + 1; i++) {
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
    for (i = 0; i < k; i++) {
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
  for (i = k - 1; i >= 0; i--) {
    for (j = k - 1; j > i; j--) {
      idx_h = (k_max + 1) * j + i;
      rho_e_vec[i] -= h_mat[idx_h] * rho_e_vec[j];
    }
    idx_h = (k_max + 1) * i + i;
    rho_e_vec[i] /= h_mat[idx_h];
  }

  // dUdt = dUdt + v_mat * y
  len = dim_u * dv;
  mul(U_buf, v_mat, rho_e_vec, len, k);
  add(dUdt, dUdt, U_buf, len);
}

void Cgmres::control(double* u, const double* x) {
  int16_t len;

  // x_dxh = x + dxdt * h
  len = dim_x;
  Model::dxdt(x_dxh, x, U);
  mul(x_dxh, x_dxh, h, len);
  add(x_dxh, x_dxh, x, len);

  // F_dxh_h = F(U, x + dxdt * h, t + h)
  F_func(F_dxh_h, U, x_dxh, t + h);

  // rhs = F(U, x, t)
  F_func(b_vec, U, x, t);

  // rhs = ( F(U, x, t) * ( 1 - zeta * h ) - F(U, x + dxdt * h, t + h) ) / h;
  len = dim_u * dv;
  mul(b_vec, b_vec, (1 - zeta * h), len);
  sub(b_vec, b_vec, F_dxh_h, len);
  div(b_vec, b_vec, h, len);

  // GMRES
  gmres();

  // U = U + dUdt * dt;
  len = dim_u * dv;
  mul(U_buf, dUdt, dt, len);
  add(U, U, U_buf, len);

  // t = t + dt
  // This value may not overflow due to loss of trailing digits
  t = t + dt;

  mov(u, U, dim_u);
}