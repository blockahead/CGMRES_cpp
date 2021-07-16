#include <stdio.h>
#include <sys/time.h>

#include <iostream>

#include "cgmres.hpp"
#include "matrix.hpp"
#include "model.hpp"
#include "simulator.hpp"

double getEtime(void) {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (double)tv.tv_sec + (double)tv.tv_usec * 1e-6;
}

int main(void) {
  //---------------------------
  FILE* fp_x;
  FILE* fp_u;

  double t_start, t_end, t_all = 0;

  double* x;
  double* u;
  double* dxdt;
  double* pt;

  x = new double[Simulator::dim_x];
  u = new double[Simulator::dim_u];
  dxdt = new double[Simulator::dim_x];
  pt = new double[Simulator::dim_p * (Simulator::dv + 1)];

  // arm_type_inverted_pendulum
  x[0] = 3.14159265358979;
  x[1] = 3.14159265358979;
  x[2] = 0.0;
  x[3] = 0.0;

  u[0] = 0.0;
  u[1] = 3.0;
  u[2] = 0.01;

  dxdt[0] = 0.0;
  dxdt[1] = 0.0;
  dxdt[2] = 0.0;
  dxdt[3] = 0.0;

  for (int i = 0; i < Simulator::dv + 1; i++) {
    pt[Simulator::dim_p * i + 0] = 3.14159265358979 / 4.0;
    pt[Simulator::dim_p * i + 1] = 0;
  }

  Cgmres<Model> controller;
  controller.set_ptau(pt);
  controller.init_u0(u);
  controller.init_u0_newton(u, x, pt, 10);

  fp_x = fopen("arm_type_inverted_pendulum_x.txt", "w");
  fp_u = fopen("arm_type_inverted_pendulum_u.txt", "w");

  if (NULL != fp_x && NULL != fp_u) {
    for (int i = 0; i <= (int)(Simulator::t_end / Simulator::dt); i++) {
      // Test code
      t_start = getEtime();
      controller.control(u, x);
      t_end = getEtime();
      t_all += (t_end - t_start);

      // x = x + dxdt * dt
      Simulator::dxdt(dxdt, x, u);
      mul(dxdt, dxdt, Simulator::dt, Simulator::dim_x);
      add(x, x, dxdt, Simulator::dim_x);

      fprintf(fp_x, "%f", Simulator::dt * i);
      fprintf(fp_u, "%f", Simulator::dt * i);
      for (int j = 0; j < Simulator::dim_x; j++) {
        fprintf(fp_x, "\t%f", x[j]);
      }
      for (int j = 0; j < Simulator::dim_u; j++) {
        fprintf(fp_u, "\t%f", u[j]);
      }
      fprintf(fp_x, "\n");
      fprintf(fp_u, "\n");
    }

    fclose(fp_x);
    fclose(fp_u);

    printf("Elapsed time = %f\n", t_all);
  }

  delete[] x;
  delete[] u;
  delete[] dxdt;
  delete[] pt;

  return 0;
}