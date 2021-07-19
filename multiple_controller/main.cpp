#include <stdio.h>
#include <sys/time.h>

#include <iostream>

#include "cgmres.hpp"
#include "matrix.hpp"
#include "model1.hpp"
#include "model2.hpp"
#include "simulator1.hpp"
#include "simulator2.hpp"

double getEtime(void) {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (double)tv.tv_sec + (double)tv.tv_usec * 1e-6;
}

int main(void) {
  //---------------------------
  FILE* fp_x1;
  FILE* fp_u1;
  FILE* fp_x2;
  FILE* fp_u2;

  double t_start, t_end, t_all = 0;

  double* x1;
  double* u1;
  double* dxdt1;
  double* pt1;
  double* x2;
  double* u2;
  double* dxdt2;
  double* pt2;

  x1 = new double[Simulator1::dim_x];
  u1 = new double[Simulator1::dim_u];
  dxdt1 = new double[Simulator1::dim_x];
  pt1 = new double[Simulator1::dim_p * (Simulator1::dv + 1)];
  x2 = new double[Simulator2::dim_x];
  u2 = new double[Simulator2::dim_u];
  dxdt2 = new double[Simulator2::dim_x];
  pt2 = new double[Simulator2::dim_p * (Simulator2::dv + 1)];

  // mass_spring_damper
  x1[0] = 2.0;
  x1[1] = 2.0;
  x1[2] = 0.0;
  x1[3] = 0.0;

  u1[0] = 0.0;
  u1[1] = 0.0;
  u1[2] = 10.0;
  u1[3] = 10.0;
  u1[4] = 5e-4;
  u1[5] = 5e-4;

  dxdt1[0] = 0.0;
  dxdt1[1] = 0.0;
  dxdt1[2] = 0.0;
  dxdt1[3] = 0.0;

  for (int i = 0; i < Simulator1::dv + 1; i++) {
    pt1[Simulator1::dim_p * i + 0] = 1;
    pt1[Simulator1::dim_p * i + 1] = -1;
  }

  // arm_type_inverted_pendulum
  x2[0] = 3.14159265358979;
  x2[1] = 3.14159265358979;
  x2[2] = 0.0;
  x2[3] = 0.0;

  u2[0] = 0.0;
  u2[1] = 3.0;
  u2[2] = 0.01;

  dxdt2[0] = 0.0;
  dxdt2[1] = 0.0;
  dxdt2[2] = 0.0;
  dxdt2[3] = 0.0;

  for (int i = 0; i < Simulator2::dv + 1; i++) {
    pt2[Simulator2::dim_p * i + 0] = 3.14159265358979 / 4.0;
    pt2[Simulator2::dim_p * i + 1] = 0;
  }

  Cgmres<Model1> controller1;
  Cgmres<Model2> controller2;
  controller1.set_ptau(pt1);
  controller2.set_ptau(pt2);
  controller1.init_u0(u1);
  controller2.init_u0(u2);
  controller1.init_u0_newton(u1, x1, pt1, 10);
  controller2.init_u0_newton(u2, x2, pt2, 10);

  fp_x1 = fopen("multiple_controller_x1.txt", "w");
  fp_u1 = fopen("multiple_controller_u1.txt", "w");
  fp_x2 = fopen("multiple_controller_x2.txt", "w");
  fp_u2 = fopen("multiple_controller_u2.txt", "w");

  if (NULL != fp_x1 && NULL != fp_u1 && NULL != fp_x2 && NULL != fp_u2) {
    for (int i = 0; i <= (int)(Simulator1::t_end / Simulator1::dt); i++) {
      // Test code
      t_start = getEtime();
      controller1.control(u1, x1);
      controller2.control(u2, x2);
      t_end = getEtime();
      t_all += (t_end - t_start);

      // x = x + dxdt * dt
      Simulator1::dxdt(dxdt1, x1, u1);
      mul(dxdt1, dxdt1, Simulator1::dt, Simulator1::dim_x);
      add(x1, x1, dxdt1, Simulator1::dim_x);
      Simulator2::dxdt(dxdt2, x2, u2);
      mul(dxdt2, dxdt2, Simulator2::dt, Simulator2::dim_x);
      add(x2, x2, dxdt2, Simulator2::dim_x);

      // File output for model1
      fprintf(fp_x1, "%f", Simulator1::dt * i);
      fprintf(fp_u1, "%f", Simulator1::dt * i);
      for (int j = 0; j < Simulator1::dim_x; j++) {
        fprintf(fp_x1, "\t%f", x1[j]);
      }
      for (int j = 0; j < Simulator1::dim_u; j++) {
        fprintf(fp_u1, "\t%f", u1[j]);
      }
      fprintf(fp_x1, "\n");
      fprintf(fp_u1, "\n");

      // File output for model2
      fprintf(fp_x2, "%f", Simulator2::dt * i);
      fprintf(fp_u2, "%f", Simulator2::dt * i);
      for (int j = 0; j < Simulator2::dim_x; j++) {
        fprintf(fp_x2, "\t%f", x2[j]);
      }
      for (int j = 0; j < Simulator2::dim_u; j++) {
        fprintf(fp_u2, "\t%f", u2[j]);
      }
      fprintf(fp_x2, "\n");
      fprintf(fp_u2, "\n");
    }

    fclose(fp_x1);
    fclose(fp_u1);
    fclose(fp_x2);
    fclose(fp_u2);

    printf("Elapsed time = %f\n", t_all);
  }

  delete[] x1;
  delete[] u1;
  delete[] dxdt1;
  delete[] pt1;
  delete[] x2;
  delete[] u2;
  delete[] dxdt2;
  delete[] pt2;

  return 0;
}