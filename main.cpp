#include <stdio.h>
#include <sys/time.h>

#include <iostream>

#include "cgmres.hpp"
#include "semiactive_damper/simulator.hpp"

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

  x = Simulator::vector(4);
  x[0] = 2.0;
  x[1] = 2.0;
  x[2] = 0.0;
  x[3] = 0.0;

  u = Simulator::vector(6);
  u[0] = 0.0;
  u[1] = 0.0;
  u[2] = 10.0;
  u[3] = 10.0;
  u[4] = 0.005;
  u[5] = 0.005;

  dxdt = Simulator::vector(4);
  dxdt[0] = 0.0;
  dxdt[1] = 0.0;
  dxdt[2] = 0.0;
  dxdt[3] = 0.0;

  Cgmres controller = Cgmres(u);

  fp_x = fopen("x.txt", "w");
  fp_u = fopen("u.txt", "w");

  if (NULL != fp_x && NULL != fp_u) {
    for (int i = 0; i <= 20000; i++) {
      // Test code
      t_start = getEtime();
      controller.control(x);
      Simulator::mov(u, controller.U_vec, 3);
      t_end = getEtime();
      t_all += (t_end - t_start);

      // x = x + dxdt * dt
      double dt = 0.001;
      Simulator::dxdt(dxdt, x, u);
      Simulator::mul(dxdt, dxdt, dt, 2);
      Simulator::add(x, x, dxdt, 2);

      fprintf(fp_x, "%f\t%f\t%f\n", dt * i, x[0], x[1]);
      fprintf(fp_u, "%f\t%f\t%f\t%f\n", dt * i, u[0], u[1], u[2]);
    }

    fclose(fp_x);
    fclose(fp_u);

    printf("Elapsed time = %f\n", t_all);
  }

  return 0;
}