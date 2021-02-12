#include <stdio.h>
#include "cgmres.h"

#include <iostream>

void print(double* vec, uint16_t n)
{
	int i;
	for (i = 0; i < n; i++)
	{
		printf("%10.3f\n", vec[i]);
	}
	printf("\n");
}

void print(double* vec, uint16_t m, uint16_t n)
{
	int i, j, index;
	for (j = 0; j < m; j++)
	{
		for (i = 0; i < n; i++)
		{
			index = m * i + j;
			printf("%10.3f%s", vec[index], ((i == n - 1) ? "\n" : " "));
		}
	}
	printf("\n");
}

int main(void)
{

	//---------------------------
	FILE* fp_x;
	FILE* fp_u;
	errno_t err_x, err_u;

	double* x;
	double* u;
	double* dxdt;

	x = Cgmres::vector(2);
	x[0] = 2.0;
	x[1] = 0.0;

	u = Cgmres::vector(3);
	u[0] = 0.028393761456739753656908220592;
	u[1] = 0.166095020295846107494242005487;
	u[2] = 0.030103250483332195247543339178;

	dxdt = Cgmres::vector(2);
	dxdt[0] = 0.0;
	dxdt[1] = 0.0;

	Cgmres controller = Cgmres(u);

	fopen_s(&fp_x, "x.txt", "w");
	fopen_s(&fp_u, "u.txt", "w");

	if (NULL != fp_x && NULL != fp_u)
	{
		for (int i = 0; i <= 20000; i++)
		{
			if (i == 948)
			{
				printf("");
			}

			// Test code
			controller.control(x);
			Cgmres::mov(u, controller.U_vec, 3);

			// x = x + dxdt * dt
			double dt = 0.001;
			controller.dxdt(dxdt, x, u);
			Cgmres::mul(dxdt, dxdt, dt, 2);
			Cgmres::add(x, x, dxdt, 2);

			fprintf(fp_x, "%f\t%f\t%f\n", dt * i, x[0], x[1]);
			fprintf(fp_u, "%f\t%f\t%f\t%f\n", dt * i, u[0], u[1], u[2]);
		}

		fclose(fp_x);
		fclose(fp_u);
	}

	return 0;
}