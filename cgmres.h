#pragma once
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

class Cgmres
{
public:
	void init();

public:
	Cgmres(double* u0);
	~Cgmres(void);

	//------------------------------------------------
	// Number of state
	int16_t dim_x;
	// Number of control input
	int16_t control_input;
	// Number of constraint
	int16_t constraint;
	// Number of dummy variable
	int16_t dummy;
	// Number of variable
	int16_t dim_u;

	// Sampling period (s)
	double dt;
	// Forward difference period (s)
	double h;
	// Stabilize gain for control input (-)
	double zeta;
	// Number of evaluation point (-)
	int16_t dv;
	// Prediction horizon (s)
	double Tf;
	// Rise rate for prediction horizon (-)
	double alpha;
	// Tolerance of convergence
	double tol;
	// Maximum iteration of GMRES
	int16_t k_max;

	// For objective function
	double* xf;
	double* Sf;
	double* Q;
	double* R;

	// For constraints
	double umin;
	double umax;
	double uc;
	double ur;

	// For internal system
	double a;
	double b;

	//------------------------------------------------
	//private:
	// Variables
	double dtau;
	double* dUdt_vec;
	double* U_vec;
	double* x_vec;
	double* dxdt_vec;
	double* lmd_vec;
	double* F_dxh_h;
	double* F_dUh_dxh_h;

	double* b_vec;
	double* v_mat;
	double* h_mat;
	double* rho_e_vec;
	double* g_vec;

	double* x_vec_buf;
	double* U_vec_buf;

	// vector, matrix allocation
	static double* vector(int16_t n);
	static double* matrix(int16_t m, int16_t n);

	// ret = vec, ret = mat
	static void mov(double* ret, const double* vec, const int16_t n);
	static void mov(double* ret, const double* mat, const int16_t m, const int16_t n);

	// ret = vec1 + vec2, ret = mat1 + mat2
	static void add(double* ret, const double* vec1, const double* vec2, const int16_t n);
	static void add(double* ret, const double* mat1, const double* mat2, const int16_t m, const int16_t n);

	// ret = vec1 - vec2, ret = mat1 - mat2
	static void sub(double* ret, const double* vec1, const double* vec2, const int16_t n);
	static void sub(double* ret, const double* mat1, const double* mat2, const int16_t m, const int16_t n);

	// ret = vec * c, ret = mat * c, ret = mat * vec, ret = mat * mat
	static void mul(double* ret, const double* vec, const double c, const int16_t n);
	static void mul(double* ret, const double* mat, const double c, const int16_t m, const int16_t n);
	static void mul(double* ret, const double* mat, const double* vec, const int16_t m, const int16_t n);
	static void mul(double* ret, const double* mat1, const double* mat2, const int16_t l, const int16_t m, const int16_t n);

	// ret = vec / c, ret = mat / c
	static void div(double* ret, const double* vec, const double c, const int16_t n);
	static void div(double* ret, const double* mat, const double c, const int16_t m, const int16_t n);

	// ret = norm(vec)
	static double norm(const double* vec, int16_t n);

	// ret = vec1' * vec2
	static double dot(const double* vec1, const double* vec2, const int16_t n);

	// ret = sign(x)
	static double sign(const double x);

	void dxdt(double* ret, const double* x, const double* u);
	void dPhidx(double* ret, const double* x);
	void dHdx(double* ret, const double* x, const double* u, const double* lmd);
	void dHdu(double* ret, const double* x, const double* u, const double* lmd);
	void ddHduu(double* ret, const double* x, const double* u, const double* lmd);

	void state_equation(const double* x0);
	void adjoint_eqation(void);
	void F_func(double* ret, const double* U_vec, const double* x_vec);
	void gmres();
	void control(double* x);
};