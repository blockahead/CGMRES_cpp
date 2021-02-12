#include "cgmres.h"

#define DEBUG_MODE

Cgmres::Cgmres(double* u0)
{
	//---------------------------------------
	dim_x = 2;
	control_input = 1;
	constraint = 1;
	dummy = 1;
	dim_u = control_input + constraint + dummy;

	dt = 0.001;
	h = 0.002;
	zeta = 1000.0;
	dv = 50;
	Tf = 1.0;
	alpha = 0.5;
	tol = 1e-6;
	k_max = 5;

	xf = Cgmres::vector(2);
	Sf = Cgmres::vector(2);
	Q = Cgmres::vector(2);
	R = Cgmres::vector(2);

	xf[0] = 0;
	xf[1] = 0;
	Sf[0] = 1;
	Sf[1] = 10;
	Q[0] = 1;
	Q[1] = 10;
	R[0] = 1;
	R[1] = 0.01;

	umin = 0.0;
	umax = 1.0;
	uc = (umax + umin) / 2.0;
	ur = (umax - umin) / 2.0;

	a = -1;
	b = -1;
	//---------------------------------------

	dtau = 0.0;
	dUdt_vec = Cgmres::vector(dim_u * dv);
	U_vec = Cgmres::vector(dim_u * dv);
	x_vec = Cgmres::vector(dim_x * (dv + 1));
	dxdt_vec = Cgmres::vector(dim_x * (dv + 1));
	lmd_vec = Cgmres::vector(dim_x * (dv + 1));
	F_dxh_h = Cgmres::vector(dim_u * dv);
	F_dUh_dxh_h = Cgmres::vector(dim_u * dv);
	b_vec = Cgmres::vector(dim_u * dv);

	v_mat = Cgmres::matrix(dim_u * dv, k_max + 1);
	h_mat = Cgmres::matrix(k_max + 1, k_max + 1);
	rho_e_vec = Cgmres::vector(k_max + 1);
	g_vec = Cgmres::matrix(3, k_max);

	x_vec_buf = Cgmres::vector(dim_x * (dv + 1));
	U_vec_buf = Cgmres::vector(dim_u * dv);

	//---------------------------------------
	for (int16_t i = 0; i < dv; i++)
	{
		int16_t index = dim_u * i;
		Cgmres::mov(&U_vec[index], u0, dim_u);
	}
	//---------------------------------------
}

Cgmres::~Cgmres(void)
{
	free(dUdt_vec);
	free(U_vec);
	free(x_vec);
	free(dxdt_vec);
	free(lmd_vec);
	free(F_dxh_h);
	free(F_dUh_dxh_h);
	free(b_vec);

	free(v_mat);
	free(h_mat);
	free(rho_e_vec);
	free(g_vec);

	free(x_vec_buf);
	free(U_vec_buf);
}

double* Cgmres::vector(int16_t n)
{
	int16_t i;
	double* ret;
	ret = (double*)malloc(sizeof(double) * n);
	if (NULL == ret)
	{
		printf("Vector malloc() failure.");
	}
	else
	{
		for (i = 0; i < n; i++)
		{
			*(ret + i) = 0.0;
		}
	}
	return ret;
}

double* Cgmres::matrix(int16_t m, int16_t n)
{
	int16_t i;
	double* ret;
	ret = (double*)malloc(sizeof(double) * m * n);
	if (NULL == ret)
	{
		printf("Matrix malloc() failure.");
	}
	else
	{
		for (i = 0; i < m * n; i++)
		{
			*(ret + i) = 0.0;
		}
	}

	return ret;
}

void Cgmres::mov(double* ret, const double* vec, const int16_t n)
{
	int16_t i;
	for (i = 0; i < n; i++)
	{
		ret[i] = vec[i];
	}
}

void Cgmres::mov(double* ret, const double* mat, const int16_t m, const int16_t n)
{
	int16_t i, j, index;
	for (i = 0; i < m; i++)
	{
		for (j = 0; j < n; j++)
		{
			index = n * i + j;
			ret[index] = mat[index];
		}
	}
}

void Cgmres::add(double* ret, const double* vec1, const double* vec2, const int16_t n)
{
	int16_t i;
	for (i = 0; i < n; i++)
	{
		ret[i] = vec1[i] + vec2[i];
	}
}

void Cgmres::add(double* ret, const double* mat1, const double* mat2, const int16_t m, const int16_t n)
{
	int16_t i, j, index;
	for (i = 0; i < m; i++)
	{
		for (j = 0; j < n; j++)
		{
			index = n * i + j;
			ret[index] = mat1[index] + mat2[index];
		}
	}
}

void Cgmres::sub(double* ret, const double* vec1, const double* vec2, const int16_t n)
{
	int16_t i;
	for (i = 0; i < n; i++)
	{
		ret[i] = vec1[i] - vec2[i];
	}
}

void Cgmres::sub(double* ret, const double* mat1, const double* mat2, const int16_t m, const int16_t n)
{
	int16_t i, j, index;
	for (i = 0; i < m; i++)
	{
		for (j = 0; j < n; j++)
		{
			index = n * i + j;
			ret[index] = mat1[index] - mat2[index];
		}
	}
}

void Cgmres::mul(double* ret, const double* vec, const double c, const int16_t n)
{
	int16_t i;
	for (i = 0; i < n; i++)
	{
		ret[i] = vec[i] * c;
	}
}

void Cgmres::mul(double* ret, const double* mat, const double c, const int16_t m, const int16_t n)
{
	int16_t i, j, index;
	for (i = 0; i < m; i++)
	{
		for (j = 0; j < n; j++)
		{
			index = n * i + j;
			ret[index] = mat[index] * c;
		}
	}
}

void Cgmres::mul(double* ret, const double* mat, const double* vec, const int16_t m, const int16_t n)
{
	int16_t i, j, index;
#ifdef DEBUG_MODE
	if (ret == vec)
	{
		printf("%s pointer error !\n", __func__);
		exit(-1);
	}
#endif
	for (i = 0; i < m; i++)
	{
		ret[i] = 0.0;
	}

	for (j = 0; j < n; j++)
	{
		for (i = 0; i < m; i++)
		{
			index = m * j + i;
			ret[i] += mat[index] * vec[j];
		}
	}
}

void Cgmres::mul(double* ret, const double* mat1, const double* mat2, const int16_t l, const int16_t m, const int16_t n)
{
	int16_t i, j, k, index1, index2, index3;
#ifdef DEBUG_MODE
	if (ret == mat1 || ret == mat2)
	{
		printf("%s pointer error !\n", __func__);
		exit(-1);
	}
#endif

	for (i = 0; i < m; i++)
	{
		for (j = 0; j < n; j++)
		{
			index1 = n * i + j;
			ret[index1] = 0;
		}

		for (k = 0; k < l; k++)
		{
			index2 = n * i + k;
			for (j = 0; j < n; j++)
			{
				index1 = n * i + j;
				index3 = n * k + j;
				ret[index1] += mat1[index2] * mat2[index3];
			}
		}
	}
}

void Cgmres::div(double* ret, const double* vec, const double c, const int16_t n)
{
	int16_t i;
	double inv_c = 1.0 / c;
	for (i = 0; i < n; i++)
	{
		ret[i] = vec[i] * inv_c;
	}
}

void Cgmres::div(double* ret, const double* mat, const double c, const int16_t m, const int16_t n)
{
	int16_t i, j, index;
	double inv_c = 1.0 / c;
	for (i = 0; i < m; i++)
	{
		for (j = 0; j < n; j++)
		{
			index = n * i + j;
			ret[index] = mat[index] * inv_c;
		}
	}
}

double Cgmres::norm(const double* vec, int16_t n)
{
	int16_t i;
	double ret = 0;
	for (i = 0; i < n; i++)
	{
		ret += vec[i] * vec[i];
	}

	return sqrt(ret);
}

double Cgmres::dot(const double* vec1, const double* vec2, const int16_t n)
{
	int16_t i;
	double ret = 0;
	for (i = 0; i < n; i++)
	{
		ret += vec1[i] * vec2[i];
	}

	return ret;
}

double Cgmres::sign(const double x)
{
	return (x < 0) ? -1 : 1;
}

void Cgmres::dxdt(double* ret, const double* x, const double* u)
{
	ret[0] = x[1];
	ret[1] = a * x[0] + b * u[0] * x[1];
}

void Cgmres::dPhidx(double* ret, const double* x)
{
	ret[0] = x[0] * Sf[0];
	ret[1] = x[1] * Sf[1];
}

void Cgmres::dHdx(double* ret, const double* x, const double* u, const double* lmd)
{
	ret[0] = x[0] * Q[0] + a * lmd[1];
	ret[1] = x[1] * Q[1] + lmd[0] + b * u[0] * lmd[1];
}

void Cgmres::dHdu(double* ret, const double* x, const double* u, const double* lmd)
{
	ret[0] = R[0] * u[0] + b * x[1] * lmd[1] + 2 * u[2] * (u[0] - uc);
	ret[1] = -R[1] + 2 * u[1] * u[2];
	ret[2] = (u[0] - uc) * (u[0] - uc) + u[1] * u[1] - ur * ur;
}

void Cgmres::ddHduu(double* ret, const double* x, const double* u, const double* lmd)
{
	ret[0] = R[0] + 2 * u[2];
	ret[1] = 0;
	ret[2] = 2 * (u[0] - uc);

	ret[3] = 0;
	ret[4] = 2 * u[2];
	ret[5] = 2 * u[1];

	ret[6] = 2 * (u[0] - uc);
	ret[7] = 2 * u[1];
	ret[8] = 0;
}

void Cgmres::state_equation(const double* x0)
{
	int16_t i, index_x, index_u;
	Cgmres::mov(x_vec, x0, dim_x);
	for (i = 0; i < dv; i++)
	{
		index_x = dim_x * i;
		index_u = dim_u * i;
		dxdt(&dxdt_vec[index_x], &x_vec[index_x], &U_vec[index_u]);
		Cgmres::mul(x_vec_buf, &dxdt_vec[index_x], dtau, dim_x);
		Cgmres::add(&x_vec[index_x + dim_x], &x_vec[index_x], x_vec_buf, dim_x);
	}
}

void Cgmres::adjoint_eqation(void)
{
	int16_t i, index_x, index_u;
	dPhidx(&lmd_vec[dim_x * dv], &x_vec[dim_x * dv]);
	for (i = dv - 1; i >= 0; i--)
	{
		index_x = dim_x * i;
		index_u = dim_u * i;
		dHdx(x_vec_buf, &x_vec[index_x], &U_vec[index_u], &lmd_vec[index_x + dim_x]);
		Cgmres::mul(x_vec_buf, x_vec_buf, dtau, dim_x);
		Cgmres::add(&lmd_vec[index_x], &lmd_vec[index_x + dim_x], x_vec_buf, dim_x);
	}
}

void Cgmres::F_func(double* ret, const double* U_vec_tmp, const double* x_vec_tmp)
{
	int16_t i, index_x, index_u;

#ifdef DEBUG_MODE
	if (ret == U_vec_tmp)
	{
		printf("%s pointer error ! (U_vec_tmp is overwritten due to the same address of ret)\n", __func__);
		exit(-1);
	}
#endif

	for (i = 0; i < dv; i++)
	{
		index_x = dim_x * i;
		index_u = dim_u * i;
		dHdu(&ret[index_u], &x_vec_tmp[index_x], &U_vec_tmp[index_u], &lmd_vec[index_x]);
	}
}

void Cgmres::gmres()
{
	int16_t len, i, j, k, index_v1, index_v2, index_h, index_g;
	double buf;

	// x + dxdt * h
	// len = dim_x * (dv + 1);
	// Cgmres::mul(x_vec_buf, dxdt_vec, h, len);
	// Cgmres::add(x_vec_buf, x_vec, x_vec_buf, len);

	// F_dUh_dxh_h = F(U + dUdt * h, x + dxdt * h)
	len = dim_u * dv;
	Cgmres::mul(U_vec_buf, dUdt_vec, h, len);
	Cgmres::add(U_vec_buf, U_vec, U_vec_buf, len);
	F_func(F_dUh_dxh_h, U_vec_buf, x_vec_buf);

	// Ax = (F(U + dUdt * h, x + dxdt * h) - F(U, x + dxdt * h)) / h
	Cgmres::sub(U_vec_buf, F_dUh_dxh_h, F_dxh_h, len);
	Cgmres::div(U_vec_buf, U_vec_buf, h, len);

	// r0 = b - Ax
	Cgmres::sub(U_vec_buf, b_vec, U_vec_buf, len);

	// rho = sqrt(r0' * r0)
	rho_e_vec[0] = Cgmres::norm(U_vec_buf, len);

	if (rho_e_vec[0] < tol)
	{
		k = 0;
		return;
	}

	// v_mat[:][0] = r0 / rho
	Cgmres::div(&v_mat[0], U_vec_buf, rho_e_vec[0], len);

	for (k = 0; k < k_max; k++)
	{
		// F_dUh_dxh_h = F(U + v[k] * h, x + dxdt * h)
		index_v1 = len * k;
		Cgmres::mul(U_vec_buf, &v_mat[index_v1], h, len);
		Cgmres::add(U_vec_buf, U_vec, U_vec_buf, len);
		F_func(F_dUh_dxh_h, U_vec_buf, x_vec_buf);

		// v[k+1] = (F(U + v[k] * h, x + dxdt * h) - F(U, x + dxdt * h)) / h
		index_v1 = len * (k + 1);
		Cgmres::sub(U_vec_buf, F_dUh_dxh_h, F_dxh_h, len);
		Cgmres::div(&v_mat[index_v1], U_vec_buf, h, len);

		// Modified Gram-Schmidt
		for (i = 0; i < k + 1; i++)
		{
			index_v2 = len * i;
			index_h = (k_max + 1) * k + i;
			h_mat[index_h] = Cgmres::dot(&v_mat[index_v2], &v_mat[index_v1], len);
			Cgmres::mul(U_vec_buf, &v_mat[index_v2], h_mat[index_h], len);
			Cgmres::sub(&v_mat[index_v1], &v_mat[index_v1], U_vec_buf, len);
		}
		index_h = (k_max + 1) * k + (k + 1);
		h_mat[index_h] = Cgmres::norm(&v_mat[index_v1], len);

		// Check breakdown
		if (fabs(h_mat[index_h]) < DBL_EPSILON)
		{
			printf("Breakdown\n");
			return;
		}
		else
		{
			Cgmres::div(&v_mat[index_v1], &v_mat[index_v1], h_mat[index_h], len);
		}

		// Transformation h_mat to upper triangular matrix with Householder transformation
		for (i = 0; i < k; i++)
		{
			index_h = (k_max + 1) * k + i;
			index_g = 3 * i;
			buf = (g_vec[index_g + 0] * h_mat[index_h + 0] + g_vec[index_g + 1] * h_mat[index_h + 1]) * g_vec[index_g + 2];
			h_mat[index_h + 0] = h_mat[index_h + 0] - buf * g_vec[index_g + 0];
			h_mat[index_h + 1] = h_mat[index_h + 1] - buf * g_vec[index_g + 1];
		}
		index_h = (k_max + 1) * k + k;
		index_g = 3 * k;
		buf = -Cgmres::sign(h_mat[index_h]) * Cgmres::norm(&h_mat[index_h], 2); // Vector length
		g_vec[index_g + 0] = h_mat[index_h + 0] - buf;
		g_vec[index_g + 1] = h_mat[index_h + 1];
		g_vec[index_g + 2] = 2.0 / Cgmres::dot(&g_vec[index_g], &g_vec[index_g], 2);
		h_mat[index_h + 0] = buf;
		h_mat[index_h + 1] = 0.0;

		// Update residual
		buf = g_vec[index_g + 0] * rho_e_vec[k + 0] * g_vec[index_g + 2];
		rho_e_vec[k + 0] = rho_e_vec[k + 0] - buf * g_vec[index_g + 0];
		rho_e_vec[k + 1] = -buf * g_vec[index_g + 1];

		//Check convergence
		if (fabs(rho_e_vec[k + 1]) < tol)
		{
			break;
		}
	}

	// Solve h_mat * y = rho_e_vec
	// h_mat is upper triangle matrix
	for (i = k - 1; i >= 0; i--)
	{
		for (j = k - 1; j > i; j--)
		{
			index_h = (k_max + 1) * j + i;
			rho_e_vec[i] -= h_mat[index_h] * rho_e_vec[j];
		}
		index_h = (k_max + 1) * i + i;
		rho_e_vec[i] /= h_mat[index_h];
	}

	// dUdt = dUdt + v_mat * y
	len = dim_u * dv;
	Cgmres::mul(U_vec_buf, v_mat, rho_e_vec, len, k);
	Cgmres::add(dUdt_vec, dUdt_vec, U_vec_buf, len);
}

void Cgmres::control(double* x)
{
	int16_t len;

	state_equation(x);
	adjoint_eqation();

	// F_dxh_h = F(U, x + dxdt * h)
	len = dim_x * (dv + 1);
	Cgmres::mul(x_vec_buf, dxdt_vec, h, len);
	Cgmres::add(x_vec_buf, x_vec, x_vec_buf, len);
	F_func(F_dxh_h, U_vec, x_vec_buf);

	// b = ( ( 1 - zeta * h ) * F(U, x) - F(U, x + dxdt * h) ) / h;
	len = dim_u * dv;
	F_func(b_vec, U_vec, x_vec);
	Cgmres::mul(b_vec, b_vec, 1 - zeta * h, len);
	Cgmres::sub(b_vec, b_vec, F_dxh_h, len);
	Cgmres::div(b_vec, b_vec, h, len);

	// GMRES
	gmres();

	// U = U + dUdt * dt;
	// len = dim_u * dv;
	Cgmres::mul(U_vec_buf, dUdt_vec, dt, len);
	Cgmres::add(U_vec, U_vec, U_vec_buf, len);

	dtau = (1 - alpha * dt) * dtau + alpha * dt * Tf / (double)dv;
}