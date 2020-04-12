#pragma once

#include "libs.h"

class Quad
{
public:
	Quad(const int DIM_);
	~Quad() {};

	const int DIM;	// dimension

	// calculate integral of a function in 1D interval [tl, tr] using Gauss-Legendre quadrature rule
	double GL_1D(std::function<double(double)> func, const double tl, const double tr, const int points) const;

	// calculate integral of a function in multiD interval using Gauss-Legendre quadrature rule
	double GL_multiD(std::function<double(std::vector<double>)> func, const std::vector<double> & tl, const std::vector<double> & tr, const int points) const;

	// calculate L1, L2 and Linf norm of multiD function defined in [0, 1]^D
	std::vector<double> norm_multiD(std::function<double(std::vector<double>)> func, const int NMAX, const int points) const;
	std::vector<double> norm_multiD_omp(std::function<double(std::vector<double>)> func, const int NMAX, const int points) const;
	std::vector<double> norm_multiD(std::function<double(std::vector<double>, int i1, int d1)> func, int ii, int dd, const int NMAX, const int points) const;

private:

	// Map storing 1d gauss-legendre quadrature:	
	std::map<int, std::vector<double>> gl_quad_x_1D;	// Key, number of quadrature. Value, node of quadrature	
	std::map<int, std::vector<double>> gl_quad_w_1D;	// Key, number of quadrature. Value, weight of quadrature points
	
	// Map storing multi-dimension gauss-legendre quadrature: 	
	std::map<int, std::vector<std::vector<double>>> gl_quad_x_multiD;	// Key, number of quadrature. Value, node of quadrature	
	std::map<int, std::vector<double>> gl_quad_w_multiD;				// Key, number of quadrature. Value, weight of quadrature

	std::vector<int> gl_quad_num;	// Only limited number of quadrature points are provided. But this can be modified, if add more data files in subfolder quad-data/.

	void read_GL_quad_1D();

	void get_GL_quad_multiD();

	static double linear_map(const double x, const std::vector<double> & xrange, const std::vector<double> & trange);

	static double linear_map(const double x, const double xl, const double xr, const double tl, const double tr);

	static std::vector<double> linear_map(const std::vector<double> & vecx, const std::vector<std::vector<double>> & xrange, const std::vector<std::vector<double>> & trange);

	static std::vector<double> linear_map(const std::vector<double> & vecx, const std::vector<double> & xl, const std::vector<double> & xr, const std::vector<double> & tl, const std::vector<double> & tr);
};
