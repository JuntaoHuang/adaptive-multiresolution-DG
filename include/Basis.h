#pragma once

#include "libs.h"
#include "Quad.h"


// the underlying hierachical basis class 1D, which can derive three basis classes LagrBasis, HermBasis, AlptBasis
// the fundamental variables in the basis class is level 'n', suppt 'j', and dgree 'p';
class Basis
{
public:
	Basis(const int level_, const int suppt_, const int dgree_);
	~Basis() {};

	const int level; 	// level of mesh, 0, 1, 2, ..., NMAX
	const int suppt; 	// index denoting difference support of basis. For n=0,1, j=1; For n>=2, j = 1, 3, ..., 2^n-1.
	const int dgree; 	// degree of polynomial, 0, 1, 2, ..., PMAX

	std::vector<double> supp_interv;	// support of this basis, supp[0] = xleft, supp[1] = xright
	std::vector<double> dis_point;		// discontinuous point, xleft, xmid, xright

	// left and right limit at discontinuous point.
	// left limit at left boundary x = 0 and right limit at right boundary x = 1 is defined as 0
	// jump of function at discontinuous point, [f(x)]=f(x^+) - f(x^-)
	std::vector<double> lft, rgt, jmp;

	// return value and left/right limits of this basis function (or its derivative) at point x
	// note that the basis is supported on [0, 1], i.e., for x < 0 or x > 1, the function will return 0
	// sgn_lim:	-1: left limit; 0: value at this point; 1: right limit
	virtual double val(const double x, const int derivative = 0, const int sgn_lim = 0) const = 0;

	// product of basis function and a given function (or its derivative) in 1D
	double product_function(std::function<double(double)> func, const int derivative = 0, const int gauss_quad_num = 10) const;

	// product (in volume) of basis function (or derivatives) of u and basis function (or derivatives) of v
	double product_volume(const Basis & v, const int derivative_u = 0, const int derivative_v = 0, const int gauss_quad_num = 10) const;

	// product (in volume) on a given interval (support of some basis function) of basis function (or derivatives) of u and basis function (or derivatives) of v
	// e.g. calculate \int_{support_of_w} u * v, integral of product of two basis functions u and v on an interval which is the support of basis function w
	// we will be use this function to calculate artificial viscosity terms for capturing shocks
	double product_volume_interv(const Basis & v, const Basis & w_interv, const int derivative_u = 0, const int derivative_v = 0, const int gauss_quad_num = 10) const;
	
	// product (in edge interface) of basis function (or derivatives) of u and basis function (or derivatives) of v
	// i.e. u^+ * v^- evaluated at all discontinuous points of v
	// sign_limit_u: 
	// 		-1: left limit of u
	// 		1: right limit of u
	// sign_limit_v: 
	// 		-1: left limit of v
	// 		1: right limit of v
	// boundary_type: 
	// 		period: basis functions outside the domain [0, 1] is defined periodic, i.e., u(0-) = u(1-), u(1+) = u(0+)
	// 		zero: basis functions outside the domain [0, 1] is defined to be zero, i.e., u(0-) = u(1+) = 0
	// 		inside: only consider discontinuous point of v which is inside the domain (excludes boundary point 0 and 1)
	double product_edge_dis_v(const Basis & v, const int sign_limit_u, const int sign_limit_v, const int derivative_u, const int derivative_v, const std::string & boundary_type) const;

	// product (in edge interface) of basis function (or derivatives) of u and basis function (or derivatives) of v
	// evaluated at all discontinuous points of u (current basis function)
	double product_edge_dis_u(const Basis & v, const int sign_limit_u, const int sign_limit_v, const int derivative_u, const int derivative_v, const std::string & boundary_type) const;

	// product at boundary of basis function (or derivatives) of u and basis function (or derivatives) of v
	// here value of u and v are taken in the inner side of boundary
	// boundary_left_right:
	// 		-1: left boundary x = 0
	// 		1: right boundary x = 1
	double product_boundary(const Basis & v, const int derivative_u, const int derivative_v, const int boundary_left_right) const;
	
private:

	// calculate numerical quadrature in 1D
	static Quad gauss_quad_1D;
};