#include "Basis.h"
#include "subs.h"

Quad Basis::gauss_quad_1D(1);

Basis::Basis(const int level_, const int suppt_, const int dgree_):
	level(level_), suppt(suppt_), dgree(dgree_), supp_interv(2), dis_point(3), lft(3), rgt(3), jmp(3)
{	
	// check that if index is in the correct range
	// if level=0, then suppt=1
	// if level>=1 then suppt=1,3,5,...,2^(level)-1
	assert(suppt % 2 == 1 && "index of support should be odd number");
	assert( ((level == 0) && (suppt == 1))
		|| ( (level >= 1) && ((suppt - 1) / 2 >= 0) && ((suppt - 1) / 2 <= pow_int(2, level - 1) - 1) ) 
		&& "index of support is not in correct range");

	// support of basis function
	// [0, 1] for n = 0,1
	// [2^(-n+1)*j, 2^(-n+1)*(j+1)] for n>=2
	// here j = (suppt-1)/2 and j = 0,1,2,...,2^(level-1)-1
	if (level <= 1)
	{
		supp_interv[0] = 0.;
		supp_interv[1] = 1.;
	}
	else
	{
		supp_interv[0] = std::pow(2., -level + 1.)*(suppt - 1.) / 2.;
		supp_interv[1] = std::pow(2., -level + 1.)*(suppt + 1.) / 2.;
	}

	// discontinuous point of basis function
	dis_point[0] = supp_interv[0];
	dis_point[1] = (supp_interv[0] + supp_interv[1]) / 2;
	dis_point[2] = supp_interv[1];
}

double Basis::product_function(std::function<double(double)> func, const int derivative, const int gauss_quad_num) const
{
	// set intergral interval to be support of Basis
	const double xl = supp_interv[0], xr = supp_interv[1], xmid = (xl + xr) / 2.;

	auto f = [&](const double & x) { return val(x, derivative, 0) * func(x); };

	// for mesh level <= N, we separate [0, 1] into 2^N parts to guarantee accuracy of Gauss quadrature
	// you can take N to be larger, if initial condition has very large gradient
	const int N = 4;
	if (level <= N)
	{
		std::vector<double> x_int = linspace(0., 1., pow_int(2, N)+1);
		double integral = 0.;
		for (size_t i = 0; i < pow_int(2, N); i++)
		{
			integral += gauss_quad_1D.GL_1D(f, x_int[i], x_int[i+1], gauss_quad_num);
		}
		return integral;
	}
	return gauss_quad_1D.GL_1D(f, xl, xmid, gauss_quad_num) + gauss_quad_1D.GL_1D(f, xmid, xr, gauss_quad_num);
}

double Basis::product_volume(const Basis & v, const int derivative_u, const int derivative_v, const int gauss_quad_num) const
{
	// if supports of two functions do not interset, return 0
	if ((v.supp_interv[0] >= supp_interv[1]) || (v.supp_interv[1] <= supp_interv[0])) return 0.;

	// set integral interval to be support of Basis with finer mesh level
	// integral interval is denoted by [xl, xr]
	double xl = supp_interv[0];
	double xr = supp_interv[1];
	if (level <= v.level) 
	{
		xl = v.supp_interv[0];
		xr = v.supp_interv[1];
	}
	const double xmid = (xl + xr) / 2.; // middle point of integral interval
	
	// number of gauss quadrature points, k points is exact for polynomials of degree (2k-1)
	auto f = [&](const double & x) { return val(x, derivative_u) * v.val(x, derivative_v); };

	return gauss_quad_1D.GL_1D(f, xl, xmid, gauss_quad_num) + gauss_quad_1D.GL_1D(f, xmid, xr, gauss_quad_num);
}

double Basis::product_volume_interv(const Basis & v, const Basis & w_interv, const int derivative_u, const int derivative_v, const int gauss_quad_num) const
{
	// if supports of two basis functions do not interset, return 0
	if ((v.supp_interv[0] >= supp_interv[1]) || (v.supp_interv[1] <= supp_interv[0])) return 0.;

	// intervals of w
	double xl_interv = w_interv.supp_interv[0];
	double xr_interv = w_interv.supp_interv[1];

	// if supports of basis functions u or v have not intersection with the support of basis function w we do integral, return 0
	if ((v.supp_interv[0] >= xr_interv) || (v.supp_interv[1] <= xl_interv)) return 0.;
	if ((supp_interv[0] >= xr_interv) || (supp_interv[1] <= xl_interv)) return 0.;

	// set integral interval to be support of Basis with finer mesh level
	// integral interval is denoted by [xl, xr]
	int max_level = std::max(std::max(level, v.level), w_interv.level);
	
	// intervals we do integral, it is taken as the smallest support of u, v and w
	double xl = supp_interv[0];
	double xr = supp_interv[1];
	if (v.level == max_level)
	{
		xl = v.supp_interv[0];
		xr = v.supp_interv[1];
	}
	else if (w_interv.level == max_level)
	{
		xl = xl_interv;
		xr = xr_interv;
	}
	const double xmid = (xl + xr) / 2.; // middle point of integral interval
	
	// number of gauss quadrature points, k points is exact for polynomials of degree (2k-1)
	auto f = [&](const double & x) { return val(x, derivative_u) * v.val(x, derivative_v); };

	return gauss_quad_1D.GL_1D(f, xl, xmid, gauss_quad_num) + gauss_quad_1D.GL_1D(f, xmid, xr, gauss_quad_num);
}

double Basis::product_edge_dis_v(const Basis & v, const int sign_limit_u, const int sign_limit_v, const int derivative_u, const int derivative_v, const std::string & boundary_type) const
{
	double prod_edge = 0.;
	if (boundary_type=="zero")
	{
		// loop over each discontinuous point of basis v
		for (auto const & pt : v.dis_point)
		{
			prod_edge += val(pt, derivative_u, sign_limit_u) * v.val(pt, derivative_v, sign_limit_v);
		}
	}
	else if (boundary_type=="inside")
	{	
		// loop over each discontinuous point of basis v
		for (auto const & pt : v.dis_point)
		{
			// if discontinuous point of v is on the boundary x = 0 or x = 1, then we do not take it into consideration
			if (std::abs(pt-0.)<Const::ROUND_OFF || std::abs(pt-1.)<Const::ROUND_OFF) { continue; }
			prod_edge += val(pt, derivative_u, sign_limit_u) * v.val(pt, derivative_v, sign_limit_v);
		}		
	}	
	else if (boundary_type=="period")
	{
		for (size_t i = 0; i < (v.dis_point).size(); i++) 
		{ 
			// discontinuous points for basis in mesh level 0 and 1 are: 0, 1/2, 1 
			// however for periodic boundary condition, we only consider 0 and 1/2 since 0 and 1 are the same points
			if (v.level<=1 && i==2) continue;
			
			double val_u = val((v.dis_point)[i], derivative_u, sign_limit_u);
			double val_v = v.val((v.dis_point)[i], derivative_v, sign_limit_v);
			// special treatment at boundary point
			// left boundary
			if (std::abs((v.dis_point)[i]-0.)<Const::ROUND_OFF)
			{
				if (sign_limit_u == -1) { val_u = val(1., derivative_u, sign_limit_u); }
				if (sign_limit_v == -1) { val_v = v.val(1., derivative_v, sign_limit_v); }
			}
			// right boundary
			else if (std::abs((v.dis_point)[i]-1.)<Const::ROUND_OFF)
			{
				if (sign_limit_u == 1) { val_u = val(0., derivative_u, sign_limit_u); }
				if (sign_limit_v == 1) { val_v = v.val(0., derivative_v, sign_limit_v); }
			}
			prod_edge += val_u * val_v;
		}
	}
	else
	{
		std::cout << "boundary type not correct in Basis::product_edge_dis_v()" << std::endl; exit(1);
	}
	return prod_edge;
}

double Basis::product_edge_dis_u(const Basis & v, const int sign_limit_u, const int sign_limit_v, const int derivative_u, const int derivative_v, const std::string & boundary_type) const
{
	double prod_edge = 0.;
	if (boundary_type=="zero")
	{
		// loop over each discontinuous point of basis u (current basis)
		for (auto const & pt : dis_point)
		{
			prod_edge += val(pt, derivative_u, sign_limit_u) * v.val(pt, derivative_v, sign_limit_v);
		}
	}
	else if (boundary_type=="inside")
	{	
		// loop over each discontinuous point of basis v
		for (auto const & pt : dis_point)
		{
			// if discontinuous point of v is on the boundary x = 0 or x = 1, then we do not take it into consideration
			if (std::abs(pt-0.)<Const::ROUND_OFF || std::abs(pt-1.)<Const::ROUND_OFF) { continue; }
			prod_edge += val(pt, derivative_u, sign_limit_u) * v.val(pt, derivative_v, sign_limit_v);
		}		
	}	
	else if (boundary_type=="period")
	{
		for (size_t i = 0; i < (dis_point).size(); i++) 
		{ 
			// discontinuous points for basis in mesh level 0 and 1 are: 0, 1/2, 1 
			// however for periodic boundary condition, we only consider 0 and 1/2 since 0 and 1 are the same points
			if (level<=1 && i==2) continue;
			
			double val_u = val(dis_point[i], derivative_u, sign_limit_u);
			double val_v = v.val(dis_point[i], derivative_v, sign_limit_v);
			// special treatment at boundary point
			// left boundary
			if (std::abs(dis_point[i]-0.)<Const::ROUND_OFF)
			{
				if (sign_limit_u == -1) { val_u = val(1., derivative_u, sign_limit_u); }
				if (sign_limit_v == -1) { val_v = v.val(1., derivative_v, sign_limit_v); }
			}
			// right boundary
			else if (std::abs(dis_point[i]-1.)<Const::ROUND_OFF)
			{
				if (sign_limit_u == 1) { val_u = val(0., derivative_u, sign_limit_u); }
				if (sign_limit_v == 1) { val_v = v.val(0., derivative_v, sign_limit_v); }
			}
			prod_edge += val_u * val_v;
		}
	}
	else
	{
		std::cout << "boundary type not correct in Basis::product_edge_dis_u()" << std::endl; exit(1);
	}
	return prod_edge;	
}

double Basis::product_boundary(const Basis & v, const int derivative_u, const int derivative_v, const int boundary_left_right) const
{
	// left boundary x = 0, take the right limit
	if (boundary_left_right == -1)
	{	
		const double x = 0. + Const::ROUND_OFF;
		return val(x, derivative_u, 0) * v.val(x, derivative_v, 0);
	}
	// right boundary x = 1, take the left limit
	else if (boundary_left_right == 1)
	{
		const double x = 1. - Const::ROUND_OFF;
		return val(x, derivative_u, 0) * v.val(x, derivative_v, 0);
	}
	else
	{
		std::cout << "boundary type not correct in Basis::product_boundary()" << std::endl; exit(1);
	}
	return 0.;
}