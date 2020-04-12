#include "Quad.h"
#include "subs.h"
#include <omp.h>

Quad::Quad(const int DIM_) : DIM(DIM_)
{
	// Only limited number of quadrature points are provided. But this can be modified, if add more data files in subfolder quad-data/.
	gl_quad_num = std::vector<int>{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

	// read Gauss-Legendre quadrature points and weights in 1D
	read_GL_quad_1D();

	// generate gauss-legendre quadrature points in multi dimension, based on quadrature points in 1D
	get_GL_quad_multiD();
}



/**
 * @brief      Read Gauss-Legendre quadrature points and weights in files
 *
 * @note       The Gauss-Legendre quadrature points and weights data files are input in the quad-data/ subfold
 */
void Quad::read_GL_quad_1D()
{

	for (auto const & num : gl_quad_num) {

		std::vector<double> quad_w(num);
		std::vector<double> quad_x(num);
		if (num==1)
		{
			quad_w = std::vector<double>{ 2 };
			quad_x = std::vector<double>{ 0 };
		}
		else if (num == 2)
		{
			quad_w = std::vector<double>{ 1, 1 };
			quad_x = std::vector<double>{ -0.5773502691896256, 0.5773502691896256 };
		}
		else if (num == 3)
		{
			quad_w = std::vector<double>{ 0.5555555555555556, 0.8888888888888895, 0.5555555555555554 };
			quad_x = std::vector<double>{ -0.7745966692414833, 0, 0.7745966692414832 };
		}
		else if (num == 4)
		{
			quad_w = std::vector<double>{ 0.3478548451374547, 0.6521451548625466, 0.6521451548625458, 0.3478548451374541 };
			quad_x = std::vector<double>{ -0.8611363115940527, -0.3399810435848563, 0.3399810435848563, 0.8611363115940526 };
		}
		else if (num == 5)
		{
			quad_w = std::vector<double>{ 0.2369268850561892, 0.4786286704993669, 0.568888888888889, 0.4786286704993672, 0.2369268850561891 };
			quad_x = std::vector<double>{ -0.9061798459386641, -0.538469310105683, 0, 0.5384693101056831, 0.9061798459386639 };
		}
		else if (num == 6)
		{
			quad_w = std::vector<double>{ 0.1713244923791705, 0.3607615730481384, 0.4679139345726904, 0.467913934572691, 0.3607615730481382, 0.1713244923791708 };
			quad_x = std::vector<double>{ -0.9324695142031522, -0.6612093864662647, -0.238619186083197, 0.2386191860831969, 0.6612093864662647, 0.9324695142031522 };
		}
		else if (num == 7)
		{
			quad_w = std::vector<double>{ 0.1294849661688697, 0.2797053914892765, 0.3818300505051193, 0.4179591836734696, 0.3818300505051192, 0.2797053914892776, 0.1294849661688697 };
			quad_x = std::vector<double>{ -0.9491079123427585, -0.7415311855993943, -0.4058451513773971, 0, 0.4058451513773971, 0.7415311855993943, 0.9491079123427584 };
		}
		else if (num == 8)
		{
			quad_w = std::vector<double>{ 0.101228536290376, 0.2223810344533743, 0.3137066458778873, 0.3626837833783622, 0.3626837833783619, 0.3137066458778869, 0.2223810344533742, 0.1012285362903759 };
			quad_x = std::vector<double>{ -0.9602898564975365, -0.796666477413627, -0.525532409916329, -0.1834346424956498, 0.1834346424956496, 0.5255324099163292, 0.7966664774136268, 0.9602898564975364 };
		}
		else if (num == 9)
		{
			quad_w = std::vector<double>{ 0.08127438836157462, 0.1806481606948576, 0.2606106964029357, 0.3123470770400033, 0.3302393550012602, 0.3123470770400024, 0.2606106964029365, 0.1806481606948582, 0.08127438836157423 };
			quad_x = std::vector<double>{ -0.968160239507626, -0.836031107326636, -0.61337143270059, -0.3242534234038094, 0, 0.3242534234038092, 0.6133714327005907, 0.8360311073266359, 0.9681602395076259 };
		}
		else if (num == 10)
		{
			quad_w = std::vector<double>{ 0.06667134430868846, 0.14945134915058, 0.2190863625159823, 0.2692667193099961, 0.2955242247147523, 0.2955242247147531, 0.2692667193099947, 0.2190863625159819, 0.1494513491505811, 0.06667134430868851 };
			quad_x = std::vector<double>{ -0.9739065285171715, -0.8650633666889844, -0.6794095682990244, -0.4333953941292472, -0.1488743389816311, 0.1488743389816314, 0.4333953941292472, 0.6794095682990243, 0.8650633666889843, 0.9739065285171714 };
		}

		// insert into map
		gl_quad_x_1D.insert(std::make_pair(num, quad_x));
		gl_quad_w_1D.insert(std::make_pair(num, quad_w));
	}
}


/**
 * @brief      Generate Gauss-Legendre quadrature points and weights in multi dimension [-1, 1]^D, based on quadrature points and weights in 1D
 *
 * @param[in]  dim   The dimension
 *
 * @note       The results are inserted into maps: quad_dat::gl_quad_x_multiD and quad_dat::gl_quad_w_multiD
 */
void Quad::get_GL_quad_multiD()
{
	for (auto const & points : gl_quad_num) {

		// pick out quadrature coordinates and weights with given quadrature points
		auto quad_x_1D = gl_quad_x_1D[points];
		auto quad_w_1D = gl_quad_w_1D[points];

		std::vector<std::vector<double>> quad_x_multiD;	// vec of size points^DIM, each element is a vec, (x_1, x_2, ..., x_dim) store quadrature points, x_i for 1<=i<=dim is taken from 1D quadrature points
		std::vector<double> quad_w_multiD;				// vec of size points^DIM, each element is corresponding quadrature weight

		std::vector<std::vector<int>> index;	// generate index array (n_1, n_2, ..., n_dim), each n_i loop from 0 to (points-1)
		IterativeNestedLoop(index, DIM, points);

		for (auto const & ind : index) {

			std::vector<double> vecx;
			double w = 1;
			for (auto const & i : ind) {

				vecx.push_back(quad_x_1D[i]);
				w *= quad_w_1D[i];
			}
			quad_x_multiD.push_back(vecx);
			quad_w_multiD.push_back(w);
		}

		// insert into map
		gl_quad_x_multiD.insert(std::make_pair(points, quad_x_multiD));
		gl_quad_w_multiD.insert(std::make_pair(points, quad_w_multiD));
	}
}



/**
 * @brief      linear map, map each point x in range [ xrange[0], xrange[1] ] to t in range [ trange[0], trange[1] ]
 *
 * @param[in]  x       x in range [ xrange[0], xrange[1] ]
 * @param      xrange  The range of x
 * @param      trange  The range of t
 *
 * @return     t in range [ trange[0], trange[1] ]
 */
double Quad::linear_map(const double x, const std::vector<double> & xrange, const std::vector<double> & trange) {

	double xl = xrange[0], xr = xrange[1];
	double tl = trange[0], tr = trange[1];

	return (x - xl) / (xr - xl)*(tr - tl) + tl;;
}


/**
 * @brief      linear map, map each point x in range [ xl, xr ] to t in range [ tl, tr ]
 *
 * @param[in]  x
 * @param[in]  xl    left end for range x
 * @param[in]  xr    right end for range x
 * @param[in]  tl    left end for range t
 * @param[in]  tr    right end for range t
 *
 * @return     t in range [ tl, tr ]
 *
 * @note       This is a overload version of linear_map( double x, std::vector<double> & xrange, std::vector<double> & trange )
 */
double Quad::linear_map(const double x, const double xl, const double xr, const double tl, const double tr) {

	return (x - xl) / (xr - xl)*(tr - tl) + tl;;
}


/**
 * @brief      linear map, map vector vecx = (x1, x2, ..., xdim), in range [xl_1, xr_1] * [xl_2, xr_2] * ... * [xl_dim, xr_dim], to vect in range [tl_1, tr_1] * [tl_2, tr_2] * ... * [tl_dim, tr_dim]
 *
 * @param      vecx    The vector vecx in range [xl_1, xr_1] * [xl_2, xr_2] * ... * [xl_dim, xr_dim]
 * @param      xrange  The [xl_1, xr_1] * [xl_2, xr_2] * ... * [xl_dim, xr_dim]
 * @param      trange  The [tl_1, tr_1] * [tl_2, tr_2] * ... * [tl_dim, tr_dim]
 *
 * @return     vector vect in range [tl_1, tr_1] * [tl_2, tr_2] * ... * [tl_dim, tr_dim]
 *
 * @note       This is a vector version of linear_map( double x, std::vector<double> & xrange, std::vector<double> & trange )
 */
std::vector<double> Quad::linear_map(const std::vector<double> & vecx, const std::vector<std::vector<double>> & xrange, const std::vector<std::vector<double>> & trange) {

	std::vector<double> vect;
	for (size_t i = 0; i < vecx.size(); ++i) {

		vect.push_back(linear_map(vecx[i], xrange[i], trange[i]));
	}
	return vect;
}


/**
 * @brief      map vector vecx = (x1, x2, ..., xdim) in left range [xl_1, xl_2, ..., xl_dim] and right range [xr_1, xr_2, ..., xr_dim] to vect = (t1, t2, ..., tdim) in left range [tl_1, tl_2, ..., tl_dim] and right range [tr_1, tr_2, ..., tr_dim]
 *
 * @param      vecx  The vecx = (x1, x2, ..., xdim)
 * @param      xl    left range [xl_1, xl_2, ..., xl_dim]
 * @param      xr    right range [xr_1, xr_2, ..., xr_dim]
 * @param      tl    left range [tl_1, tl_2, ..., tl_dim]
 * @param      tr    right range [tr_1, tr_2, ..., tr_dim]
 *
 * @return     vect = (t1, t2, ..., tdim)
 *
 * @note       This is a overload version of linear_map( std::vector<double> & vecx, std::vector<std::vector<double>> & xrange, std::vector<std::vector<double>> & trange ), also a vector version of linear_map( double x, double xl, double xr, double tl, double tr )
 */
std::vector<double> Quad::linear_map(const std::vector<double> & vecx, const std::vector<double> & xl, const std::vector<double> & xr, const std::vector<double> & tl, const std::vector<double> & tr) {

	std::vector<double> vect;
	for (size_t d = 0; d < vecx.size(); ++d) {

		vect.push_back(linear_map(vecx[d], xl[d], xr[d], tl[d], tr[d]));
	}
	return vect;
}


/**
 * @brief      calculate integral of a function in 1D using Gauss-Legendre quadrature rule
 *
 * @param[in]  func    The function
 * @param[in]  tl      left interval end
 * @param[in]  tr      right interval end
 * @param[in]  points  number of quadrature points
 *
 * @return     integral of the function
 */
double Quad::GL_1D(std::function<double(double)> func, const double tl, const double tr, const int points) const
{
	// check if quadrature number is in our data
	assert(std::find(gl_quad_num.begin(), gl_quad_num.end(), points) != gl_quad_num.end() && "quadrature number not correct");

	// pick out quadrature coordinates and weights with given quadrature points
	auto quad_x = gl_quad_x_1D.at(points);
	auto quad_w = gl_quad_w_1D.at(points);

	std::vector<double> xrange = { -1., 1. };	// regular interval [-1, 1] of gauss-legendre quadrature
	std::vector<double> trange = { tl, tr };	// our integral interval, [tl, tr]

	double integral = 0;
	for (int i = 0; i < points; ++i) {

		integral += quad_w[i] * func(linear_map(quad_x[i], xrange, trange));
	}
	integral *= (tr - tl) / 2.;		// normalize

	return integral;
}


/**
 * @brief      calculate \f$ L_1 \f$, \f$ L_2 \f$ and \f$ L_{inf}\f$ norm of piecewise smooth function defined in \f$[0, 1]^D\f$
 *
 * @details    The function are piecewise smooth in every smallest cube of size \f$h^D\f$ with \f$h = 1/2^NMAX\f$. In each small cube, we use the Gauss-Legendre quadrature rule in multi-dimension to calculate the integral.
 *
 * @param[in]  func    The function
 * @param[in]  points  Number of Gauss-Legendre points in each dimension
 *
 * @return     a vector of size 3, storing \f$ L_1 \f$, \f$ L_2 \f$ and \f$ L_{inf}\f$ norm of the function
 *
 * @note       You can also use integral_GL_multiD( std::function<double(std::vector<double>)> func, std::vector<double> & tl, std::vector<double> & tr, int points ) to calculate integral in each small cube and sum them up to
 * 				get the global norm. But transform function in lambda form to functional form will cost some time.
 */

std::vector<double> Quad::norm_multiD(std::function<double(std::vector<double>)> func, const int NMAX, const int points) const
{
	// check if quadrature number is in our data, or quadrature array empty 
	assert(std::find(gl_quad_num.begin(), gl_quad_num.end(), points) != gl_quad_num.end() && "quadrature number not correct");
	assert((!(gl_quad_x_multiD.empty()) && (!(gl_quad_w_multiD.empty()))) && "quadrature array is empty");

	// pick out quadrature coordinates and weights with given quadrature points
	auto quad_x = gl_quad_x_multiD.at(points);
	auto quad_w = gl_quad_w_multiD.at(points);

	const double xl = 0, xr = 1;
	const int N = pow_int(2, NMAX);
	const double h = (xr - xl) / N;

	// generate array storing index, each element of the array is a vec (n_1, n_2, ..., n_(dim)), n_i for 1<=i<=dim loops from 0 to (N-1)
	std::vector<std::vector<int>> index;
	IterativeNestedLoop(index, DIM, N);

	std::vector<double> xl_normal(DIM, -1.);
	std::vector<double> xr_normal(DIM, 1.);

	double L1_norm = 0, L2_norm = 0, Linf_norm = 0;
	omp_set_num_threads(4);
	// divide [0, 1]^D into cube, and calculate integral in each cube by Gauss-Legendre quadrature rule
	for (auto const & ind : index) {
		// left, right, middle point of the cube in each dimension
		std::vector<double> tl(DIM), tr(DIM);
		for (int d = 0; d < DIM; ++d) {

			tl[d] = ind[d] * h;
			tr[d] = (ind[d] + 1)*h;
		}

		// calculate absolute values of function at quadrature points
		std::vector<double> abs_func_val(quad_x.size());
		for (size_t i = 0; i < quad_x.size(); ++i) {

			abs_func_val[i] = std::abs(func(linear_map(quad_x[i], xl_normal, xr_normal, tl, tr)));
		}

		// calculate norm by quadrature rule
		for (size_t i = 0; i < quad_x.size(); ++i) 
		{

			L1_norm += quad_w[i] * abs_func_val[i];
			L2_norm += quad_w[i] * abs_func_val[i] * abs_func_val[i];
			Linf_norm = std::max(Linf_norm, abs_func_val[i]);		
		}

	}
	// normalize
	L1_norm *= pow(h / 2., DIM);
	L2_norm = pow(L2_norm*pow(h / 2., DIM), 0.5);

	std::vector<double> norm = { L1_norm, L2_norm, Linf_norm };
	return norm;
}



std::vector<double> Quad::norm_multiD_omp(std::function<double(std::vector<double>)> func, const int NMAX, const int points) const
{
	// check if quadrature number is in our data, or quadrature array empty 
	assert(std::find(gl_quad_num.begin(), gl_quad_num.end(), points) != gl_quad_num.end() && "quadrature number not correct");
	assert(( !(gl_quad_x_multiD.empty()) && (!(gl_quad_w_multiD.empty())) ) && "quadrature array is empty");

	// pick out quadrature coordinates and weights with given quadrature points
	auto quad_x = gl_quad_x_multiD.at(points);
	auto quad_w = gl_quad_w_multiD.at(points);

	const double xl = 0, xr = 1;
	const int N = pow_int(2, NMAX);
	const double h = (xr - xl) / N;

	// generate array storing index, each element of the array is a vec (n_1, n_2, ..., n_(dim)), n_i for 1<=i<=dim loops from 0 to (N-1)
	std::vector<std::vector<int>> index;
	IterativeNestedLoop(index, DIM, N);

	std::vector<double> xl_normal(DIM, -1.);
	std::vector<double> xr_normal(DIM, 1.);

	double L1_norm = 0, L2_norm = 0, Linf_norm = 0;
	omp_set_num_threads(8);
	// divide [0, 1]^D into cube, and calculate integral in each cube by Gauss-Legendre quadrature rule
# pragma omp parallel default(shared) reduction(+:L1_norm, L2_norm) 
	{
# pragma omp for 
			for (int iter = 0; iter < index.size(); ++iter) {
				auto const & ind = index[iter];
				//std::cout << iter << " " << omp_get_thread_num() << std::endl;
				// left, right, middle point of the cube in each dimension
				std::vector<double> tl(DIM), tr(DIM);
				for (int d = 0; d < DIM; ++d) {

					tl[d] = ind[d] * h;
					tr[d] = (ind[d] + 1)*h;
				}

				// calculate absolute values of function at quadrature points
				std::vector<double> abs_func_val(quad_x.size());
				for (size_t i = 0; i < quad_x.size(); ++i) {

					abs_func_val[i] = std::abs(func(linear_map(quad_x[i], xl_normal, xr_normal, tl, tr)));
				}

				// calculate norm by quadrature rule
				for (size_t i = 0; i < quad_x.size(); ++i) {

					L1_norm += quad_w[i] * abs_func_val[i];
					L2_norm += quad_w[i] * abs_func_val[i] * abs_func_val[i];
					//Linf_norm = std::max(Linf_norm, abs_func_val[i]); // this stupid version of vs does not support reduction of max
				}

			}
	}
	
	L1_norm *= pow(h / 2., DIM);
	L2_norm = pow(L2_norm*pow(h / 2., DIM), 0.5);

	std::vector<double> norm = { L1_norm, L2_norm, Linf_norm };
	return norm;
}


std::vector<double> Quad::norm_multiD(std::function<double(std::vector<double>, int i1, int d1)> func, int ii, int dd, const int NMAX, const int points) const
{
	// check if quadrature number is in our data, or quadrature array empty 
	assert(std::find(gl_quad_num.begin(), gl_quad_num.end(), points) != gl_quad_num.end() && "quadrature number not correct");
	assert(( !(gl_quad_x_multiD.empty()) && (!(gl_quad_w_multiD.empty())) ) && "quadrature array is empty");

	// pick out quadrature coordinates and weights with given quadrature points
	auto quad_x = gl_quad_x_multiD.at(points);
	auto quad_w = gl_quad_w_multiD.at(points);

	double xl = 0, xr = 1;
	int N = pow_int(2, NMAX);
	double h = (xr - xl) / N;

	// generate array storing index, each element of the array is a vec (n_1, n_2, ..., n_(dim)), n_i for 1<=i<=dim loops from 0 to (N-1)
	std::vector<std::vector<int>> index;
	IterativeNestedLoop(index, DIM, N);

	std::vector<double> xl_normal(DIM, -1.);
	std::vector<double> xr_normal(DIM, 1.);

	double L1_norm = 0, L2_norm = 0, Linf_norm = 0;
	// divide [0, 1]^D into cube, and calculate integral in each cube by Gauss-Legendre quadrature rule
	for (auto const & ind : index) {

		// left, right, middle point of the cube in each dimension
		std::vector<double> tl(DIM), tr(DIM);
		for (int d = 0; d < DIM; ++d) {

			tl[d] = ind[d] * h;
			tr[d] = (ind[d] + 1)*h;
		}

		// calculate absolute values of function at quadrature points
		std::vector<double> abs_func_val(quad_x.size());
		for (size_t i = 0; i < quad_x.size(); ++i) {

			abs_func_val[i] = std::abs(func(linear_map(quad_x[i], xl_normal, xr_normal, tl, tr), ii , dd));
		}

		// calculate norm by quadrature rule
		for (size_t i = 0; i < quad_x.size(); ++i) {

			L1_norm += quad_w[i] * abs_func_val[i];
			L2_norm += quad_w[i] * abs_func_val[i] * abs_func_val[i];
			Linf_norm = std::max(Linf_norm, abs_func_val[i]);
		}
	}
	// normalize
	L1_norm *= pow(h / 2., DIM);
	L2_norm = pow(L2_norm*pow(h / 2., DIM), 0.5);

	std::vector<double> norm = { L1_norm, L2_norm, Linf_norm };
	return norm;
}

/**
 * @brief      calculate integeral of a given smooth function in hypercubes (multi-dimension), using Gauss-Legendre rule
 *
 * @param[in]  func    The multi-variable function
 * @param      tl      integration points in left end, tl = { tl_1, tl_2, ..., tl_dim }
 * @param      tr      integration points in right end, tr = { tr_1, tr_2, ..., tr_dim }
 * @param[in]  points  The number of quadrature points in each dimension
 *
 * @return     integral of the function
 *
 * @warning    The given function should be smooth in the given cube, otherwise the integration is not accurate
 */
double Quad::GL_multiD(std::function<double(std::vector<double>)> func, const std::vector<double> & tl, const std::vector<double> & tr, const int points) const
{
	assert(tl.size()==DIM && tr.size()==DIM);

	// check if quadrature number is in our data
	assert(std::find(gl_quad_num.begin(), gl_quad_num.end(), points) != gl_quad_num.end() && "quadrature number not correct");

	// dimension for our integral cube
	const int dim = tl.size();

	// pick out quadrature coordinates and weights with given quadrature points
	auto quad_x = gl_quad_x_multiD.at(points);
	auto quad_w = gl_quad_w_multiD.at(points);

	std::vector<double> xl(dim, -1.), xr(dim, 1.);		// regular interval [-1, 1] of gauss-legendre quadrature

	// calculate volume of cube
	double vol = 1;
	for (int d = 0; d < dim; ++d) {

		vol *= (tr[d] - tl[d]);	// length of the cube in d-th dimension
	}

	double integral = 0;
	for (size_t i = 0; i < quad_x.size(); ++i) {

		integral += quad_w[i] * func(linear_map(quad_x[i], xl, xr, tl, tr));
	}
	integral *= vol / pow(2, dim);		// normalize
	return integral;
}