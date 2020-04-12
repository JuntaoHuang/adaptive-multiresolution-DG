#include "AlptBasis.h"

int AlptBasis::PMAX;

AlptBasis::AlptBasis(const int level_, const int suppt_, const int dgree_) : 
	Basis(level_, suppt_, dgree_)
{
	for (size_t i = 0; i < 3; i++)
	{
		lft[i] = val(dis_point[i], 0, -1);
		rgt[i] = val(dis_point[i], 0, 1);
		jmp[i] = rgt[i] - lft[i];
	}
}

double AlptBasis::val(const double x, const int derivative, const int sgn) const
{
	double xlim = x;
	if (sgn == -1)
	{
		if (std::abs(x) <= Const::ROUND_OFF) return 0.;
		xlim -= Const::ROUND_OFF;
	}
	else if (sgn == 1)
	{
		if (std::abs(x-1) <= Const::ROUND_OFF) return 0.;
		xlim += Const::ROUND_OFF;
	}

	if (derivative == 0)
	{
		return val0(xlim, level, suppt, dgree);
	}
	else if (derivative == 1)
	{
		return val1(xlim, level, suppt, dgree);
	}
	else if (derivative == 2)
	{
		return val2(xlim, level, suppt, dgree);
	}
	else if (derivative == 3)
	{
		return val3(xlim, level, suppt, dgree);
	}
	else if (derivative == 4)
	{
		return val4(xlim, level, suppt, dgree);
	}
	else
	{
		std::cout << "derivative degree is too large in definition of val" << std::endl;
		exit(1);
	}
}

double AlptBasis::val0(const double x, const int n, const int j, const int p)
{
	if (n == 0)
	{
		// mesh level 0: normalized legendre polynomial on [0, 1]
		// v_{i,0}(x) = P_i(2x-1) * sqrt(2i+1), where P_i(x) is Legendre polynomial on [-1, 1]
		return legendre_poly(p, 2 * x - 1) * sqrt(2 * p + 1.);
	}
	else if (n == 1)
	{
		// mesh level 1: 
		// v_{i,1}(x) = h_i(x) = 2^(1/2)*f_i(2x-1)
		// here f_i(x) is Alpert's wavelets
		if (x >= 0 && x <= 1) return sqrt(2.) * phi(2 * x - 1, PMAX, p);
		else return 0.;
	}
	else
	{
		// mesh level >=2: 
		// v_{i,n}^j(x) = 2^{(n-1)/2} * h_i( 2^(n-1)x - j )
		const int j_odd = (j - 1) / 2;
		double shift_x = pow(2, n - 1)*x - j_odd;
		return pow(2, (n - 1) / 2.) * val0(shift_x, 1, 0, p);
	}
}

double AlptBasis::val1(const double x, const int n, const int j, const int p)
{
	// Alpert's wavelets
	if (n == 0) {

		// mesh level 0: normalized legendre polynomial on [0, 1]
		// v_{i,0}(x) = P_i(2x-1) * sqrt(2i+1), where P_i(x) is Legendre polynomial on [-1, 1]
		// (d/dx)v_{i,0}(x) = 2*(d/dx)P_i(2x-1) * sqrt(2i+1), where P_i(x) is Legendre polynomial on [-1, 1]
		return legendre_poly_prime(p, 2 * x - 1) * 2 * sqrt(2 * p + 1.);
	}
	else if (n == 1) {

		// mesh level 1: 
		// v_{i,1}(x) = h_i(x) = 2^(1/2)*f_i(2x-1)
		// here f_i(x) is Alpert's wavelets
		// (d/dx)v_{i,1}(x) = 2^(3/2)*(d/dx)f_i(2x-1)
		if (x >= 0 && x <= 1) return pow(2, 1.5) * phi(2 * x - 1, PMAX, p, 1);
		else return 0;
	}
	else {

		// mesh level >=2: 
		// v_{i,n}^j(x) = 2^{(n-1)/2} * h_i( 2^(n-1)x - j )
		// (d/dx)v_{i,n}^j(x) = 2^{3(n-1)/2} * h_i( 2^(n-1)x - j )
		const int j_odd = (j - 1) / 2;
		double shift_x = pow(2, n - 1)*x - j_odd;
		return pow(2, (n - 1)*1.5) * val1(shift_x, 1, 0, p);
	}
}

double AlptBasis::val2(const double x, const int n, const int j, const int p)
{
	// Alpert's wavelets
	if (n == 0) {

		// mesh level 0: normalized legendre polynomial on [0, 1]
		// v_{i,0}(x) = P_i(2x-1) * sqrt(2i+1), where P_i(x) is Legendre polynomial on [-1, 1]
		// (d^2/dx^2)v_{i,0}(x) = 4*(d^2/dx^2)P_i(2x-1) * sqrt(2i+1), where P_i(x) is Legendre polynomial on [-1, 1]
		return legendre_poly_prime2(p, 2 * x - 1) * 4 * sqrt(2 * p + 1.);

		// // mesh level 0: second derivative of normalized legendre polynomial on [0, 1]		
		// if (p == 0 || p == 1)
		// {
		// 	return 0.;
		// }
		// else if (p == 2)
		// {
		// 	return 12.*sqrt(5.);
		// }
		// else if (p == 3)
		// {
		// 	return 60.*sqrt(7.)*(2 * x - 1);
		// }
		// else
		// {
		// 	std::cout << "parameter out of range in Basis::val2" << std::endl; exit(1);
		// }
	}
	else if (n == 1) {

		// mesh level 1: 
		// v_{i,1}(x) = h_i(x) = 2^(1/2)*f_i(2x-1)
		// here f_i(x) is Alpert's wavelets
		// (d2/dx2)v_{i,1}(x) = 2^(5/2)*(d2/dx2)f_i(2x-1)
		if (x >= 0 && x <= 1) return pow(2., 2.5) * phi(2 * x - 1, PMAX, p, 2);
		else return 0.;
	}
	else {

		// mesh level >=2: 
		// v_{i,n}^j(x) = 2^{(n-1)/2} * h_i( 2^(n-1)x - j )
		// (d2/d2x)v_{i,n}^j(x) = 2^{5(n-1)/2} * (d2/dx2)h_i( 2^(n-1)x - j )
		const int j_odd = (j - 1) / 2;
		double shift_x = pow(2, n - 1)*x - j_odd;
		return pow(2, (n - 1)*2.5) * val2(shift_x, 1, 0, p);
	}
}


double AlptBasis::val3(const double x, const int n, const int j, const int p)
{
	// Alpert's wavelets
	if (n == 0) {

		// mesh level 0: normalized legendre polynomial on [0, 1]
		// v_{i,0}(x) = P_i(2x-1) * sqrt(2i+1), where P_i(x) is Legendre polynomial on [-1, 1]
		// (d^3/dx^3)v_{i,0}(x) = 8 * (d^3/dx^3)P_i(2x-1) * sqrt(2i+1), where P_i(x) is Legendre polynomial on [-1, 1]
		return legendre_poly_prime3(p, 2 * x - 1) * 8 * sqrt(2 * p + 1.);

	}
	else if (n == 1) {

		// mesh level 1: 
		// v_{i,1}(x) = h_i(x) = 2^(1/2)*f_i(2x-1)
		// here f_i(x) is Alpert's wavelets
		// (d3/dx3)v_{i,1}(x) = 2^(7/2)*(d3/dx3)f_i(2x-1)
		if (x >= 0 && x <= 1) return pow(2., 3.5) * phi(2 * x - 1, PMAX, p, 3);
		else return 0.;
	}
	else {

		// mesh level >=2: 
		// v_{i,n}^j(x) = 2^{(n-1)/2} * h_i( 2^(n-1)x - j )
		// (d3/dx3)v_{i,n}^j(x) = 2^{7(n-1)/2} * (d3/dx3)h_i( 2^(n-1)x - j )
		const int j_odd = (j - 1) / 2;
		double shift_x = pow(2, n - 1)*x - j_odd;
		return pow(2, (n - 1) * 3.5) * val3(shift_x, 1, 0, p);
	}
}


double AlptBasis::val4(const double x, const int n, const int j, const int p)
{
	// Alpert's wavelets
	if (n == 0) {

		// mesh level 0: normalized legendre polynomial on [0, 1]
		// v_{i,0}(x) = P_i(2x-1) * sqrt(2i+1), where P_i(x) is Legendre polynomial on [-1, 1]
		// (d^4/dx^4)v_{i,0}(x) = 16 * (d^4/dx^4)P_i(2x-1) * sqrt(2i+1), where P_i(x) is Legendre polynomial on [-1, 1]
		return legendre_poly_prime4(p, 2 * x - 1) * 16 * sqrt(2 * p + 1.);

	}
	else if (n == 1) {

		// mesh level 1: 
		// v_{i,1}(x) = h_i(x) = 2^(1/2)*f_i(2x-1)
		// here f_i(x) is Alpert's wavelets
		// (d4/dx4)v_{i,1}(x) = 2^(9/2)*(d4/dx4)f_i(2x-1)
		if (x >= 0 && x <= 1) return pow(2., 4.5) * phi(2 * x - 1, PMAX, p, 4);
		else return 0.;
	}
	else {

		// mesh level >=2: 
		// v_{i,n}^j(x) = 2^{(n-1)/2} * h_i( 2^(n-1)x - j )
		// (d4/dx4)v_{i,n}^j(x) = 2^{9(n-1)/2} * (d4/dx4)h_i( 2^(n-1)x - j )
		const int j_odd = (j - 1) / 2;
		double shift_x = pow(2, n - 1)*x - j_odd;
		return pow(2, (n - 1) * 4.5) * val4(shift_x, 1, 0, p);
	}
}


double AlptBasis::phi(double x, int P, int p, int l)
{
	if (l == 0)
	{
		return phi0(x, P, p);
	}
	else if (l == 1)
	{
		return phi1(x, P, p);
	}
	else if (l == 2)
	{
		return phi2(x, P, p);
	}
	else if (l == 3)
	{
		return phi3(x, P, p);
	}
	else if (l == 4)
	{
		return phi4(x, P, p);
	}
	else
	{
		std::cout << "error in Basis::phi function in AlptBasis.cpp file" << std::endl;
		exit(1);
	}
}

//////////// function value
double AlptBasis::phi0(const double x, const int P, const int p)
{
	double sgn = 1.;
	if (x < 0)
	{
		if ((p + P + 1) % 2 != 0) sgn = -1.;
		return sgn * phi0(-x, P, p);
	}

	double result = 0.;
	if (P == 0) {
		if (p == 0) result = sqrt(0.5);
	}
	else if (P == 1) {
		if (p == 0) result = sqrt(1.5)*(-1. + 2.*x);
		else if (p == 1) result = sqrt(0.5)*(-2. + 3.*x);
	}
	else if (P == 2) {
		if (p == 0) result = sqrt(0.5) / 3. * (1. - 24.*x + 30.*x*x);
		else if (p == 1) result = sqrt(1.5)*0.5 * (3. - 16.*x + 15.*x*x);
		else if (p == 2) result = sqrt(2.5) / 3. * (4. - 15.*x + 12.*x*x);
	}
	else if (P == 3) {
		if (p == 0) result = sqrt(15. / 34.) * (1. + 4.*x - 30.*x*x + 28. *x*x*x);
		else if (p == 1) result = sqrt(1. / 42.) * (-4. + 105.*x - 300.*x*x + 210.*x*x*x);
		else if (p == 2) result = sqrt(35. / 34)*0.5 * (-5. + 48.*x - 105.*x*x + 64.*x*x*x);
		else if (p == 3) result = sqrt(5. / 42)*0.5 * (-16. + 105.*x - 192.*x*x + 105. *x*x*x);
	}
	else if (P == 4)
	{
		if (p == 0) result = sqrt(1. / 186.) * (1. + 30.*x + 210.*x*x - 840.*x*x*x + 630.*x*x*x*x);
		else if(p == 1) result = sqrt(1. / 38.) * 0.5 * (-5. - 144.*x + 1155.*x*x - 2240.*x*x*x + 1260.*x*x*x*x);
		else if(p == 2) result = sqrt(35. / 14694.) * (22. - 735.*x + 3504.*x*x - 5460.*x*x*x + 2700.*x*x*x*x);
		else if(p == 3) result = sqrt(21. / 38.) / 8. * (35. - 512.*x + 1890.*x*x - 2560.*x*x*x + 1155.*x*x*x*x);
		else if(p == 4) result = sqrt(7. / 158.) * 0.5 * (32. - 315.*x + 960.*x*x - 1155.*x*x*x + 480.*x*x*x*x);
	} 
	else {
		std::cout << "polynomial degree is too large in AlptBasis::phi0, modify code in definition of basis function in 1D" << std::endl;
		exit(1);
	}

	return result;
}

//////////// first derivative
double AlptBasis::phi1(const double x, const int P, const int p)
{
	double sgn = 1.;
	if (x < 0)
	{
		if ((p + P + 1) % 2 == 0) sgn = -1.;
		return sgn * phi1(-x, P, p);
	}

	double result = 0.;
	if (P == 0) {
		if (p == 0) result = 0.0;
	}
	else if (P == 1) {
		if (p == 0) result = 2. * sqrt(1.5);
		else if (p == 1) result = 3. * sqrt(0.5);
	}
	else if (P == 2) {
		if (p == 0) result = (-24. + 60.*x) / (3.*sqrt(2.));
		else if (p == 1) result = (sqrt(1.5)*(-16. + 30.*x)) / 2.;
		else if (p == 2) result = (sqrt(2.5)*(-15. + 24.*x)) / 3.;
	}
	else if (P == 3) {
		if (p == 0) result = 0.6642111641550714*(4. - 60.*x + 84 * x*x);
		else if (p == 1) result = 0.1543033499620919*(105. - 600.*x + 630.*x*x);
		else if (p == 2) result = 0.5072996561958923*(48. - 210.*x + 192.*x*x);
		else if (p == 3) result = 0.17251638983558856*(105. - 384.*x + 315.*x*x);
	}
	else if (P == 4)
	{
		if (p == 0) result = sqrt(1. / 186.) * (30. + 420.*x - 2520.*x*x + 2520.*x*x*x);
		else if(p == 1) result = sqrt(1. / 38.) * 0.5 * (-144. + 2310.*x - 6720.*x*x + 5040.*x*x*x);
		else if(p == 2) result = sqrt(35. / 14694.) * (-735. + 7008.*x - 16380.*x*x + 10800.*x*x*x);
		else if(p == 3) result = sqrt(21. / 38.) / 8. * (-512. + 3780.*x - 7680.*x*x + 4620.*x*x*x);
		else if(p == 4) result = sqrt(7. / 158.) * 0.5 * (-315. + 1920.*x - 3465.*x*x + 1920.*x*x*x);
	} 
	else {
		std::cout << "polynomial degree is too large in AlptBasis::phi1, modify code in definition of basis function in 1D" << std::endl;
		exit(1);
	}

	return result;
}

//////////// second derivative
double AlptBasis::phi2(const double x, const int P, const int p)
{
	double sgn = 1.;
	if (x < 0)
	{
		if ((p + P + 1) % 2 != 0) sgn = -1.;
		return sgn * phi2(-x, P, p);
	}

	double result = 0.;
	if (P == 0) {
		if (p == 0) result = 0.;
	}
	else if (P == 1) {
		if (p == 0 || p == 1) result = 0.;
	}
	else if (P == 2) {
		if (p == 0) result = 20. / (sqrt(2.));
		else if (p == 1) result = sqrt(1.5) * 15.;
		else if (p == 2) result = sqrt(2.5) * 8.;
	}
	else if (P == 3) {
		if (p == 0) result = 0.6642111641550714*(-60. + 168. * x);
		else if (p == 1) result = 0.1543033499620919*(-600. + 1260. * x);
		else if (p == 2) result = 0.5072996561958923*(-210. + 384. * x);
		else if (p == 3) result = 0.17251638983558856*(-384. + 630. * x);
	}
	else if (P == 4)
	{
		if (p == 0) result = sqrt(1. / 186.) * (420. - 5040.*x + 7560.*x*x);
		else if(p == 1) result = sqrt(1. / 38.) * 0.5 * (2310. - 13440.*x + 15120.*x*x);
		else if(p == 2) result = sqrt(35. / 14694.) * (7008. - 32760.*x + 32400.*x*x);
		else if(p == 3) result = sqrt(21. / 38.) / 8. * (3780. - 15360.*x + 13860.*x*x);
		else if(p == 4) result = sqrt(7. / 158.) * 0.5 * (1920. - 6930.*x + 5760.*x*x);
	} 
	else {
		std::cout << "polynomial degree is too large in AlptBasis::phi2, modify code in definition of basis function in 1D" << std::endl;
		exit(1);
	}

	return result;
}

//////////// third derivative
double AlptBasis::phi3(const double x, const int P, const int p)
{
	double sgn = 1.;
	if (x < 0)
	{
		if ((p + P + 1) % 2 == 0) sgn = -1.;
		return sgn * phi3(-x, P, p);
	}

	double result = 0.;
	if (P == 0) {
		if (p == 0) result = 0.;
	}
	else if (P == 1) {
		if (p == 0 || p == 1) result = 0.;
	}
	else if (P == 2) {
		if (p == 0 || p == 1 || p == 2) result = 0.;
	}
	else if (P == 3) {
		if (p == 0) result = 168. * sqrt(15. / 34.);
		else if (p == 1) result = 1260. * sqrt(1. / 42.);
		else if (p == 2) result = 192. * sqrt(35. / 34.);
		else if (p == 3) result = 315. * sqrt(5. / 42.);
	}
	else if (P == 4)
	{
		if (p == 0) result = sqrt(1. / 186.) * (-5040. + 15120.*x);
		else if(p == 1) result = sqrt(1. / 38.) * 0.5 * (-13440. + 30240.*x);
		else if(p == 2) result = sqrt(35. / 14694.) * (-32760. + 64800.*x);
		else if(p == 3) result = sqrt(21. / 38.) / 8. * (-15360. + 27720.*x);
		else if(p == 4) result = sqrt(7. / 158.) * 0.5 * (-6930. + 11520.*x);
	} 
	else {
		std::cout << "polynomial degree is too large in AlptBasis::phi3, modify code in definition of basis function in 1D" << std::endl;
		exit(1);
	}

	return result;
}

//////////// fourth derivative
double AlptBasis::phi4(const double x, const int P, const int p)
{
	double sgn = 1.;
	if (x < 0)
	{
		if ((p + P + 1) % 2 != 0) sgn = -1.;
		return sgn * phi4(-x, P, p);
	}

	double result = 0.;
	if (P == 0) {
		if (p == 0) result = 0.;
	}
	else if (P == 1) {
		if (p == 0 || p == 1) result = 0.;
	}
	else if (P == 2) {
		if (p == 0 || p == 1 || p == 2) result = 0.;
	}
	else if (P == 3) {
		if (p == 0 || p == 1 || p == 2 || p == 3) result = 0.;
	}
	else if (P == 4)
	{
		if (p == 0) result = sqrt(1. / 186.) * 15120.;
		else if(p == 1) result = sqrt(1. / 38.) * 15120.;
		else if(p == 2) result = sqrt(35. / 14694.) * 64800.;
		else if(p == 3) result = sqrt(21. / 38.) * 3465.;
		else if(p == 4) result = sqrt(7. / 158.) * 5760.;
	} 
	else {
		std::cout << "polynomial degree is too large in AlptBasis::phi4, modify code in definition of basis function in 1D" << std::endl;
		exit(1);
	}

	return result;
}

/////-----------------------------------------------------------

//////////// function value
double AlptBasis::legendre_poly(const int p, const double x)
{
	if (std::abs(x) > 1.)
	{
		return 0.;
	}

	if (p == 0)
	{
		return 1.;
	}
	else if (p == 1)
	{
		return x;
	}
	else if (p == 2)
	{
		return (3*x*x-1.)/2.;
	}
	else if (p == 3)
	{
		return (5*x*x*x-3*x)/2.;
	}
	else if (p == 4)
	{
		return (35*x*x*x*x-30*x*x+3)/8.;
	}
	else if (p == 5)
	{
		return (63*x*x*x*x*x-70*x*x*x+15*x)/8.;
	}
	else if (p == 6)
	{
		return (231*x*x*x*x*x*x-315*x*x*x*x+105*x*x-5)/16.;
	}
	else
	{
		std::cout << "polynomials too high in function AlptBasis::legendre_poly" << std::endl;
		exit(1);
	}	
}

////////////// first derivative
double AlptBasis::legendre_poly_prime(const int p, const double x)
{
	if (std::abs(x) > 1.)
	{
		return 0.;
	}

	if (p == 0)
	{
		return 0.;
	}
	else if (p == 1)
	{
		return 1;
	}
	else if (p == 2)
	{
		return 3*x;
	}
	else if (p == 3)
	{
		return (15*x*x-3)/2.;
	}
	else if (p == 4)
	{
		return (140*x*x*x-60*x)/8.;
	}
	else if (p == 5)
	{
		return (315*x*x*x*x-210*x*x+15)/8.;
	}
	else if (p == 6)
	{
		return (1386*x*x*x*x*x-1260*x*x*x+210*x)/16.;
	}
	else
	{
		std::cout << "polynomials too high in function AlptBasis::legendre_poly_prime" << std::endl;
		exit(1);
	}	
}

////////////// second derivative
double AlptBasis::legendre_poly_prime2(const int p, const double x)
{
	if (std::abs(x) > 1.)
	{
		return 0.;
	}

	if (p == 0 || p == 1)
	{
		return 0.;
	}
	else if (p == 2)
	{
		return 3;
	}
	else if (p == 3)
	{
		return 15.0 * x;
	}
	else if (p == 4)
	{
		return (420*x*x-60)/8.;
	}
	else if (p == 5)
	{
		return (1260*x*x*x-420*x)/8.;
	}
	// else if (p == 6)
	// {
	// 	return (1386*x*x*x*x*x-1260*x*x*x+210*x)/16.;
	// }
	else
	{
		std::cout << "polynomials too high in function AlptBasis::legendre_poly_prime2" << std::endl;
		exit(1);
	}	
}

////////////// third derivative
double AlptBasis::legendre_poly_prime3(const int p, const double x)
{
	if (std::abs(x) > 1.)
	{
		return 0.;
	}

	if (p == 0 || p == 1 || p == 2)
	{
		return 0.;
	}
	else if (p == 3)
	{
		return 15.0;
	}
	else if (p == 4)
	{
		return 105.0 * x;
	}
	else if (p == 5)
	{
		return (3780 * x * x - 420)/8.;
	}
	else
	{
		std::cout << "polynomials too high in function AlptBasis::legendre_poly_prime3" << std::endl;
		exit(1);
	}	
}

/////////////// fourth derivative
double AlptBasis::legendre_poly_prime4(const int p, const double x)
{
	if (std::abs(x) > 1.)
	{
		return 0.;
	}

	if (p == 0 || p == 1 || p == 2 || p == 3)
	{
		return 0.;
	}
	else if (p == 4)
	{
		return 105.0;
	}
	else if (p == 5)
	{
		return 945.0 * x;
	}
	else
	{
		std::cout << "polynomials too high in function AlptBasis::legendre_poly_prime4" << std::endl;
		exit(1);
	}	
}