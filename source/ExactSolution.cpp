#include "ExactSolution.h"

// ------------------------------------------------------------
// 
// functions for class HJBurgersExact
// 
// ------------------------------------------------------------
double BurgersExact::norm_1d(const double x, const double t)
{	
	// since solution is periodic, we make it periodic in the range of 0 < x < 1
    double x_period = x;
	x_period = std::fmod(x_period, 1.);
	if (x_period<0) x_period = std::fmod(x_period+1, 1.);

	if ( std::abs(x_period)<rnd_off || std::abs(1-x_period)<rnd_off || std::abs(x_period-0.5)<rnd_off ) return 0.;
	
	// exact solution satisfies u(x) = -u(1-x), thus we only care about x>=0.5
	if ( x_period<0.5 ) return - norm_1d( 1 - x_period, t);

	// exact solution u=u(x,t) satisfies: u = u_0(x-u*t) = sin(2*pi*(x-u*t))
	// solve f(u) := u - sin(2*pi*(x-u*t)) = 0 with Newton's iteration
	// u^(n+1) = G(u^n) := u^n - f(u^n)/f'(u^n), with G(u) = u - ( u + sin(2*pi*(u*t-x)) )/( 1 + 2*pi*t*cos(2*pi*(u*t-x)) )
	auto Gu = [&](double u) { return u - ( u + sin(2*pi*(u*t-x_period)) )/( 1 + 2*pi*t*cos(2*pi*(u*t-x_period)) ); };

	// pich up initial value for Newton's iteration
	// generate uniform array x0 between [0.5, 1], like linspace(0.5, 1, N) in matlab
	// and get characteristic line x = x0 + u0(x0)*t
	const int N = 100;
	const double xl = 0.5, xr = 1;
	const double h = (xr-xl)/N;

	double x0 = 0., xT = 0.;
	double u0=0.;
	for (int i = 0; i < N; ++i){	
		
	 	x0 = xr - i*h;
		xT = x0 + sin(2*pi*x0)*t;

		// find the closest value of x in the array at time t
		// take the value of u at this point (x0,0) as initial value for iteration
		if ( x_period>xT )
		{
			u0 = sin(2*pi*x0);
			break;
		}
	}

	// begin iteration
	double err = 1;		// record error between u in current step and the next step
	double u = u0;
	int iter = 0;
	while ( err > rnd_off )
    {
		double un = Gu(u);
		err = std::abs(un - u);
		u = un;
		
		++iter;
		if ( iter>10 )
        {
			std::cout << "too many iterations in function BurgersExact::norm_1d"; exit(1);
		}
	}
	
	return u;
}

double BurgersExact::exact_1d(const double x, const double t )
{
	return a + b * norm_1d( x-a*t+c, b*t );
}

double BurgersExact::norm_2d(const double x, const double y, const double t)
{
	return norm_1d( x+y, 2*t );
}

double BurgersExact::exact_2d(const double x, const double y, const double t)
{
	return a + b * norm_1d( x+y-2*a*t+c, 2*b*t );
}


// ------------------------------------------------------------
// 
// functions for class HJBurgersExact
// 
// ------------------------------------------------------------
double HJBurgersExact::exact_1d(const double x, const double t)
{
	// since solution is periodic, we make it periodic in the range of 0 < x < 1
    double x_period = x;
	x_period = std::fmod(x, 1.);
	if (x_period < 0) x_period = std::fmod(x_period + 1, 1.);

	if (std::abs(x_period) < rnd_off || std::abs(1 - x_period) < rnd_off) return -1./(2*pi);
	if (std::abs(x_period - 0.5) < rnd_off) return 1. / (2 * pi);

	// exact solution satisfies u(x) = u(1-x), thus we only care about x>=0.5
	if (x_period < 0.5) return exact_1d(1 - x_period, t);

	// we first solve out x0 by x = x0 + u0(x0)*t
	// solve f(x0) := x0 - x + sin(2*pi*x0)*t = 0 with Newton's iteration
	// x^(n+1) = G(x^n) := x^n - f(x^n)/f'(x^n)
	auto Gx = [&](double x0) { return x0 - (x0 - x_period + sin(2 * pi*x0)*t) / (1 + 2 * pi*cos(2 * pi*x0)*t); };

	// pich up initial value for Newton's iteration
	// generate uniform array x0 between [0, 1]
	// and get characteristic line x = x0 + u0(x0)*t
	const int N = 100;
	const double xl = 0.5, xr = 1;
	const double h = (xr - xl) / N;

	double x0 = 0., xT = 0.;
	for (int i = 0; i < N; ++i) {

		x0 = xr - i * h;
		xT = x0 + sin(2 * pi*x0)*t;

		// find the closest value of x in the array at time t
		// take the value of x0 as initial value for iteration
		if (x_period > xT) break;
	}

	// begin iteration
	double err = 1;		// record error between x in current step and the next step
	int iter = 0;
	while (err > rnd_off) {

		double xn = Gx(x0);
		err = std::abs(xn - x0);
		x0 = xn;

		++iter;
		if (iter > 10) {

			std::cout << "too many iterations in function HJBurgers::norm_1d";
			exit(1);
		}
	}

	// exact solution to HJ equation
	return -cos(2*pi*x0)/(2.*pi) + 0.5*sin(2 * pi*x0)*sin(2 * pi*x0)*t;
}

double HJBurgersExact::exact_2d(const std::vector<double> x, const double t)
{
	return exact_1d(x[0] + x[1], 4 * t);
}


double HJBurgersExact::exact_3d(const std::vector<double> x, const double t)
{
	return exact_1d(x[0] + x[1] + x[2], 9 * t);
}



double HJCosExact::exact_1d(const double x, const double t)
{
	// since solution is periodic, we make it periodic in the range of 0 < x < 1
	double x_period = x;
	x_period = std::fmod(x, 1.);
	if (x_period < 0) x_period = std::fmod(x_period + 1, 1.);

	//if (std::abs(x_period) < rnd_off || std::abs(1 - x_period) < rnd_off) return -1. / (2 * pi);
	//if (std::abs(x_period - 0.5) < rnd_off) return 1. / (2 * pi);

	// exact solution satisfies u(x) = u(1-x), thus we only care about x>=0.5
	//if (x_period < 0.5) return exact_1d(1 + x_period, t);

	// we first solve out x0 by x = x0 + u0(x0)*t
	// solve f(x0) := x0 - x + sin(2*sin(4*pi*x0)+1)*t = 0 with Newton's iteration
	// x^(n+1) = G(x^n) := x^n - f(x^n)/f'(x^n)
	auto Gx = [&](double x0) { return x0 - (x0 - x_period + sin(2.*sin(4 * pi*x0)+1)*t ) / (1 + 8 * pi* cos(2.*sin(4 * pi*x0)+1)*cos(4 * pi*x0)*t); };

	// pich up initial value for Newton's iteration
	// generate uniform array x0 between [0, 1]
	// and get characteristic line x = x0 + u0(x0)*t
	const int N = 100;
	const double xl = 0, xr = 1;
	const double h = (xr - xl) / N;

	double x0 = 0., xT = 0.;
	for (int i = 0; i < N; ++i) {

		x0 = xr - i * h;
		xT = x0 + sin(2.*sin(4 * pi*x0) + 1.)*t;

		// find the closest value of x in the array at time t
		// take the value of x0 as initial value for iteration
		if (x_period > xT) break;
	}

	// begin iteration
	double err = 1;		// record error between x in current step and the next step
	int iter = 0;
	while (err > rnd_off) {

		double xn = Gx(x0);
		err = std::abs(xn - x0);
		x0 = xn;

		++iter;
		if (iter > 10) {

			std::cout << "too many iterations in function HJBurgers::norm_1d";
			exit(1);
		}
	}

	// exact solution to HJ equation
	double u = 2.*sin(4 * pi*x0);
	//double xx = x0 + sin(u + 1)*t;
return -cos(4 * pi*x0) / (2.*pi) + ( cos(u+1) + sin(u + 1)*u)*t;
}

double HJCosExact::exact_2d(const std::vector<double> x, const double t)
{
	return exact_1d((x[0] + x[1]) * 0.5, t);
}


double HJCosExact::exact_3d(const std::vector<double> x, const double t)
{
	return exact_1d((x[0] + x[1] + x[2])/3., t);
}


double HJNonlinearExact::exact_2d(const std::vector<double> x, const double t)
{

	double y1 = x[1];
	double y = y1 + 1;
	int icount = 1;
	while (std::abs(y - y1) > rnd_off && icount++ < 100)
	{
		y = y1;
		y1 = y - (-x[1] + y - t * cos(2 * pi*(x[0] - t * sin(2.*pi*y)))) / (1. - pow(t * 2 * pi, 2.)*sin(2 * pi*(x[0] - t * sin(2.*pi*y)))*cos(2.*pi*y));



	}

	

	double x1 = x[0] - sin(2.*pi*y1) * t;

	return -1. / (2.*pi) *(sin(2.*pi*x1) + cos(2.*pi*y1)) - t * cos(2.*pi*x1) * sin(2.*pi*y1);
}