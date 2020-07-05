#pragma once
#include "libs.h"


//  class for generate exact solutions for Burgers' equations in 1D and 2D with specific initial value.
class BurgersExact
{
public:
	BurgersExact(): a(0.), b(1.), c(0.) {};
	BurgersExact(const double a_, const double b_, const double c_ ): a(a_), b(b_), c(c_) {};
	~BurgersExact(){};
	
	// initial value u(x,0) = a + b*v0(x+c) = a + b * sin( 2*pi*(x+c) )
	// u(x,t) = a + b * v( x-a*t+c, b*t )
	double exact_1d(const double x, const double t);

	// intial value u(x,y,0) = a + b*v0(x+y+c) = a + b*sin(2*pi*(x+y+c))
	// then exact solution is u(x,y,t) = a + b*v( x+y-2*a*t+c, 2*b*t )
	double exact_2d(const double x, const double y, const double t);

private:
	const double pi = Const::PI;
	const double rnd_off = Const::ROUND_OFF;
	const double a, b, c;

	// 1D Burgers' equation, v_t + (v^2/2)_x = 0, in 0 < x < 1
	// initial value v(x,0) = sin(2*pi*x)
	// we call it normalized initial value, which will be used to generate solutions for other initial values in 1D and 2D
	double norm_1d(const double x, const double t);

	// 2D Burgers' equation, u_t + (u^2/2)_x + (u^2/2)_y = 0, in 0 < x, y < 1
	// intial value u(x,y,0) = sin(2*pi*(x+y))
	// then exact solution is u(x,y,t) = v( x+y, 2*t )
    double norm_2d(const double x, const double y, const double t);
};

class HJexact 
{
public:
	virtual double exact_1d(const double x, const double t) = 0;

	virtual double exact_2d(const std::vector<double> x, const double t) = 0;

	virtual double exact_3d(const std::vector<double> x, const double t) = 0;

};

class HJBurgersExact:
	public HJexact
{
public:
	HJBurgersExact(){};
	~HJBurgersExact(){};

	virtual double exact_1d(const double x, const double t) override;

	virtual double exact_2d(const std::vector<double> x, const double t) override;

	virtual double exact_3d(const std::vector<double> x, const double t) override;
private:
	const double pi = Const::PI;
	const double rnd_off = Const::ROUND_OFF;    
};


class HJCosExact:
	public HJexact
{
public:
	HJCosExact() {};
	~HJCosExact() {};

	virtual double exact_1d(const double x, const double t) override;

	
	virtual double exact_2d(const std::vector<double> x, const double t) override;

	virtual double exact_3d(const std::vector<double> x, const double t) override;

private:
	double exact_1d(const double x, const double t, const int dim);
	const double pi = Const::PI;
	const double rnd_off = Const::ROUND_OFF;
};


class HJNonlinearExact:
	public HJexact
{
public:
	HJNonlinearExact() {};
	~HJNonlinearExact() {};


	virtual double exact_1d(const double x, const double t) override { return 0.; } 

	virtual double exact_2d(const std::vector<double> x, const double t) override;

	virtual double exact_3d(const std::vector<double> x, const double t) override { return 0.; }

private:
	const double pi = Const::PI;
	const double rnd_off = Const::ROUND_OFF;
};

