#pragma once
#include "Basis.h"

class AlptBasis :
	public Basis
{
public:
	AlptBasis(const int level_, const int suppt_, const int dgree_);
	~AlptBasis(){};

	static int PMAX;

	virtual double val(const double x, const int derivative = 0, const int sgn_lim = 0) const override;

private:

	// basis function (values and derivatives) with index (n, j, p); x here is global before shifting
	static double val0(const double x, const int n, const int j, const int p);	// value of function
	static double val1(const double x, const int n, const int j, const int p);	// first-order derivative
	static double val2(const double x, const int n, const int j, const int p);	// second-order derivative
	static double val3(const double x, const int n, const int j, const int p);  // third-order derivative
	static double val4(const double x, const int n, const int j, const int p);  // fourth-order derivative

	// Alpert's multiwavelet basis at mesh level 1. They are normalized orthogonal on [-1, 1]
	// P: maximum polynomial degree
	// p: from 0 to P
	// l: derivative degree
	// phi (for different derivative l) is defined by phi0 (l=0), phi1 (l=1), phi2 (l=2)
	static double phi(const double x, const int P, const int p, const int l = 0);
	static double phi0(const double x, const int P, const int p); // value of function
	static double phi1(const double x, const int P, const int p); // first-order derivative of function
	static double phi2(const double x, const int P, const int p); // second-order derivative of function
	static double phi3(const double x, const int P, const int p); // third-order derivative of function
	static double phi4(const double x, const int P, const int p); // fourth-order derivative of function

	
	// Legendre polynomial and its derivative on [-1, 1]
	// vanish for x outside [-1, 1]
	static double legendre_poly(const int p, const double x);
	static double legendre_poly_prime(const int p, const double x);
	static double legendre_poly_prime2(const int p, const double x);
	static double legendre_poly_prime3(const int p, const double x);
	static double legendre_poly_prime4(const int p, const double x);

};

