#pragma once
#include "Basis.h"

class HermBasis :
	public Basis
{
public:
	HermBasis(const int level_, const int suppt_, const int dgree_);
	~HermBasis() {};

	static int PMAX;
	
	// there are many choice for interpolation points, we denote this by mesh case, need more documents here
	static int msh_case;
	
	static std::vector<double> intp_msh0;	// interpolation point at mesh level 0, defined in [0, 1]
	static std::vector<double> intp_msh1;	// interpolation point at mesh level 1, defined in [0, 1]
	
	double intep_pt;	// interpolation point of this basis function

	static void set_interp_msh01();		// set interpolation points in mesh level 0 and 1
	
	virtual double val(const double x, const int derivative = 0, const int sgn = 0) const override;

private:

	// return index for point and derivative
	static void deg_pt_deri_1d(const int pp, int & p, int & l);

	// return value of this interpolation function with given x and given index (n, j, p)
	static double val0(const double x, const int n, const int j, const int p);

	// return 1st-order derivative value of this interpolation function with given x and given index (n, j, p)
	static double val1(const double x, const int n, const int j, const int p);

	// return 2nd-order derivative value of this interpolation function with given x and given index (n, j, p)
	static double val2(const double x, const int n, const int j, const int p);

	// return 3rd-order derivative value of this interpolation function with given x and given index (n, j, p)
	static double val3(const double x, const int n, const int j, const int p);

	///////////// value of interpolation basis in mesh level 0 and 1

	// P=1: 2 interpolation points
	// L=1: every point has value and 1st-order derivative
	static double phi_P1_L1(const double x, int msh, const int pp);

	// P=1: 2 interpolation points
	// L=1: every point has value and 1st-order derivative
	// D=1: 1st-order derivative of basis function
	static double phi_P1_L1_D1(const double x, int msh, const int pp);


	// P=1: 2 interpolation points
	// L=1: every point has value and 1st-order derivative
	// D=2: 2nd-order derivative of basis function
	static double phi_P1_L1_D2(const double x, int msh, const int pp);


	// P=1: 2 interpolation points
	// L=1: every point has value and 1st-order derivative
	// D=3: 3rd-order derivative of basis function
	static double phi_P1_L1_D3(const double x, int msh, const int pp);

	// P=1: 2 interpolation points
	// L=2: every point has value, 1st-order derivative and 2nd-order derivative
	static double phi_P1_L2(const double x, int msh, const int pp);

	// P=1: 2 interpolation points
	// L=2: every point has value, 1st-order derivative and 2nd_order derivative
	// D=1: 1st-order derivative of basis function
	static double phi_P1_L2_D1(const double x, int msh, const int pp);

	// P=1: 2 interpolation points
	// L=2: every point has value, 1st-order derivative and 2nd_order derivative
	// D=2: 2nd-order derivative of basis function
	static double phi_P1_L2_D2(const double x, int msh, const int pp);	


};