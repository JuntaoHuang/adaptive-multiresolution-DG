#pragma once
#include "Basis.h"

class LagrBasis :
	public Basis
{
public:
	LagrBasis(const int level_, const int suppt_, const int dgree_);
	~LagrBasis() {};

	static int PMAX;
	
	// there are many choices for interpolation points, we denote this by mesh case, need more documents here
	static int msh_case;
	
	static std::vector<double> intp_msh0;	// interpolation point at mesh level 0, defined in [0, 1]
	static std::vector<double> intp_msh1;	// interpolation point at mesh level 1, defined in [0, 1]
	
	double intep_pt;	// interpolation point of this basis function

	static void set_interp_msh01();		// set interpolation points in mesh level 0 and 1
	
	virtual double val(const double x, const int derivative = 0, const int sgn = 0) const override;

private:

	// value of interpolation basis in mesh level 0 and 1
	static double phi(const double x, const int msh_level, const int P, const int p);

	// return value of this interpolation function with given x and given index (n, j, p)
	static double val0(const double x, const int n, const int j, const int p);
};

