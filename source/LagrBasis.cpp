#include "LagrBasis.h"

int LagrBasis::PMAX;
int LagrBasis::msh_case;

std::vector<double> LagrBasis::intp_msh0;
std::vector<double> LagrBasis::intp_msh1;

LagrBasis::LagrBasis(const int level_, const int suppt_, const int dgree_) : 
	Basis(level_, suppt_, dgree_) 
{
	for (size_t i = 0; i < 3; i++)
	{
		lft[i] = val(dis_point[i], 0, -1);
		rgt[i] = val(dis_point[i], 0, 1);
		jmp[i] = rgt[i] - lft[i];
	}

	if(level == 0)
	{
		intep_pt = intp_msh0[dgree];
	}
	else
	{
		double h = supp_interv[1] - supp_interv[0];

		intep_pt = supp_interv[0] + h * intp_msh1[dgree];
	}
	
}


void LagrBasis::set_interp_msh01()
{
	if (PMAX == 1)
	{
		if (msh_case == 1) {
			// p = 1, case 3

			intp_msh0 = std::vector<double>{ -1. / 3., 1. / 3. };
			intp_msh1 = std::vector<double>{ -2. / 3., 2. / 3. };
		}
		else if (msh_case == 2) {
			// p = 1, case 2

			intp_msh0 = std::vector<double>{ -1 + Const::ROUND_OFF, 1 - Const::ROUND_OFF };
			intp_msh1 = std::vector<double>{ 0 - Const::ROUND_OFF, 0 + Const::ROUND_OFF };
		}
	}
	else if (PMAX == 2) {

		if (msh_case == 1) {
			// p = 2, case 7

			intp_msh0 = std::vector<double>{ -2. / 3., -1. / 3., 1. / 3. };
			intp_msh1 = std::vector<double>{ -5. / 6., 1. / 6., 2. / 3. };
		}
		else if (msh_case == 2) {
			// p = 2, case 3

			intp_msh0 = std::vector<double>{ -1 + Const::ROUND_OFF, 0 - Const::ROUND_OFF, 1 - Const::ROUND_OFF };
			intp_msh1 = std::vector<double>{ -0.5 - Const::ROUND_OFF, 0 + Const::ROUND_OFF, 0.5 - Const::ROUND_OFF };
		}
	}
	else if (PMAX == 3) {

		if (msh_case == 1) {
			// p = 3, case 2, discontinuous		

			intp_msh0 = std::vector<double>{ -3. / 5., -1. / 5., 1. / 5., 3. / 5. };
			intp_msh1 = std::vector<double>{ -4. / 5., -2. / 5., 2. / 5., 4. / 5. };
		}
		else if (msh_case == 2) {
			// p = 3, see mathematica file interpolation-p3

			intp_msh0 = std::vector<double>{ -1 + Const::ROUND_OFF, -1. / 2. - Const::ROUND_OFF, 0 - Const::ROUND_OFF, 1 - Const::ROUND_OFF };
			intp_msh1 = std::vector<double>{ -3. / 4. - Const::ROUND_OFF, 0 + Const::ROUND_OFF, 1. / 4. - Const::ROUND_OFF, 1. / 2. - Const::ROUND_OFF };
		}
		else if (msh_case == 3) {
			// p = 3, case 1, continuous

			intp_msh0 = std::vector<double>{ -1 + Const::ROUND_OFF, -1. / 3., 1. / 3., 1 - Const::ROUND_OFF };
			intp_msh1 = std::vector<double>{ -2. / 3., 0 - Const::ROUND_OFF, 0 + Const::ROUND_OFF, 2. / 3. };
		}
	}
	else if (PMAX == 4) {

		if (msh_case == 1) {
			// p = 4, see mathematica file interpolation-p4

			intp_msh0 = std::vector<double>{ -2. / 3., -5. / 12., -1. / 3., 1. / 6., 1. / 3. };
			intp_msh1 = std::vector<double>{ -5. / 6., -17. / 24., 7. / 24., 7. / 12., 2. / 3. };
		}
		else if (msh_case == 2) {
			// p = 4, case 1, continuous

			intp_msh0 = std::vector<double>{ -1 + Const::ROUND_OFF, -0.5 - Const::ROUND_OFF, 0 - Const::ROUND_OFF, 0.5 - Const::ROUND_OFF, 1 - Const::ROUND_OFF };
			intp_msh1 = std::vector<double>{ -0.75 - Const::ROUND_OFF, -0.25 - Const::ROUND_OFF, 0 + Const::ROUND_OFF, 0.25 - Const::ROUND_OFF, 0.75 - Const::ROUND_OFF };
		}
	}
	else if (PMAX == 5) {

		if (msh_case == 1) {
			// p = 5, case 1, continuous

			intp_msh0 = std::vector<double>{ -1 + Const::ROUND_OFF, -0.6, -0.2, 0.2, 0.6, 1 - Const::ROUND_OFF };
			intp_msh1 = std::vector<double>{ -0.8, -0.4, 0.0 - Const::ROUND_OFF, 0.0 + Const::ROUND_OFF, 0.4, 0.8 };
		}
		else if (msh_case == 2) {
			// p = 5, case 2, discontinuos

			intp_msh0 = std::vector<double>{ -5./6., -2./3., -5./12., -1./3., 1./6., 1./3. };
			intp_msh1 = std::vector<double>{ -17./24., -11./12., 7./24., 7./12., 2./3., 1./12. };
		}		
		else
		{
			std::cout << "msh_case out of range in function set_intp_msh01 in LagrBasis.cpp for case P=5" << std::endl;
			exit(1);
		}
	}
	else if (PMAX == 6) {

		std::cout << "to be completed in function set_intp_msh01 in LagrBasis.cpp for case P=6" << std::endl;
		exit(1);
	}	
	else if (PMAX == 7) {

		if (msh_case == 1) {
			// p = 7, case 1, continuous

			const double dx = 1./7.;
			intp_msh0 = std::vector<double>{ -1 + Const::ROUND_OFF, -1 + 2 * dx, -1 + 4 * dx, -1 + 6 * dx, 1 - 6 * dx, 1 - 4 * dx, 1 - 2 * dx, 1 - Const::ROUND_OFF };
			intp_msh1 = std::vector<double>{ -1 + dx, -1 + 3 * dx, -1 + 5 * dx, 0.0 - Const::ROUND_OFF, 0.0 + Const::ROUND_OFF, 1 - 5 * dx, 1 - 3 * dx, 1 - dx };
		}
		else
		{
			std::cout << "msh_case out of range in function set_intp_msh01 in LagrBasis.cpp for case P=6" << std::endl;
			exit(1);
		}		
	}	
	else {

		std::cout << "parameter out of range in function set_intp_msh01 in LagrBasis.cpp" << std::endl;
		exit(1);
	}

	// linear map from [-1, 1] to [0, 1]
	auto f = [](double x) { return (x + 1) / 2.; };
	for (int i = 0; i < PMAX + 1; ++i) {

		intp_msh0[i] = f(intp_msh0[i]);
		intp_msh1[i] = f(intp_msh1[i]);
	}
}


double LagrBasis::val(const double x, const int derivative, const int sgn) const
{
	double xlim = x;
	if (sgn == -1)
	{
		if (std::abs(x) <= Const::ROUND_OFF) return 0.;
		xlim -= Const::ROUND_OFF;
	}
	else if (sgn == 1)
	{
		if (std::abs(x - 1) <= Const::ROUND_OFF) return 0.;
		xlim += Const::ROUND_OFF;
	}

	if (derivative == 0)
	{
		return val0(xlim, level, suppt, dgree);
	}
	else { std::cout << "out of range in val function in LagrBasis.cpp" << std::endl; exit(1); }
}

/**
 * @brief      interpolation basis in mesh level 0 and 1.
 *
 * @param[in]  x     defined on [0, 1]
 * @param[in]  mesh  the mesh level, 0 or 1
 * @param[in]  P     polynomial degree of function space, p = 0, 1, 2, 3, 4
 * @param[in]  p     polynomial degree of this basis, p = 0, 1, 2, ... p
 *
 * @return     value of basis function at a given point x
 *
 * @details    for different types of mesh, see the comments in function set_intp_msh01
 */
double LagrBasis::phi(const double xt, const int msh, const int P, const int p) {

	double x = 2 * xt - 1;	// transfer from [0, 1] to [-1, 1]
	if (x < -1 || x > 1) return 0.;

	if (P == 0) {

		// p=0, type 1
		if (msh == 0) {

			return 1.;
		}
		else if (msh == 1) {

			if (x < 0) return 0.;
			else if (x >= 0) return 1.;
		}

	}
	else if (P == 1) {

		if (msh_case == 1) {

			// p=1
			if (msh == 0) {

				if (p == 0) {

					return -3. / 2.*(x - 1. / 3.);
				}
				else if (p == 1) {

					return 3. / 2.*(x + 1. / 3.);
				}

			}
			else if (msh == 1) {

				if (p == 0) {

					if (x < 0) return -3 * (x + 1. / 3.);
					else if (x >= 0) return 0.;
				}
				else if (p == 1) {

					if (x < 0) return 0.;
					else if (x >= 0) return 3 * (x - 1. / 3.);
				}

			}
		}
		else if (msh_case == 2) {

			// p=1
			if (msh == 0) {

				if (p == 0) {

					return -1. / 2.*(x - 1);
				}
				else if (p == 1) {

					return 1. / 2.*(x + 1);
				}

			}
			else if (msh == 1) {

				if (p == 0) {

					if (x <= 0) return x + 1;
					else if (x > 0) return 0.;
				}
				else if (p == 1) {

					if (x < 0) return 0.;
					else if (x >= 0) return -x + 1;
				}

			}
		}

	}
	else if (P == 2) {

		if (msh_case == 1) {

			// p=2, type 7
			if (msh == 0) {

				if (p == 0) {

					return 3 * (x + 1. / 3.)*(x - 1. / 3.);
				}
				else if (p == 1) {

					return -9. / 2.*(x + 2. / 3.)*(x - 1. / 3.);
				}
				else if (p == 2) {

					return 3. / 2.*(x + 2. / 3.)*(x + 1. / 3.);
				}

			}
			else if (msh == 1) {

				if (p == 0) {

					if (x <= 0) return 12 * (x + 2. / 3.)*(x + 1. / 3.);
					else if (x > 0) return 0.;
				}
				else if (p == 1) {

					if (x < 0) return 0.;
					else if (x >= 0) return 12 * (x - 1. / 3.)*(x - 2. / 3.);
				}
				else if (p == 2) {

					if (x < 0) return 0.;
					else if (x >= 0) return 6 * (x - 1. / 6.)*(x - 1. / 3.);
				}
			}

		}

		else if (msh_case == 2) {

			// p=2, interface
			if (msh == 0) {

				if (p == 0) {

					return x * (x - 1) / 2;
				}
				else if (p == 1) {

					return -(x + 1)*(x - 1);
				}
				else if (p == 2) {

					return x * (x + 1) / 2;
				}

			}
			else if (msh == 1) {

				if (p == 0) {

					if (x <= 0) return -4 * x*(x + 1);
					else if (x > 0) return 0.;
				}
				else if (p == 1) {

					if (x < 0) return 0.;
					else if (x >= 0) return 2 * (x - 0.5)*(x - 1);
				}
				else if (p == 2) {

					if (x < 0) return 0.;
					else if (x >= 0) return -4 * x*(x - 1);
				}
			}

		}
	}
	else if (P == 3) {

		if (msh_case == 1) {

			// p=3, type 2
			if (msh == 0) {

				if (p == 0) {

					return -125. / 48.*(x + 1. / 5.)*(x - 1. / 5.)*(x - 3. / 5.);
				}
				else if (p == 1) {

					return 125. / 16.*(x + 3. / 5.)*(x - 1. / 5.)*(x - 3. / 5.);
				}
				else if (p == 2) {

					return -125. / 16.*(x + 3. / 5.)*(x + 1. / 5.)*(x - 3. / 5.);
				}
				else if (p == 3) {

					return 125. / 48.*(x + 3. / 5.)*(x + 1. / 5.)*(x - 1. / 5.);
				}

			}
			else if (msh == 1) {

				if (p == 0) {

					if (x <= 0) return -125. / 6.*(x + 3. / 5.)*(x + 2. / 5.)*(x + 1. / 5.);
					else if (x > 0) return 0.;
				}
				else if (p == 1) {

					if (x <= 0) return -125. / 2.*(x + 4. / 5.)*(x + 3. / 5.)*(x + 1. / 5.);
					else if (x > 0) return 0.;
				}
				else if (p == 2) {

					if (x < 0) return 0.;
					else if (x >= 0) return 125. / 2.*(x - 1. / 5.)*(x - 3. / 5.)*(x - 4. / 5.);
				}
				else if (p == 3) {

					if (x < 0) return 0.;
					else if (x >= 0) return 125. / 6.*(x - 1. / 5.)*(x - 2. / 5.)*(x - 3. / 5.);
				}

			}
		}
		else if (msh_case == 2) {

			if (msh == 0) {

				if (p == 0) {

					return (-1. / 2. - x)*(-1 + x)*x;
				}
				else if (p == 1) {

					return 8. / 3.*(-1 + x)*x*(1 + x);
				}
				else if (p == 2) {

					return -2 * (-1 + x)*(1. / 2. + x)*(1 + x);
				}
				else if (p == 3) {

					return 1. / 3.*x*(1. / 2. + x)*(1 + x);
				}

			}
			else if (msh == 1) {

				if (p == 0) {

					if (x <= 0) return 64. / 3.*x*(1. / 2. + x)*(1 + x);
					else if (x > 0) return 0.;
				}
				else if (p == 1) {

					if (x <= 0) return 0.;
					else if (x > 0) return -8 * (-1 + x)*(-(1. / 2.) + x)*(-(1. / 4.) + x);
				}
				else if (p == 2) {

					if (x < 0) return 0.;
					else if (x >= 0) return 64. / 3.*(-1 + x)*(-(1. / 2.) + x)*x;
				}
				else if (p == 3) {

					if (x < 0) return 0.;
					else if (x >= 0) return -16 * (-1 + x)*(-(1. / 4.) + x)*x;
				}

			}
		}
		else if (msh_case == 3) {

			if (msh == 0) {

				if (p == 0) {

					return -9. / 16.*(x + 1. / 3.)*(x - 1. / 3.)*(x - 1);
				}
				else if (p == 1) {

					return 27. / 16.*(x + 1)*(x - 1. / 3.)*(x - 1);
				}
				else if (p == 2) {

					return -27. / 16.*(x + 1)*(x + 1. / 3.)*(x - 1);
				}
				else if (p == 3) {

					return 9. / 16.*(x + 1)*(x + 1. / 3.)*(x - 1. / 3.);
				}

			}
			else if (msh == 1) {

				if (p == 0) {

					if (x <= 0) return 27. / 2.*x*(x + 1)*(x + 1. / 3.);
					else if (x > 0) return 0.;
				}
				else if (p == 1) {

					if (x <= 0) return 9. / 2.*(x + 1)*(x + 2. / 3.)*(x + 1. / 3.);
					else if (x > 0) return 0.;
				}
				else if (p == 2) {

					if (x < 0) return 0.;
					else if (x >= 0) return -9. / 2.*(x - 1. / 3.)*(x - 2. / 3.)*(x - 1);
				}
				else if (p == 3) {

					if (x < 0) return 0.;
					else if (x >= 0) return -27. / 2.*x*(x - 1. / 3.)*(x - 1);
				}

			}
		}
	}
	else if (P == 4) {

		if (msh_case == 1) {

			// p=4
			if (msh == 0) {

				if (p == 0) {

					return 72. / 5.*(x - 1. / 3.)*(x - 1. / 6.)*(x + 1. / 3.)*(x + 5. / 12.);
				}
				else if (p == 1) {

					return -768. / 7.*(x - 1. / 3.)*(x - 1. / 6.)*(x + 1. / 3.)*(x + 2. / 3.);
				}
				else if (p == 2) {

					return 108 * (x - 1. / 3.)*(x - 1. / 6.)*(x + 5. / 12.)*(x + 2. / 3.);
				}
				else if (p == 3) {

					return -864. / 35.*(x - 1. / 3.)*(x + 1. / 3.)*(x + 5. / 12.)*(x + 2. / 3.);
				}
				else if (p == 4) {

					return 12 * (x - 1. / 6.)*(x + 1. / 3.)*(x + 5. / 12.)*(x + 2. / 3.);
				}

			}
			else if (msh == 1) {

				if (p == 0) {

					if (x <= 0) return 1152. / 5.*(x + 1. / 3.)*(x + 5. / 12.)*(x + 2. / 3.)*(x + 17. / 24.);
					else if (x > 0) return 0.;
				}
				else if (p == 1) {

					if (x <= 0) return -12288. / 7.*(x + 1. / 3.)*(x + 5. / 12.)*(x + 2. / 3.)*(x + 5. / 6.);
					else if (x > 0) return 0.;
				}
				else if (p == 2) {

					if (x < 0) return 0.;
					else if (x >= 0) return -12288. / 7.*(x - 2. / 3.)*(x - 7. / 12.)*(x - 1. / 3.)*(x - 1. / 6.);
				}
				else if (p == 3) {

					if (x < 0) return 0.;
					else if (x >= 0) return -13824. / 35.*(x - 2. / 3.)*(x - 1. / 3.)*(x - 7. / 24.)*(x - 1. / 6.);
				}
				else if (p == 4) {

					if (x < 0) return 0.;
					else if (x >= 0) return 192 * (x - 7. / 12.)*(x - 1. / 3.)*(x - 7. / 24.)*(x - 1. / 6.);
				}

			}
		}
		else if (msh_case == 2) {

			// p=4
			if (msh == 0) {

				if (p == 0) {

					return 2. / 3.*(x + 1. / 2.)*x*(x - 1. / 2.)*(x - 1);
				}
				else if (p == 1) {

					return -8. / 3.*(x + 1)*x*(x - 1. / 2.)*(x - 1);
				}
				else if (p == 2) {

					return 4 * (x + 1)*(x + 1. / 2.)*(x - 1. / 2.)*(x - 1);
				}
				else if (p == 3) {

					return -8. / 3.*(x + 1)*x*(x + 1. / 2.)*(x - 1);
				}
				else if (p == 4) {

					return 2. / 3.*(x + 1. / 2.)*x*(x - 1. / 2.)*(x + 1);
				}

			}
			else if (msh == 1) {

				if (p == 0) {

					if (x <= 0) return -128. / 3.*(x + 1)*(x + 1. / 2.)*(x + 1. / 4.)*x;
					else if (x > 0) return 0.;
				}
				else if (p == 1) {

					if (x <= 0) return -128. / 3.*(x + 1)*(x + 3. / 4.)*(x + 1. / 2.)*x;
					else if (x > 0) return 0.;
				}
				else if (p == 2) {

					if (x < 0) return 0.;
					else if (x >= 0) return 32. / 3.*(x - 1. / 4.)*(x - 1. / 2.)*(x - 3. / 4.)*(x - 1);
				}
				else if (p == 3) {

					if (x < 0) return 0.;
					else if (x >= 0) return -128. / 3.*x*(x - 1. / 2.)*(x - 3. / 4.)*(x - 1);
				}
				else if (p == 4) {

					if (x < 0) return 0.;
					else if (x >= 0) return -128. / 3.*x*(x - 1. / 4.)*(x - 1. / 2.)*(x - 1);
				}

			}
		}
	}
	else if(P == 5)
	{
		if (msh_case == 1)
		{
			// p=5
			if (msh == 0) {

				if (p == 0) {

					return -1.0 / 768. * (x - 1.) * (5.0*x - 3.) * (5.0*x + 3.) * (5.0*x + 1.) * (5.0*x - 1.);
				}
				else if (p == 1) {

					return 25.0 / 768. * (x - 1.) * (x + 1.) * (5.0*x - 3.)  * (5.0*x + 1.) * (5.0*x - 1.);
				}
				else if (p == 2) {

					return -25.0 / 384. * (x - 1.) * (x + 1.) * (5.0*x - 3.)  * (5.0*x + 3.) * (5.0*x - 1.);
				}
				else if (p == 3) {

					return 25.0 / 384. * (x - 1.) * (x + 1.) * (5.0*x - 3.)  * (5.0*x + 3.) * (5.0*x + 1.);
				}
				else if (p == 4) {

					return -25.0 / 768. * (x - 1.) * (x + 1.) * (5.0*x + 3.)  * (5.0*x + 1.) * (5.0*x - 1.);
				}
				else if (p == 5)
				{
					return 1.0 / 768. * (x + 1.) * (5.0*x - 3.) * (5.0*x + 3.) * (5.0*x + 1.) * (5.0*x - 1.);
				}

			}
			else if (msh == 1) {

				if (p == 0) {

					if (x <= 0) return 25./24. * x * (x + 1.0) * (5.0*x + 1.) * (5.0*x + 2.) * (5.0*x + 3.);
					else if (x > 0) return 0.;
				}
				else if (p == 1) {

					if (x <= 0) return 25./12. * x * (x + 1.0) * (5.0*x + 1.) * (5.0*x + 4.) * (5.0*x + 3.);
					else if (x > 0) return 0.;
				}
				else if (p == 2) {

					if (x <= 0) return 1./24. * (x + 1.0) * (5.0*x + 1.) * (5.0*x + 2.) * (5.0*x + 3.) * (5.0*x + 4.);
					else if (x > 0) return 0.;
				}
				else if (p == 3) {

					if (x < 0) return 0.;
					else if (x >= 0) return -1./24. * (x - 1.0) * (5.0*x - 1.) * (5.0*x - 2.) * (5.0*x - 3.) * (5.0*x - 4.);
				}
				else if (p == 4) {

					if (x < 0) return 0.;
					else if (x >= 0) return -25./12. * x * (x - 1.0) * (5.0*x - 1.) * (5.0*x - 4.) * (5.0*x - 3.);
				}
				else if (p == 5) {

					if (x < 0) return 0.;
					else if (x >= 0) return -25./24. * x * (x - 1.0) * (5.0*x - 1.) * (5.0*x - 2.) * (5.0*x - 3.);
				}

			}
		}
		else if (msh_case == 2)
		{
			if (msh == 0) {

				if (p == 0) {

					return -(864./35.)*(-(1./3.) + x)*(-(1./6.) + x)*(1./3. + x)*(5./12. + x)*(2./3. + x);
				}
				else if (p == 1) {

					return 432./5.*(-(1./3.) + x)*(-(1./6.) + x)*(1./3. + x)*(5./12. + x)*(5./6. + x);
				}
				else if (p == 2) {

					return -(9216./35.)*(-(1./3.) + x)*(-(1./6.) + x)*(1./3. + x)*(2./3. + x)*(5./6. + x);
				}
				else if (p == 3) {

					return 216.*(-(1./3.) + x)*(-(1./6.) + x)*(5./12. + x)*(2./3. + x)*(5./6. + x);
				}
				else if (p == 4) {

					return -(864./35.)*(-(1./3.) + x)*(1./3. + x)*(5./12. + x)*(2./3. + x)*(5./6. + x);
				}
				else if (p == 5)
				{
					return 72./7.*(-(1./6.) + x)*(1./3. + x)*(5./12. + x)*(2./3. + x)*(5./6. + x);
				}

			}
			else if (msh == 1) {

				if (p == 0) {

					if (x <= 0) return -(294912./35.)*(1./3. + x)*(5./12. + x)*(2./3. + x)*(5./6. + x)*(11./12. + x);
					else if (x > 0) return 0.;
				}
				else if (p == 1) {

					if (x <= 0) return -(27648./35.)*(1./3. + x)*(5./12. + x)*(2./3. + x)*(17./24. + x)*(5./6. + x);
					else if (x > 0) return 0.;
				}
				else if (p == 2) {

					if (x <= 0) return 0.;
					else if (x > 0) return -(294912./35.)*(-(2./3.) + x)*(-(7./12.) + x)*(-(1./3.) + x)*(-(1./6.) + x)*(-(1./12.) + x);
				}
				else if (p == 3) {

					if (x < 0) return 0.;
					else if (x >= 0) return -(27648./35.)*(-(2./3.) + x)*(-(1./3.) + x)*(-(7./24.) + x)*(-(1./6.) + x)*(-(1./12.) + x);
				}
				else if (p == 4) {

					if (x < 0) return 0.;
					else if (x >= 0) return 2304./7.*(-(7./12.) + x)*(-(1./3.) + x)*(-(7./24.) + x)*(-(1./6.) + x)*(-(1./12.) + x);
				}
				else if (p == 5) {

					if (x < 0) return 0.;
					else if (x >= 0) return -(27648./35.)*(-(2./3.) + x)*(-(7./12.) + x)*(-(1./3.) + x)*(-(7./24.) + x)*(-(1./6.) + x);
				}

			}
		}		
		else
		{
			std::cout << "msh_case out of range in function phi in LagrBasis.cpp for case P=5" << std::endl;
			exit(1);
		}
		
	}
	else if(P == 7)
	{
		if (msh_case == 1)
		{
			if (msh == 0) {

				if (p == 0) 
				{
					return -(((-1 + x)*(-5 + 7*x)*(-3 + 7*x)*(-1 + 7*x)*(1 + 7*x)*(3 + 7*x)*(5 + 7*x))/92160.);
				}
				else if (p == 1) 
				{
					return (49.*(-1 + x)*(1 + x)*(-5 + 7*x)*(-3 + 7*x)*(-1 + 7*x)*(1 + 7*x)*(3 + 7*x))/92160.;
				}
				else if (p == 2) 
				{
					return -((49.*(-1 + x)*(1 + x)*(-5 + 7*x)*(-3 + 7*x)*(-1 + 7*x)*(1 + 7*x)*(5 + 7*x))/30720.);
				}
				else if (p == 3) 
				{
					return (49.*(-1 + x)*(1 + x)*(-5 + 7*x)*(-3 + 7*x)*(-1 + 7*x)*(3 + 7*x)*(5 + 7*x))/18432.;
				}
				else if (p == 4) 
				{
					return -((49.*(-1 + x)*(1 + x)*(-5 + 7*x)*(-3 + 7*x)*(1 + 7*x)*(3 + 7*x)*(5 + 7*x))/18432.);
				}
				else if (p == 5)
				{
					return (49.*(-1 + x)*(1 + x)*(-5 + 7*x)*(-1 + 7*x)*(1 + 7*x)*(3 + 7*x)*(5 + 7*x))/30720.;
				}
				else if (p == 6)
				{
					return -((49.*(-1 + x)*(1 + x)*(-3 + 7*x)*(-1 + 7*x)*(1 + 7*x)*(3 + 7*x)*(5 + 7*x))/92160.);
				}
				else if (p == 7)
				{
					return ((1 + x)*(-5 + 7*x)*(-3 + 7*x)*(-1 + 7*x)*(1 + 7*x)*(3 + 7*x)*(5 + 7*x))/92160.;
				}
			}
			else if (msh == 1) {

				if (p == 0) 
				{
					if (x <= 0) return 49./720.*x*(1 + x)*(1 + 7*x)*(2 + 7*x)*(3 + 7*x)*(4 + 7*x)*(5 + 7*x);
					else if (x > 0) return 0.;
				}
				else if (p == 1) 
				{
					if (x <= 0) return 49./144.*x*(1 + x)*(1 + 7*x)*(2 + 7*x)*(3 + 7*x)*(5 + 7*x)*(6 + 7*x);
					else if (x > 0) return 0.;
				}
				else if (p == 2) 
				{
					if (x <= 0) return 49./240.*x*(1 + x)*(1 + 7*x)*(3 + 7*x)*(4 + 7*x)*(5 + 7*x)*(6 + 7*x);
					else if (x > 0) return 0.;
				}
				else if (p == 3) 
				{
					if (x <= 0) return 1./720.*(1 + x)*(1 + 7*x)*(2 + 7*x)*(3 + 7*x)*(4 + 7*x)*(5 + 7*x)*(6 + 7*x);
					else if (x > 0) return 0.;
				}
				else if (p == 4) 
				{
					if (x < 0) return 0.;
					else if (x >= 0) return -(1./720.)*(-1 + x)*(-6 + 7*x)*(-5 + 7*x)*(-4 + 7*x)*(-3 + 7*x)*(-2 + 7*x)*(-1 + 7*x);
				}
				else if (p == 5) 
				{
					if (x < 0) return 0.;
					else if (x >= 0) return -(49./240.)*(-1 + x)*x*(-6 + 7*x)*(-5 + 7*x)*(-4 + 7*x)*(-3 + 7*x)*(-1 + 7*x);
				}
				else if (p == 6)
				{
					if (x < 0) return 0.;
					else if (x >= 0) return -(49./144.)*(-1 + x)*x*(-6 + 7*x)*(-5 + 7*x)*(-3 + 7*x)*(-2 + 7*x)*(-1 + 7*x);
				}
				else if (p == 7)
				{
					if (x < 0) return 0.;
					else if (x >= 0) return -(49./720.)*(-1 + x)*x*(-5 + 7*x)*(-4 + 7*x)*(-3 + 7*x)*(-2 + 7*x)*(-1 + 7*x);
				}
			}
		}
		else
		{
			std::cout << "msh_case out of range in function phi in LagrBasis.cpp for case P=5" << std::endl;
			exit(1);
		}
		
	}	

	std::cout << "parameter out of range in function phi_interp" << std::endl;
	exit(1);
}

double LagrBasis::val0(const double x, const int n, const int j, const int p)
{
	if (n <= 1)
	{
		return phi(x, n, PMAX, p);
	}
	else
	{
		int odd_j = (j - 1)/2;
		double xtrans = pow(2, n - 1)*x - odd_j;
		return phi(xtrans, 1, PMAX, p);
	}
}
