#include "Interpolation.h"
#include "Timer.h"
int Interpolation::DIM;
int Interpolation::VEC_NUM;

// compute dgsolution_ptr->hash key for 1d element
int Interpolation::hash_key1d(const int n, const int j)
{
	return (n == 0) ? 0 : ((pow_int(2, n - 1) + (j - 1) / 2));
}

//////////////////////////----> for Lagrange interpolation

// compute the function value of 1D Alpt basis at Lagrange interpolation point

void LagrInterpolation::eval_Lag_pt_at_Alpt_1D()
{
	int n1 = pow_int(2, dgsolution_ptr->NMAX) * (LagrBasis::PMAX + 1);

	int n2 = pow_int(2, dgsolution_ptr->NMAX) * (AlptBasis::PMAX + 1);

	Lag_pt_Alpt_1D.resize(n1);

	for (int i = 0; i < n1; i++)
	{
		Lag_pt_Alpt_1D[i].resize(n2);
	}

	///////////////////

	int m1 = 0;
	int m2 = 0;

	// location of Lagrange interpolation pt
	double pos = 0.;

	double xl = 0.;
	double xr = 0.;

	// loop for Lagrange bases
	for (int k = 0; k < dgsolution_ptr->NMAX + 1; k++)
	{
		int p2k = std::max( pow_int(2, k), 2);

		for (int i = 1; i < p2k; i += 2)
		{
			int a = hash_key1d(k, i) * (LagrBasis::PMAX + 1);

			for (int p1 = 0; p1 < LagrBasis::PMAX + 1; p1++)
			{
				pos = dgsolution_ptr->all_bas_Lag.at(k, i, p1).intep_pt;

				// index for 1st dgsolution_ptr->DIMension of matrix Lag_pt_Alpt_1D
				m1 = a + p1;
				
				// loop for Alpt bases 
				for (int l = 0; l < dgsolution_ptr->NMAX + 1; l++)
				{
					int p2l = std::max( pow_int(2, l), 2);

					for (int j = 1; j < p2l; j += 2)
					{
						int b = hash_key1d(l, j) * (AlptBasis::PMAX + 1);

						for (int p2 = 0; p2 < AlptBasis::PMAX + 1; p2++)
						{
							AlptBasis const &bas = dgsolution_ptr->all_bas.at(l, j, p2);

							xl = bas.supp_interv[0];

							xr = bas.supp_interv[1];

							// index for 2nd dgsolution_ptr->DIMension of matrix Lag_pt_Alpt_1D
							m2 = b + p2;

							if (pos < xl || pos > xr)
							{
								Lag_pt_Alpt_1D[m1][m2] = 0.;
							}
							else
							{
								Lag_pt_Alpt_1D[m1][m2] = bas.val(pos, 0, 0);
							}

						}
						
					}
					
				}
				
 
			}
			 
		}
		
	}
	//////////////////
	
}


// compute the first derivative of 1D Alpt basis at Lagrange interpolation point 

void LagrInterpolation::eval_Lag_pt_at_Alpt_1D_d1()
{
	int n1 = pow_int(2, dgsolution_ptr->NMAX) * (LagrBasis::PMAX + 1);

	int n2 = pow_int(2, dgsolution_ptr->NMAX) * (AlptBasis::PMAX + 1);

	Lag_pt_Alpt_1D_d1.resize(n1);

	for (int i = 0; i < n1; i++)
	{
		Lag_pt_Alpt_1D_d1[i].resize(n2);
	}

	///////////////////

	int m1 = 0;
	int m2 = 0;

	// location of Lagrange interpolation pt
	double pos = 0.;

	double xl = 0.;
	double xr = 0.;

	// loop for Lagrange bases
	for (int k = 0; k < dgsolution_ptr->NMAX + 1; k++)
	{
		int p2k = std::max( pow_int(2, k), 2);

		for (int i = 1; i < p2k; i += 2)
		{
			int a = hash_key1d(k, i) * (LagrBasis::PMAX + 1);

			for (int p1 = 0; p1 < LagrBasis::PMAX + 1; p1++)
			{
				pos = dgsolution_ptr->all_bas_Lag.at(k, i, p1).intep_pt;

				// index for 1st dimension of matrix Lag_pt_Alpt_1D_d1
				m1 = a + p1;
				
				// loop for Alpt bases 
				for (int l = 0; l < dgsolution_ptr->NMAX + 1; l++)
				{
					int p2l = std::max( pow_int(2, l), 2);

					for (int j = 1; j < p2l; j += 2)
					{
						int b = hash_key1d(l, j) * (AlptBasis::PMAX + 1);

						for (int p2 = 0; p2 < AlptBasis::PMAX + 1; p2++)
						{
							AlptBasis const &bas = dgsolution_ptr->all_bas.at(l, j, p2);

							xl = bas.supp_interv[0];

							xr = bas.supp_interv[1];

							// index for 2nd dimension of matrix Lag_pt_Alpt_1D_d1
							m2 = b + p2;

							if (pos < xl || pos > xr)
							{
								Lag_pt_Alpt_1D_d1[m1][m2] = 0.;
							}
							else
							{
								Lag_pt_Alpt_1D_d1[m1][m2] = bas.val(pos, 1, 0); // second index: 1 denotes the first derivative
							}

						}
						
					}
					
				}
				
 
			}
			 
		}
		
	}
	//////////////////
	
}


// compute numerical solution and flux function at Lagrange basis interpolation points

void LagrInterpolation::eval_up_fp_Lag(std::function<double(std::vector<double>, int, int)> func, std::vector< std::vector<bool> > is_intp)
{
	for (auto it = dgsolution_ptr->dg.begin(); it != dgsolution_ptr->dg.end(); it++)
	{
		const std::vector<int> & l = it->second.level;
		const std::vector<int> & j = it->second.suppt;

		std::vector<std::vector<int>> & order = it->second.order_local_intp;

		std::vector< VecMultiD<double> > & up = it->second.up_intp;
		// VecMultiD<double> & fp = it->second.fp_intp;
		std::vector< std::vector< VecMultiD<double> > > & fp = it->second.fp_intp;

		std::vector<double> pos(DIM);

		std::vector<int> p(DIM);

		std::vector<int> m1(DIM);

		for (auto it0 = order.begin(); it0 != order.end(); it0++)
		{
			for (int d = 0; d < DIM; d++)
			{
				p[d] = (*it0)[d];

				m1[d] = hash_key1d(l[d], j[d]) * (LagrBasis::PMAX + 1) + p[d];

				pos[d] = dgsolution_ptr->all_bas_Lag.at(l[d], j[d], p[d]).intep_pt;
			}

			////////////////////////

			std::vector<double> val(VEC_NUM, 0.);

			eval_point_val_Alpt_Lag(pos, m1, val);

			for (int i = 0; i < VEC_NUM; i++)
			{
				up[i].at(*it0) = val[i];
			}

			for (int i = 0; i < VEC_NUM; i++)
			{
				for (int d = 0; d < DIM; d++)
				{
					if (is_intp[i][d] == true)
					{
						fp[i][d].at(*it0) = func(val, i, d);
					}
					
					// fp[i][d].at(*it0) = func(val, i, d);
				}
				
			}
			
			
		}

	}
	
}

// compute flux function at Lagrange basis interpolation points

void LagrInterpolation::eval_fp_Lag(std::function<double(std::vector<double>, int, int)> func, std::vector< std::vector<bool> > is_intp)
{
	for (auto it = dgsolution_ptr->dg.begin(); it != dgsolution_ptr->dg.end(); it++)
	{
		const std::vector<int> & l = it->second.level;
		const std::vector<int> & j = it->second.suppt;

		std::vector<std::vector<int>> & order = it->second.order_local_intp;

		std::vector< VecMultiD<double> > & up = it->second.up_intp;
		// VecMultiD<double> & fp = it->second.fp_intp;
		std::vector< std::vector< VecMultiD<double> > > & fp = it->second.fp_intp;

		for (auto it0 = order.begin(); it0 != order.end(); it0++)
		{
			std::vector<double> val(VEC_NUM, 0.);

			for (int i = 0; i < VEC_NUM; i++)
			{
				val[i] = up[i].at(*it0);
			}

			for (int i = 0; i < VEC_NUM; i++)
			{
				for (int d = 0; d < DIM; d++)
				{
					if (is_intp[i][d] == true)
					{
						fp[i][d].at(*it0) = func(val, i, d);
					}
				}

			}


		}

	}

}




void LagrInterpolation::eval_fp_Lag_full(std::function<double(std::vector<double>, int, int, std::vector<double>)> func, std::vector< std::vector<bool> > is_intp)
{
	for (auto it = dgsolution_ptr->dg.begin(); it != dgsolution_ptr->dg.end(); it++)
	{
		const std::vector<int> & l = it->second.level;
		const std::vector<int> & j = it->second.suppt;

		std::vector<std::vector<int>> & order = it->second.order_local_intp;

		std::vector< VecMultiD<double> > & up = it->second.up_intp;
		// VecMultiD<double> & fp = it->second.fp_intp;
		std::vector< std::vector< VecMultiD<double> > > & fp = it->second.fp_intp;

		std::vector<double> p(DIM), pos(DIM);
		for (auto it0 = order.begin(); it0 != order.end(); it0++)
		{
			for (int d = 0; d < DIM; d++)
			{
				p[d] = (*it0)[d];
				pos[d] = dgsolution_ptr->all_bas_Lag.at(l[d], j[d], p[d]).intep_pt;
			}
			std::vector<double> val(VEC_NUM, 0.);

			for (int i = 0; i < VEC_NUM; i++)
			{
				val[i] = up[i].at(*it0);
			}

			for (int i = 0; i < VEC_NUM; i++)
			{
				for (int d = 0; d < DIM; d++)
				{
					if (is_intp[i][d] == true)
					{
						fp[i][d].at(*it0) = func(val, i, d, pos);
					}
				}

			}


		}

	}

}

// compute flux function at Lagrange basis interpolation points
// just for HJ equation
// just compute case: VEC_NUM = 0, d = 0

void LagrInterpolation::eval_fp_Lag_HJ(std::function<double(std::vector<double>, int, int)> func)
{
	for (auto it = dgsolution_ptr->dg.begin(); it != dgsolution_ptr->dg.end(); it++)
	{
		const std::vector<int> & l = it->second.level;
		const std::vector<int> & j = it->second.suppt;

		std::vector<std::vector<int>> & order = it->second.order_local_intp;

		std::vector< VecMultiD<double> > & up = it->second.up_intp;
		// VecMultiD<double> & fp = it->second.fp_intp;
		std::vector< std::vector< VecMultiD<double> > > & fp = it->second.fp_intp;

		for (auto it0 = order.begin(); it0 != order.end(); it0++)
		{
			std::vector<double> val(VEC_NUM, 0.);

			for (int i = 0; i < VEC_NUM; i++)
			{
				val[i] = up[i].at(*it0);
			}

			fp[0][0].at(*it0) = func(val, 0, 0);

		}

	}

}

// compute numerical solution of nonlinear function in wave at Lagrange basis interpolation points
// D=0: k(x,y) * u_x
// D=1: k(x,y) * u_y

void LagrInterpolation::wave_eval_coe_grad_u_Lag(std::function<double(std::vector<double>, int)> coe_func, std::vector< std::vector<bool> > is_intp)
{
	for (auto it = dgsolution_ptr->dg.begin(); it != dgsolution_ptr->dg.end(); it++)
	{
		const std::vector<int> & l = it->second.level;
		const std::vector<int> & j = it->second.suppt;

		std::vector<std::vector<int>> & order = it->second.order_local_intp;

		// std::vector< VecMultiD<double> > & up = it->second.up_intp;
		// VecMultiD<double> & fp = it->second.fp_intp;
		std::vector< std::vector< VecMultiD<double> > > & fp = it->second.fp_intp;

		std::vector<double> pos(DIM);

		std::vector<int> p(DIM);

		std::vector<int> m1(DIM);

		for (auto it0 = order.begin(); it0 != order.end(); it0++)
		{
			for (int d = 0; d < DIM; d++)
			{
				p[d] = (*it0)[d];

				m1[d] = hash_key1d(l[d], j[d]) * (LagrBasis::PMAX + 1) + p[d];

				pos[d] = dgsolution_ptr->all_bas_Lag.at(l[d], j[d], p[d]).intep_pt;
			}

			////////////////////////

			// std::vector<double> val(VEC_NUM, 0.);

			// eval_point_val_Alpt_Lag(pos, m1, val);

			// for (int i = 0; i < VEC_NUM; i++)
			// {
			// 	up[i].at(*it0) = val[i];
			// }

			double val = 0.;

			for (int i = 0; i < VEC_NUM; i++)
			{
				for (int d = 0; d < DIM; d++)
				{
					if (is_intp[i][d] == true)
					{
						eval_der1_val_Alpt_Lag(pos, m1, d, val);

						fp[i][d].at(*it0) = coe_func(pos, d) * val;
					}

					// eval_der1_val_Alpt_Lag(pos, m1, d, val);

					// fp[i][d].at(*it0) = coe_func(pos, d) * val;

				}

			}


		}

	}

}

// compute numerical solution of nonlinear function in wave at Lagrange basis interpolation points
// D=0: k(x,y) * u
// D=1: k(x,y) * u

void LagrInterpolation::wave_eval_coe_u_Lag(std::function<double(std::vector<double>, int)> coe_func, std::vector< std::vector<bool> > is_intp)
{
	for (auto it = dgsolution_ptr->dg.begin(); it != dgsolution_ptr->dg.end(); it++)
	{
		const std::vector<int> & l = it->second.level;
		const std::vector<int> & j = it->second.suppt;

		std::vector<std::vector<int>> & order = it->second.order_local_intp;

		// std::vector< VecMultiD<double> > & up = it->second.up_intp;
		// VecMultiD<double> & fp = it->second.fp_intp;
		std::vector< std::vector< VecMultiD<double> > > & fp = it->second.fp_intp;

		std::vector<double> pos(DIM);

		std::vector<int> p(DIM);

		std::vector<int> m1(DIM);

		for (auto it0 = order.begin(); it0 != order.end(); it0++)
		{
			for (int d = 0; d < DIM; d++)
			{
				p[d] = (*it0)[d];

				m1[d] = hash_key1d(l[d], j[d]) * (LagrBasis::PMAX + 1) + p[d];

				pos[d] = dgsolution_ptr->all_bas_Lag.at(l[d], j[d], p[d]).intep_pt;
			}

			////////////////////////

			// for (int i = 0; i < VEC_NUM; i++)
			// {
			// 	up[i].at(*it0) = val[i];
			// }

			std::vector<double> val(VEC_NUM, 0.);

			eval_point_val_Alpt_Lag(pos, m1, val);

			for (int i = 0; i < VEC_NUM; i++)
			{
				for (int d = 0; d < DIM; d++)
				{

					if (is_intp[i][d] == true)
					{
						fp[i][d].at(*it0) = coe_func(pos, d) * val[i];
					}

					// fp[i][d].at(*it0) = coe_func(pos, d) * val[i];
				}

			}


		}

	}

}


// compute numerical solution of nonlinear function in artificial viscosity of burgers' equation at Lagrange basis interpolation points
// used in LU decompotion version
// D=0: k(x,y) * u_x
// D=1: k(x,y) * u_y

void LagrInterpolation::eval_coe_grad_u_Lag(std::function<double(std::vector<double>, int)> coe_func,
	std::vector< std::vector<bool> > is_intp, int d0)
{
	for (auto it = dgsolution_ptr->dg.begin(); it != dgsolution_ptr->dg.end(); it++)
	{
		const std::vector<int> & l = it->second.level;
		const std::vector<int> & j = it->second.suppt;

		std::vector<std::vector<int>> & order = it->second.order_local_intp;

		std::vector< VecMultiD<double> > & up = it->second.up_intp;
		// VecMultiD<double> & fp = it->second.fp_intp;
		std::vector< std::vector< VecMultiD<double> > > & fp = it->second.fp_intp;

		std::vector<double> pos(DIM);

		std::vector<int> p(DIM);

		// std::vector<int> m1(DIM);

		for (auto it0 = order.begin(); it0 != order.end(); it0++)
		{
			for (int d = 0; d < DIM; d++)
			{
				p[d] = (*it0)[d];

				// m1[d] = hash_key1d(l[d], j[d]) * (LagrBasis::PMAX + 1) + p[d];

				pos[d] = dgsolution_ptr->all_bas_Lag.at(l[d], j[d], p[d]).intep_pt;
			}

			////////////////////////

			for (int i = 0; i < VEC_NUM; i++)
			{

				if (is_intp[i][d0] == true)
				{
					// eval_der1_val_Alpt_Lag(pos, m1, d, val);

					fp[i][d0].at(*it0) = coe_func(pos, d0) * up[i].at(*it0);
				}

			}


		}

	}

}

void LargViscosInterpolation::eval_artificial_viscos_grad_u_Lag(const std::unordered_set<Element*> & support_element, const double coefficient, std::vector< std::vector<bool> > is_intp, int d0, const bool overlap)
{
	std::vector<double> pos(DIM);
	std::vector<int> p(DIM);

	for (auto & elem : dgsolution_ptr->viscosity_intersect_element)
	{
		const std::vector<int> & l = elem->level;
		const std::vector<int> & j = elem->suppt;

		std::vector<std::vector<int>> & order = elem->order_local_intp;

		std::vector< VecMultiD<double> > & up = elem->up_intp;

		std::vector< std::vector< VecMultiD<double> > > & fp = elem->fp_intp;

		for (auto it0 = order.begin(); it0 != order.end(); it0++)
		{
			for (int d = 0; d < DIM; d++)
			{
				p[d] = (*it0)[d];

				pos[d] = dgsolution_ptr->all_bas_Lag.at(l[d], j[d], p[d]).intep_pt;
			}

			for (int i = 0; i < VEC_NUM; i++)
			{
				if (is_intp[i][d0])
				{
					if (!overlap)
					{
						if (is_pt_in_set_element(pos, support_element))
						{
							fp[i][d0].at(*it0) = coefficient * up[i].at(*it0);
						}
						else
						{
							fp[i][d0].at(*it0) = 0.;
						}
					}
					else
					{
						const int num_support = num_pt_in_set_element(pos, support_element);
						if (num_support != 0)
						{
							fp[i][d0].at(*it0) = num_support * coefficient * up[i].at(*it0);
						}
						else
						{
							fp[i][d0].at(*it0) = 0.;
						}
					}
				}

			}


		}

	}

}


// compute numerical solution of nonlinear function in artificial viscosity of burgers' equation at Lagrange basis interpolation points
// used in LU decompotion version
// D=0: k(x,y) * u
// D=1: k(x,y) * u

void LagrInterpolation::eval_coe_u_Lag(std::function<double(std::vector<double>, int)> coe_func, std::vector< std::vector<bool> > is_intp)
{
	for (auto it = dgsolution_ptr->dg.begin(); it != dgsolution_ptr->dg.end(); it++)
	{
		const std::vector<int> & l = it->second.level;
		const std::vector<int> & j = it->second.suppt;

		std::vector<std::vector<int>> & order = it->second.order_local_intp;

		std::vector< VecMultiD<double> > & up = it->second.up_intp;
		// VecMultiD<double> & fp = it->second.fp_intp;
		std::vector< std::vector< VecMultiD<double> > > & fp = it->second.fp_intp;

		std::vector<double> pos(DIM);

		std::vector<int> p(DIM);

		for (auto it0 = order.begin(); it0 != order.end(); it0++)
		{
			for (int d = 0; d < DIM; d++)
			{
				p[d] = (*it0)[d];

				pos[d] = dgsolution_ptr->all_bas_Lag.at(l[d], j[d], p[d]).intep_pt;
			}

			////////////////////////

			for (int i = 0; i < VEC_NUM; i++)
			{
				for (int d = 0; d < DIM; d++)
				{

					if (is_intp[i][d] == true)
					{
						fp[i][d].at(*it0) = coe_func(pos, d) * up[i].at(*it0);
// std::cout << fp[i][d].at(*it0) << " " << coe_func(pos, d) << " " << up[i].at(*it0) << std::endl;
					}

				}
// exit(1);
			}


		}

	}

}

// read function value at Lagrange point from exact function directly

void LagrInterpolation::eval_up_exact_Lag(std::function<double(std::vector<double>, int)> func)
{
	for (auto it = dgsolution_ptr->dg.begin(); it != dgsolution_ptr->dg.end(); it++)
	{
		const std::vector<int> & l = it->second.level;
		const std::vector<int> & j = it->second.suppt;

		std::vector<std::vector<int>> & order = it->second.order_local_intp;

		std::vector< VecMultiD<double> > & up = it->second.up_intp;

		std::vector<double> pos(DIM);

		std::vector<int> p(DIM);

		for (auto it0 = order.begin(); it0 != order.end(); it0++)
		{
			for (int d = 0; d < DIM; d++)
			{
				p[d] = (*it0)[d];

				pos[d] = dgsolution_ptr->all_bas_Lag.at(l[d], j[d], p[d]).intep_pt;
			}

			for (int i = 0; i < VEC_NUM; i++)
			{
				up[i].at(*it0) = func(pos, i);
			}

		}

	}
}

// read function value at Lagrange point from exact function directly for adaptive interpolation initialization

void LagrInterpolation::eval_up_exact_ada_Lag(std::function<double(std::vector<double>, int)> func)
{
	for (auto it = dgsolution_ptr->dg.begin(); it != dgsolution_ptr->dg.end(); it++)
	{
		// just do for new added elements
		if (it->second.new_add == false) continue;

		const std::vector<int> & l = it->second.level;
		const std::vector<int> & j = it->second.suppt;

		std::vector<std::vector<int>> & order = it->second.order_local_intp;

		std::vector< VecMultiD<double> > & up = it->second.up_intp;

		std::vector<double> pos(DIM);

		std::vector<int> p(DIM);

		for (auto it0 = order.begin(); it0 != order.end(); it0++)
		{
			for (int d = 0; d < DIM; d++)
			{
				p[d] = (*it0)[d];

				pos[d] = dgsolution_ptr->all_bas_Lag.at(l[d], j[d], p[d]).intep_pt;
			}

			for (int i = 0; i < VEC_NUM; i++)
			{
				up[i].at(*it0) = func(pos, i);
			}

		}

	}
}

// find points and weights which will affact the coefficients of the Lagrange interpolation points

void LagrInterpolation::set_pts_wts_1d_ada_Lag(int l, int j)
{
	const int P1 = LagrBasis::PMAX + 1;

	double pos1;

	std::vector<double> p_pos(P1);

	std::vector<int> p_k(P1), p_i(P1), p_num(P1);

	double xleft, xright;

	int lj = hash_key1d(l, j);

	xleft = dgsolution_ptr->all_bas_Lag.at(l, j, 0).supp_interv[0];

	xright = dgsolution_ptr->all_bas_Lag.at(l, j, 0).supp_interv[1];

	pwts a;

	if (l > 0) {

		int ic = 0;

		for (int k = 0; k < l; ++k) {

			if (ic == P1) continue;

			int p2k = std::max(2, pow_int(2, k));

			for (int i = 1; i < p2k; i += 2) {

				for (int q0 = 0; q0 < P1; ++q0) {

					pos1 = dgsolution_ptr->all_bas_Lag.at(k, i, q0).intep_pt;

					double eps = 0.0;

					if (pos1 > xleft - eps && pos1 < xright + eps) {

						if (ic == P1)  std::cout << ic << std::endl;

						p_pos[ic] = pos1;

						p_k[ic] = k;

						p_i[ic] = i;

						p_num[ic] = q0;

						ic++;

					}
				}
				///////////////////
			}

		}

		//////////////////////

		for (int i1 = 0; i1 < P1; ++i1) { //means P1 points will affect

			a.p_k.push_back(p_k[i1]);

			a.p_i.push_back(p_i[i1]);

			a.p_num.push_back(p_num[i1]);
		}

		////////////compute wts

		a.wt.resize(P1);

		std::vector<double> pos;

		for (int ic = 0; ic < P1; ++ic) {

			pos.push_back(p_pos[ic]);
		}

		sort(pos.begin(), pos.end());

		for (int p0 = 0; p0 < P1; ++p0) {

			for (int ic = 0; ic < P1; ++ic) {

				int ic0;

				for (int i2 = 0; i2 < P1; ++i2) {

					if (std::fabs(p_pos[ic] - pos[i2]) < 1.0e-15) {

						ic0 = i2;
					}

				}
				///////////////////

				double xt = LagrBasis::intp_msh1[p0];

				a.wt[p0].push_back(-1.0 * dgsolution_ptr->all_bas_Lag.at(0, 1, ic0).val(xt));

				//////////////////////
			}

		}

		/////////////////////
		pw1d.insert(std::make_pair(lj, a));

	}
}

// u: compute hierarchical coefficients of Lagrange basis based on function value at Lagrange point 

void LagrInterpolation::eval_up_to_coe_D_Lag()
{
	int P1 = LagrBasis::PMAX + 1;

	///////////////////

	for (auto it = dgsolution_ptr->dg.begin(); it != dgsolution_ptr->dg.end(); it++)
	{

		std::vector<std::vector<int>> & order = it->second.order_local_intp;

		std::vector< VecMultiD<double> > & up = it->second.up_intp;

		std::vector< VecMultiD<std::vector<double>>> & ucoe_1 = it->second.ucoe_intp_inter;

		for (auto it0 = order.begin(); it0 != order.end(); it0++)
		{
			for (int i = 0; i < VEC_NUM; i++)
			{
				ucoe_1[i].at(*it0)[0] = up[i].at(*it0);
			}

		}

	}

	////////////////////

	for (int d = 0; d < DIM; d++)
	{
		for (auto it = dgsolution_ptr->dg.begin(); it != dgsolution_ptr->dg.end(); it++)
		{
			const std::vector<int> & l = it->second.level;
			const std::vector<int> & j = it->second.suppt;

			std::vector<std::vector<int>> & order = it->second.order_local_intp;

			std::vector< VecMultiD<std::vector<double>>> & ucoe_1 = it->second.ucoe_intp_inter;

			//////////////////////////

			int lj = (it->second.order_elem)[d];

			auto itpw = pw1d.find(lj);

			if (itpw == pw1d.end() && lj > 0)
			{
				set_pts_wts_1d_ada_Lag(l[d], j[d]);

				itpw = pw1d.find(lj);
			}

			for (auto it0 = order.begin(); it0 != order.end(); it0++)
			{
				std::vector<int> k(DIM), i(DIM), p(DIM), q(DIM);

				////////////////

				std::vector<double> coe(VEC_NUM, 0.);

				for (int i = 0; i < VEC_NUM; i++)
				{
					coe[i] = (ucoe_1[i].at(*it0))[d];
				}

				///////////////////

				if (l[d] > 0)
				{
					for (int d0 = 0; d0 < DIM; d0++)
					{
						p[d0] = (*it0)[d0];
					}

					for (int ic = 0; ic < P1; ic++)
					{
						std::vector<int> mm;

						for (int d1 = 0; d1 < DIM; d1++)
						{
							if (d1 == d)
							{
								k[d] = itpw->second.p_k[ic];

								i[d] = itpw->second.p_i[ic];

								q[d] = itpw->second.p_num[ic];

								mm.push_back(itpw->second.p_num[ic]);

							}
							else
							{
								k[d1] = l[d1];

								i[d1] = j[d1];

								q[d1] = p[d1];

								mm.push_back(p[d1]);
							}

						}
						//////////////

						int hak = dgsolution_ptr->hash.hash_key(k, i);

						auto its = dgsolution_ptr->dg.find(hak);

						if (its == dgsolution_ptr->dg.end())
						{
							std::cout << "wrong" << std::endl;

						}

						for (int i = 0; i < VEC_NUM; i++)
						{
							coe[i] += itpw->second.wt[p[d]][ic] * ((its->second.ucoe_intp_inter)[i].at(mm))[d];
						}

						// coe += itpw->second.wt[p[d]][ic] * ((its->second.ucoe_intp_inter).at(mm))[d];

					}

				}

				for (int i = 0; i < VEC_NUM; i++)
				{
					(ucoe_1[i].at(*it0))[d + 1] = coe[i];
				}

			}
			/////////////////////////

		}

	}
	/////////////////
	for (auto it = dgsolution_ptr->dg.begin(); it != dgsolution_ptr->dg.end(); it++)
	{
		std::vector< VecMultiD<double> > & ucoe = it->second.ucoe_intp;

		std::vector< VecMultiD<std::vector<double>> > & ucoe_1 = it->second.ucoe_intp_inter;

		std::vector<std::vector<int>> & order = it->second.order_local_intp;

		for (auto it0 = order.begin(); it0 != order.end(); it0++)
		{
			for (int i = 0; i < VEC_NUM; i++)
			{
				ucoe[i].at(*it0) = (ucoe_1[i].at(*it0))[DIM];
			}

		}

	}

}

// u: compute hierarchical coefficients of Lagrange basis based on function value at Lagrange point
// for adaptive interpolation

void LagrInterpolation::eval_up_to_coe_D_ada_Lag()
{
	int P1 = LagrBasis::PMAX + 1;

	///////////////////

	for (auto it = dgsolution_ptr->dg.begin(); it != dgsolution_ptr->dg.end(); it++)
	{
		// just do for new added elements
		if (it->second.new_add == false) continue;

		std::vector<std::vector<int>> & order = it->second.order_local_intp;

		std::vector< VecMultiD<double> > & up = it->second.up_intp;

		std::vector< VecMultiD<std::vector<double>>> & ucoe_1 = it->second.ucoe_intp_inter;

		for (auto it0 = order.begin(); it0 != order.end(); it0++)
		{
			for (int i = 0; i < VEC_NUM; i++)
			{
				ucoe_1[i].at(*it0)[0] = up[i].at(*it0);
			}

		}

	}

	////////////////////

	for (int d = 0; d < DIM; d++)
	{
		for (auto it = dgsolution_ptr->dg.begin(); it != dgsolution_ptr->dg.end(); it++)
		{
			// just do for new added elements
			if (it->second.new_add == false) continue;

			const std::vector<int> & l = it->second.level;
			const std::vector<int> & j = it->second.suppt;

			std::vector<std::vector<int>> & order = it->second.order_local_intp;

			std::vector< VecMultiD<std::vector<double>>> & ucoe_1 = it->second.ucoe_intp_inter;

			//////////////////////////

			int lj = (it->second.order_elem)[d];

			auto itpw = pw1d.find(lj);

			if (itpw == pw1d.end() && lj > 0)
			{
				set_pts_wts_1d_ada_Lag(l[d], j[d]);

				itpw = pw1d.find(lj);
			}

			for (auto it0 = order.begin(); it0 != order.end(); it0++)
			{
				std::vector<int> k(DIM), i(DIM), p(DIM), q(DIM);

				////////////////

				std::vector<double> coe(VEC_NUM, 0.);

				for (int i = 0; i < VEC_NUM; i++)
				{
					coe[i] = (ucoe_1[i].at(*it0))[d];
				}

				///////////////////

				if (l[d] > 0)
				{
					for (int d0 = 0; d0 < DIM; d0++)
					{
						p[d0] = (*it0)[d0];
					}

					for (int ic = 0; ic < P1; ic++)
					{
						std::vector<int> mm;

						for (int d1 = 0; d1 < DIM; d1++)
						{
							if (d1 == d)
							{
								k[d] = itpw->second.p_k[ic];

								i[d] = itpw->second.p_i[ic];

								q[d] = itpw->second.p_num[ic];

								mm.push_back(itpw->second.p_num[ic]);

							}
							else
							{
								k[d1] = l[d1];

								i[d1] = j[d1];

								q[d1] = p[d1];

								mm.push_back(p[d1]);
							}

						}
						//////////////

						int hak = dgsolution_ptr->hash.hash_key(k, i);

						auto its = dgsolution_ptr->dg.find(hak);

						if (its == dgsolution_ptr->dg.end())
						{
							std::cout << "wrong" << std::endl;

						}

						for (int i = 0; i < VEC_NUM; i++)
						{
							coe[i] += itpw->second.wt[p[d]][ic] * ((its->second.ucoe_intp_inter)[i].at(mm))[d];
						}

						// coe += itpw->second.wt[p[d]][ic] * ((its->second.ucoe_intp_inter).at(mm))[d];

					}

				}

				for (int i = 0; i < VEC_NUM; i++)
				{
					(ucoe_1[i].at(*it0))[d + 1] = coe[i];
				}

			}
			/////////////////////////

		}

	}
	/////////////////
	for (auto it = dgsolution_ptr->dg.begin(); it != dgsolution_ptr->dg.end(); it++)
	{
		// just do for new added elements
		if (it->second.new_add == false) continue;

		std::vector< VecMultiD<double> > & ucoe = it->second.ucoe_intp;

		std::vector< VecMultiD<std::vector<double>> > & ucoe_1 = it->second.ucoe_intp_inter;

		std::vector<std::vector<int>> & order = it->second.order_local_intp;

		for (auto it0 = order.begin(); it0 != order.end(); it0++)
		{
			for (int i = 0; i < VEC_NUM; i++)
			{
				ucoe[i].at(*it0) = (ucoe_1[i].at(*it0))[DIM];
			}

		}

	}

}

// fu: compute hierarchical coefficients of Lagrange basis based on function value at Lagrange point 

void LagrInterpolation::eval_fp_to_coe_D_Lag(std::vector< std::vector<bool> > is_intp)
{
	int P1 = LagrBasis::PMAX + 1;

	///////////////////

	for (auto it = dgsolution_ptr->dg.begin(); it != dgsolution_ptr->dg.end(); it++)
	{

		std::vector<std::vector<int>> & order = it->second.order_local_intp;

		std::vector< std::vector< VecMultiD<double> > > & fp = it->second.fp_intp;

		std::vector< std::vector< VecMultiD<std::vector<double>> > > & fucoe_1 = it->second.fucoe_intp_inter;

		for (auto it0 = order.begin(); it0 != order.end(); it0++)
		{
			for (int i = 0; i < VEC_NUM; i++)
			{
				for (int d = 0; d < DIM; d++)
				{
					if (is_intp[i][d] == true)
					{
						fucoe_1[i][d].at(*it0)[0] = fp[i][d].at(*it0);
					}

					// fucoe_1[i][d].at(*it0)[0] = fp[i][d].at(*it0);
				}

			}

		}

	}

	////////////////////

	for (int d = 0; d < DIM; d++)
	{
		for (auto it = dgsolution_ptr->dg.begin(); it != dgsolution_ptr->dg.end(); it++)
		{
			const std::vector<int> & l = it->second.level;
			const std::vector<int> & j = it->second.suppt;

			std::vector<std::vector<int>> & order = it->second.order_local_intp;

			// VecMultiD<std::array<double, dgsolution_ptr->DIM+1>> & fucoe_1 = it->second.fucoe_intp_inter;

			std::vector< std::vector< VecMultiD<std::vector<double>> > > & fucoe_1 = it->second.fucoe_intp_inter;

			//////////////////////////

			int lj = (it->second.order_elem)[d];

			auto itpw = pw1d.find(lj);

			if (itpw == pw1d.end() && lj > 0)
			{
				set_pts_wts_1d_ada_Lag(l[d], j[d]);

				itpw = pw1d.find(lj);
			}

			std::vector<int> k(DIM), i(DIM), p(DIM), q(DIM);
			std::vector<std::vector<double>> coe;
			coe.resize(VEC_NUM);
			for (auto & u : coe) u.resize(DIM);

			for (auto it0 = order.begin(); it0 != order.end(); it0++)
			{
				//std::vector<int> k(DIM), i(DIM), p(DIM), q(DIM);

				////////////////

				//std::vector<std::vector<double>> coe;

				//coe.resize(VEC_NUM);

				for (int i = 0; i < VEC_NUM; i++)
				{
					//coe[i].resize(DIM);

					for (int d2 = 0; d2 < DIM; d2++)
					{
						// coe[i][d2] = (fucoe_1[i][d2].at(*it0))[d];

						if (is_intp[i][d2] == true)
						{
							coe[i][d2] = (fucoe_1[i][d2].at(*it0))[d];
						}


					}
				}

				//////////////////

				if (l[d] > 0)
				{
					for (int d0 = 0; d0 < DIM; d0++)
					{
						p[d0] = (*it0)[d0];
					}

					for (int ic = 0; ic < P1; ic++)
					{
						std::vector<int> mm;

						for (int d1 = 0; d1 < DIM; d1++)
						{
							if (d1 == d)
							{
								k[d] = itpw->second.p_k[ic];

								i[d] = itpw->second.p_i[ic];

								q[d] = itpw->second.p_num[ic];

								mm.push_back(itpw->second.p_num[ic]);

							}
							else
							{
								k[d1] = l[d1];

								i[d1] = j[d1];

								q[d1] = p[d1];

								mm.push_back(p[d1]);
							}

						}
						//////////////

						int hak = dgsolution_ptr->hash.hash_key(k, i);

						auto its = dgsolution_ptr->dg.find(hak);

						if (its == dgsolution_ptr->dg.end())
						{
							std::cout << "wrong" << std::endl;

						}

						for (int i = 0; i < VEC_NUM; i++)
						{
							for (int d2 = 0; d2 < DIM; d2++)
							{
								if (is_intp[i][d2] == true)
								{
									coe[i][d2] += itpw->second.wt[p[d]][ic] * ((its->second.fucoe_intp_inter)[i][d2].at(mm))[d];
								}

							}
						}

						// coe += itpw->second.wt[p[d]][ic] * ((its->second.fucoe_intp_inter).at(mm))[d];

					}

				}

				for (int i = 0; i < VEC_NUM; i++)
				{
					for (int d2 = 0; d2 < DIM; d2++)
					{
						if (is_intp[i][d2] == true)
						{
							(fucoe_1[i][d2].at(*it0))[d + 1] = coe[i][d2];
						}
					}
				}

			}
			/////////////////////////

		}

	}
	/////////////////
	for (auto it = dgsolution_ptr->dg.begin(); it != dgsolution_ptr->dg.end(); it++)
	{
		std::vector< std::vector< VecMultiD<double> > > & fucoe = it->second.fucoe_intp;

		std::vector< std::vector< VecMultiD<std::vector<double>> > > & fucoe_1 = it->second.fucoe_intp_inter;

		std::vector<std::vector<int>> & order = it->second.order_local_intp;

		for (auto it0 = order.begin(); it0 != order.end(); it0++)
		{
			for (int i = 0; i < VEC_NUM; i++)
			{
				for (int d2 = 0; d2 < DIM; d2++)
				{

					if (is_intp[i][d2] == true)
					{
						fucoe[i][d2].at(*it0) = (fucoe_1[i][d2].at(*it0))[DIM];
					}

				}
			}

		}

	}

}

// fu: compute hierarchical coefficients of Lagrange basis based on function value at Lagrange point 
// for HJ equation
// just compute case: VEC_NUM = 0, d = 0

void LagrInterpolation::eval_fp_to_coe_D_Lag_HJ()
{
	int P1 = LagrBasis::PMAX + 1;

	///////////////////

	for (auto it = dgsolution_ptr->dg.begin(); it != dgsolution_ptr->dg.end(); it++)
	{

		std::vector<std::vector<int>> & order = it->second.order_local_intp;

		std::vector< std::vector< VecMultiD<double> > > & fp = it->second.fp_intp;

		std::vector< std::vector< VecMultiD<std::vector<double>> > > & fucoe_1 = it->second.fucoe_intp_inter;

		for (auto it0 = order.begin(); it0 != order.end(); it0++)
		{
			fucoe_1[0][0].at(*it0)[0] = fp[0][0].at(*it0);
		}

	}

	////////////////////

	for (int d = 0; d < DIM; d++)
	{
		for (auto it = dgsolution_ptr->dg.begin(); it != dgsolution_ptr->dg.end(); it++)
		{
			const std::vector<int> & l = it->second.level;
			const std::vector<int> & j = it->second.suppt;

			std::vector<std::vector<int>> & order = it->second.order_local_intp;

			// VecMultiD<std::array<double, dgsolution_ptr->DIM+1>> & fucoe_1 = it->second.fucoe_intp_inter;

			std::vector< std::vector< VecMultiD<std::vector<double>> > > & fucoe_1 = it->second.fucoe_intp_inter;

			//////////////////////////

			int lj = (it->second.order_elem)[d];

			auto itpw = pw1d.find(lj);

			if (itpw == pw1d.end() && lj > 0)
			{
				set_pts_wts_1d_ada_Lag(l[d], j[d]);

				itpw = pw1d.find(lj);
			}

			for (auto it0 = order.begin(); it0 != order.end(); it0++)
			{
				std::vector<int> k(DIM), i(DIM), p(DIM), q(DIM);

				////////////////

				std::vector<std::vector<double>> coe;

				coe.resize(VEC_NUM);

				for (int i = 0; i < VEC_NUM; i++)
				{
					coe[i].resize(DIM);

					coe[i][0] = (fucoe_1[i][0].at(*it0))[d];

					// for (int d2 = 0; d2 < DIM; d2++)
					// {
					// 	// coe[i][d2] = (fucoe_1[i][d2].at(*it0))[d];

					// 	if (is_intp[i][d2] == true)
					// 	{
					// 		coe[i][d2] = (fucoe_1[i][d2].at(*it0))[d];
					// 	}

					// }
				}

				//////////////////

				if (l[d] > 0)
				{
					for (int d0 = 0; d0 < DIM; d0++)
					{
						p[d0] = (*it0)[d0];
					}

					for (int ic = 0; ic < P1; ic++)
					{
						std::vector<int> mm;

						for (int d1 = 0; d1 < DIM; d1++)
						{
							if (d1 == d)
							{
								k[d] = itpw->second.p_k[ic];

								i[d] = itpw->second.p_i[ic];

								q[d] = itpw->second.p_num[ic];

								mm.push_back(itpw->second.p_num[ic]);

							}
							else
							{
								k[d1] = l[d1];

								i[d1] = j[d1];

								q[d1] = p[d1];

								mm.push_back(p[d1]);
							}

						}
						//////////////

						int hak = dgsolution_ptr->hash.hash_key(k, i);

						auto its = dgsolution_ptr->dg.find(hak);

						if (its == dgsolution_ptr->dg.end())
						{
							std::cout << "wrong" << std::endl;

						}

						// for (int i = 0; i < VEC_NUM; i++)
						// {
						// 	for (int d2 = 0; d2 < DIM; d2++)
						// 	{
						// 		if (is_intp[i][d2] == true)
						// 		{
						// 			coe[i][d2] += itpw->second.wt[p[d]][ic] * ((its->second.fucoe_intp_inter)[i][d2].at(mm))[d];									
						// 		}

						// 	}
						// }

						coe[0][0] += itpw->second.wt[p[d]][ic] * ((its->second.fucoe_intp_inter)[0][0].at(mm))[d];

					}

				}

				// for (int i = 0; i < VEC_NUM; i++)
				// {
				// 	for (int d2 = 0; d2 < DIM; d2++)
				// 	{
				// 		if (is_intp[i][d2] == true)
				// 		{
				// 			(fucoe_1[i][d2].at(*it0))[d+1] = coe[i][d2];
				// 		}
				// 	}
				// }

				(fucoe_1[0][0].at(*it0))[d + 1] = coe[0][0];

			}
			/////////////////////////

		}

	}
	/////////////////
	for (auto it = dgsolution_ptr->dg.begin(); it != dgsolution_ptr->dg.end(); it++)
	{
		std::vector< std::vector< VecMultiD<double> > > & fucoe = it->second.fucoe_intp;

		std::vector< std::vector< VecMultiD<std::vector<double>> > > & fucoe_1 = it->second.fucoe_intp_inter;

		std::vector<std::vector<int>> & order = it->second.order_local_intp;

		for (auto it0 = order.begin(); it0 != order.end(); it0++)
		{
			// for (int i = 0; i < VEC_NUM; i++)
			// {
			// 	for (int d2 = 0; d2 < DIM; d2++)
			// 	{

			// 		if (is_intp[i][d2] == true)
			// 		{
			// 			fucoe[i][d2].at(*it0) = (fucoe_1[i][d2].at(*it0))[DIM];
			// 		}

			// 	}
			// }

			fucoe[0][0].at(*it0) = (fucoe_1[0][0].at(*it0))[DIM];

		}

	}

}

// compute coefficients of Alpt basis from coefficients of Lagrange basis

void LagrInterpolation::eval_coe_alpt_from_coe_Lag(const OperatorMatrix1D<LagrBasis, AlptBasis> & matrix)
{
	for (auto it = dgsolution_ptr->dg.begin(); it != dgsolution_ptr->dg.end(); it++)
	{
		const std::vector<int> & l = it->second.level;
		const std::vector<int> & j = it->second.suppt;

		std::vector<std::vector<int>> & order = it->second.order_local_alpt;

		std::vector<VecMultiD<double>> & ucoe = it->second.ucoe_alpt;

		std::vector<double> & xl = it->second.xl;
		std::vector<double> & xr = it->second.xr;

		for (auto it0 = order.begin(); it0 != order.end(); it0++)
		{
			std::vector<int> p(DIM, 0);

			std::vector<int> m1(DIM, 0);

			std::vector<double> val(VEC_NUM, 0.);

			for (int d = 0; d < DIM; d++)
			{
				p[d] = (*it0)[d];

				m1[d] = hash_key1d(l[d], j[d]) * (AlptBasis::PMAX + 1) + p[d];
			}

			for (auto its = dgsolution_ptr->dg.begin(); its != dgsolution_ptr->dg.end(); its++)
			{
				const std::vector<int> & k = its->second.level;
				const std::vector<int> & i = its->second.suppt;

				std::vector<std::vector<int>> & order_s = its->second.order_local_intp;

				std::vector<VecMultiD<double>> & ucoe_intp = its->second.ucoe_intp;

				std::vector<double> & xl_s = its->second.xl;
				std::vector<double> & xr_s = its->second.xr;

				///////////////////

				int kk = 0;

				for (int d = 0; d < DIM; d++)
				{
					if (xl[d] > xr_s[d] || xl_s[d] > xr[d])
					{
						kk = 1;
					}
				}

				if (kk == 1) continue;

				/////////////////////

				for (auto it1 = order_s.begin(); it1 != order_s.end(); it1++)
				{
					std::vector<int> q(DIM, 0);

					std::vector<int> m2(DIM, 0);

					double tep = 1.0;

					for (int d = 0; d < DIM; d++)
					{
						q[d] = (*it1)[d];

						m2[d] = hash_key1d(k[d], i[d]) * (LagrBasis::PMAX + 1) + q[d];

						tep = tep * matrix.u_v.at(m2[d], m1[d]); // since: row: Lag, col: Alpt 
					}

					for (int i = 0; i < VEC_NUM; i++)
					{
						val[i] += ucoe_intp[i].at(*it1) * tep;
					}

				} // it1


			} // its

			for (int i = 0; i < VEC_NUM; i++)
			{
				ucoe[i].at(*it0) = val[i];
			}

		} // it0

	} // it

}


// function to compute point value at pos[D] based on Alpt basis
// system case

void LagrInterpolation::eval_point_val_Alpt_Lag(std::vector<double> & pos, std::vector<int> & m1, std::vector<double> & val)
{
	for (int i = 0; i < VEC_NUM; i++)
	{
		val[i] = 0.;
	}

	int k;

	for (auto it = dgsolution_ptr->dg.begin(); it != dgsolution_ptr->dg.end(); it++)
	{
		const std::vector<int> & l = it->second.level;
		const std::vector<int> & j = it->second.suppt;

		std::vector<std::vector<int>> & order = it->second.order_local_alpt;

		std::vector< VecMultiD<double> > & ucoe = it->second.ucoe_alpt;

		std::vector<double> & xl = it->second.xl;
		std::vector<double> & xr = it->second.xr;

		std::vector<int> p(DIM);

		k = 0;

		// check whether pos[D] is in this element

		for (int d = 0; d < DIM; d++)
		{
			if (pos[d] < xl[d] || pos[d] > xr[d])
			{
				k = 1;
			}
		}

		// if k=0, this means pos[D] is in this element
		// then do the following step

		if (k == 1) continue;

		int m2 = 0;

		for (auto it0 = order.begin(); it0 != order.end(); it0++)
		{
			double tep = 1.0;

			for (int d = 0; d < DIM; d++)
			{
				p[d] = (*it0)[d];

				m2 = hash_key1d(l[d], j[d]) * (AlptBasis::PMAX + 1) + p[d];

				// tep = tep * dgsolution_ptr->all_bas.at(l[d], j[d], p[d]).val(pos[d], 0, 0); 

				tep = tep * Lag_pt_Alpt_1D[m1[d]][m2];

			}

			for (int i = 0; i < VEC_NUM; i++)
			{
				val[i] += ucoe[i].at(*it0) * tep;
			}

		}

	}

}

// function to compute first derivative value at pos[D] based on Alpt basis
// scalar case 
// u_x, u_y, ...

void LagrInterpolation::eval_der1_val_Alpt_Lag(std::vector<double> & pos, std::vector<int> & m1, int d0, double & val)
{
	val = 0.;

	int k;

	for (auto it = dgsolution_ptr->dg.begin(); it != dgsolution_ptr->dg.end(); it++)
	{
		const std::vector<int> & l = it->second.level;
		const std::vector<int> & j = it->second.suppt;

		std::vector<std::vector<int>> & order = it->second.order_local_alpt;

		std::vector< VecMultiD<double> > & ucoe = it->second.ucoe_alpt;

		std::vector<double> & xl = it->second.xl;
		std::vector<double> & xr = it->second.xr;

		std::vector<int> p(DIM);

		k = 0;

		// check whether pos[D] is in this element

		for (int d = 0; d < DIM; d++)
		{
			if (pos[d] < xl[d] || pos[d] > xr[d])
			{
				k = 1;
			}
		}

		// if k=0, this means pos[D] is in this element
		// then do the following step

		if (k == 1) continue;

		int m2 = 0;

		for (auto it0 = order.begin(); it0 != order.end(); it0++)
		{
			double tep = 1.0;

			for (int d = 0; d < DIM; d++)
			{
				p[d] = (*it0)[d];

				m2 = hash_key1d(l[d], j[d]) * (AlptBasis::PMAX + 1) + p[d];

				// tep = tep * dgsolution_ptr->all_bas.at(l[d], j[d], p[d]).val(pos[d], 0, 0); 

				if (d == d0)
				{
					tep = tep * Lag_pt_Alpt_1D_d1[m1[d]][m2];
				}
				else
				{
					tep = tep * Lag_pt_Alpt_1D[m1[d]][m2];
				}

			}

			// for (int i = 0; i < VEC_NUM; i++)
			// {
			// 	val[i] += ucoe[i].at(*it0) * tep;
			// }

			val += ucoe[0].at(*it0) * tep;

		}

	}

}

//////////////////////////---->for Hermite interpolation

// compute the value of 1D hermite interpolation point with Alpt basis

void HermInterpolation::eval_Her_pt_at_Alpt_1D()
{
	int n1 = pow_int(2, dgsolution_ptr->NMAX) * (HermBasis::PMAX + 1);

	int n2 = pow_int(2, dgsolution_ptr->NMAX) * (AlptBasis::PMAX + 1);

	Her_pt_Alpt_1D.resize(n1);

	for (int i = 0; i < n1; i++)
	{
		Her_pt_Alpt_1D[i].resize(n2);
	}

	///////////////////

	int m1 = 0;
	int m2 = 0;

	// index for point
	int q = 0;
	// index for function, 1st-order derivative or 2nd-order derivative
	int m = 0;
	// location of hermite interpolation pt
	double pos = 0.;

	double xl = 0.;
	double xr = 0.;

	// loop for hermite bases
	for (int k = 0; k < dgsolution_ptr->NMAX + 1; k++)
	{
		int p2k = std::max(pow_int(2, k), 2);

		for (int i = 1; i < p2k; i += 2)
		{
			int a = hash_key1d(k, i) * (HermBasis::PMAX + 1);

			for (int p1 = 0; p1 < HermBasis::PMAX + 1; p1++)
			{
				deg_pt_deri_1d(p1, q, m);

				pos = dgsolution_ptr->all_bas_Her.at(k, i, p1).intep_pt;

				// index for 1st dgsolution_ptr->DIMension of matrix Her_pt_Alpt_1D
				m1 = a + p1;

				// loop for Alpt bases 
				for (int l = 0; l < dgsolution_ptr->NMAX + 1; l++)
				{
					int p2l = std::max(pow_int(2, l), 2);

					for (int j = 1; j < p2l; j += 2)
					{
						int b = hash_key1d(l, j) * (AlptBasis::PMAX + 1);

						for (int p2 = 0; p2 < AlptBasis::PMAX + 1; p2++)
						{
							AlptBasis const &bas = dgsolution_ptr->all_bas.at(l, j, p2);

							xl = bas.supp_interv[0];

							xr = bas.supp_interv[1];

							// index for 2nd dgsolution_ptr->DIMension of matrix Her_pt_Alpt_1D
							m2 = b + p2;

							if (pos < xl || pos > xr)
							{
								Her_pt_Alpt_1D[m1][m2] = 0.;
							}
							else
							{
								Her_pt_Alpt_1D[m1][m2] = bas.val(pos, m, 0);
							}

						}

					}

				}


			}

		}

	}
	//////////////////

}

// compute numerical solution at Hermite basis interpolation points

void HermInterpolation::eval_up_Her()
{

	for (auto it = dgsolution_ptr->dg.begin(); it != dgsolution_ptr->dg.end(); it++)
	{
		const std::vector<int> & l = it->second.level;
		const std::vector<int> & j = it->second.suppt;

		std::vector<std::vector<int>> & order = it->second.order_local_intp;

		std::vector< VecMultiD<double> > & up = it->second.up_intp;
		// VecMultiD<double> & fp = it->second.fp_intp;

		std::vector<double> pos(DIM);

		std::vector<int> p(DIM);

		std::vector<int> m1(DIM);

		////////////// compute value of function or derivative for u

		for (auto it0 = order.begin(); it0 != order.end(); it0++)
		{
			for (int d = 0; d < DIM; d++)
			{
				p[d] = (*it0)[d];

				m1[d] = hash_key1d(l[d], j[d]) * (HermBasis::PMAX + 1) + p[d];

				pos[d] = dgsolution_ptr->all_bas_Her.at(l[d], j[d], p[d]).intep_pt;

			}

			std::vector<double> val(VEC_NUM, 0.);

			eval_point_val_Alpt_Her(pos, m1, val);

			for (int i = 0; i < VEC_NUM; i++)
			{
				up[i].at(*it0) = val[i];
			}

		}

	}

}

//compute numerical solution for f(u) at Hermite basis interpolation points
// func[i][j]: 
// i: the number of vector
// j: j-th dgsolution_ptr->DIMension

// func_d1[i][j][k]: 
// i: the number of vector
// j: j-th dgsolution_ptr->DIMension
// k=0: 1st_order derivative ------> fu
// k=1: 1st_order derivative  -----> fv

// func_d2[i][j][k][l]: 
// i: the number of vector
// j: j-th dgsolution_ptr->DIMension
// k=0, l=0: 2nd_order derivative ------> fuu
// k=0, l=1: 2nd_order derivative ------> fuv
// k=1, l=0: 2nd_order derivative ------> fvu
// k=1, l=1: 2nd_order derivative ------> fvv

void HermInterpolation::eval_fp_Her_2D(std::function<double(std::vector<double>, int, int)> func,
	std::function<double(std::vector<double>, int, int, int)> func_d1,
	std::function<double(std::vector<double>, int, int, int, int)> func_d2,
	std::vector< std::vector<bool> > is_intp)
{
	assert(dgsolution_ptr->DIM == 2 && HermBasis::PMAX == 3);

	//////////////// compute value of function or derivative for f(u) in 2D

	for (auto it = dgsolution_ptr->dg.begin(); it != dgsolution_ptr->dg.end(); it++)
	{
		const std::vector<int> & l = it->second.level;
		const std::vector<int> & j = it->second.suppt;

		std::vector<std::vector<int>> & order = it->second.order_local_intp;

		std::vector<VecMultiD<double> > & up = it->second.up_intp;
		// VecMultiD<double> & fp = it->second.fp_intp;
		std::vector< std::vector< VecMultiD<double> > > & fp = it->second.fp_intp;

		std::vector<double> pos(DIM);

		std::vector<int> p(DIM);

		// index for point
		std::vector<int> q(DIM);

		// index for function or derivative
		std::vector<int> k(DIM);

		////////////// 

		for (auto it0 = order.begin(); it0 != order.end(); it0++)
		{

			for (int d = 0; d < DIM; ++d) {

				deg_pt_deri_1d((*it0)[d], q[d], k[d]);
			}

			//////////// case by case

			if (k[0] == 0 && k[1] == 0) // f
			{

				// store the vector of solution
				std::vector<double> u(VEC_NUM, 0.);

				for (int i = 0; i < VEC_NUM; i++)
				{
					u[i] = up[i].at(*it0);
				}

				for (int i = 0; i < VEC_NUM; i++)
				{
					for (int d = 0; d < DIM; d++)
					{
						if (is_intp[i][d] == true)
						{
							fp[i][d].at(*it0) = func(u, i, d);
						}

						// fp[i][d].at(*it0) = func(u, i, d);
					}
				}

			}
			else if (k[0] == 1 && k[1] == 0) // fx
			{
				// find u
				p[0] = (*it0)[0] - 2;
				p[1] = (*it0)[1];

				std::vector<double> u(VEC_NUM, 0.);

				for (int i = 0; i < VEC_NUM; i++)
				{
					u[i] = up[i].at(p);
				}

				// double u = up.at(p);

				//////
				for (int i = 0; i < VEC_NUM; i++)
				{
					for (int d = 0; d < DIM; d++)
					{
						if (is_intp[i][d] == true)
						{
							fp[i][d].at(*it0) = 0.;

							for (int i1 = 0; i1 < VEC_NUM; i1++) // fu, fv, fw, ...
							{
								fp[i][d].at(*it0) += func_d1(u, i, d, i1) * up[i1].at(*it0);
							}
						}

						// fp[i][d].at(*it0) = 0.;

						// for (int i1 = 0; i1 < VEC_NUM; i1++) // fu, fv, fw, ...
						// {
						// 	fp[i][d].at(*it0) += func_d1(u, i, d, i1) * up[i1].at(*it0);
						// }

					}
				}

			}
			else if (k[0] == 0 && k[1] == 1) // fy
			{
				// find u
				p[0] = (*it0)[0];
				p[1] = (*it0)[1] - 2;

				std::vector<double> u(VEC_NUM, 0.);

				for (int i = 0; i < VEC_NUM; i++)
				{
					u[i] = up[i].at(p);
				}

				// double u = up.at(p);

				//////
				for (int i = 0; i < VEC_NUM; i++)
				{
					for (int d = 0; d < DIM; d++)
					{
						if (is_intp[i][d] == true)
						{
							fp[i][d].at(*it0) = 0.;

							for (int i1 = 0; i1 < VEC_NUM; i1++) // fu, fv, fw, ...
							{
								fp[i][d].at(*it0) += func_d1(u, i, d, i1) * up[i1].at(*it0);
							}
						}

						// fp[i][d].at(*it0) = 0.;

						// for (int i1 = 0; i1 < VEC_NUM; i1++) // fu, fv, fw, ...
						// {
						// 	fp[i][d].at(*it0) += func_d1(u, i, d, i1) * up[i1].at(*it0);
						// }

					}
				}

			}
			else if (k[0] == 1 && k[1] == 1) // fxy
			{

				// find u
				p[0] = (*it0)[0] - 2;
				p[1] = (*it0)[1] - 2;

				std::vector<double> u(VEC_NUM, 0.);

				for (int i = 0; i < VEC_NUM; i++)
				{
					u[i] = up[i].at(p);
				}

				// double u = up.at(p);

				// find ux
				p[0] = (*it0)[0];
				p[1] = (*it0)[1] - 2;

				std::vector<double> ux(VEC_NUM, 0.);

				for (int i = 0; i < VEC_NUM; i++)
				{
					ux[i] = up[i].at(p);
				}

				// double ux = up.at(p);

				// find uy
				p[0] = (*it0)[0] - 2;
				p[1] = (*it0)[1];

				std::vector<double> uy(VEC_NUM, 0.);

				for (int i = 0; i < VEC_NUM; i++)
				{
					uy[i] = up[i].at(p);
				}

				// double uy = up.at(p);

				//////
				for (int i = 0; i < VEC_NUM; i++)
				{
					for (int d = 0; d < DIM; d++)
					{
						if (is_intp[i][d] == true)
						{
							fp[i][d].at(*it0) = 0.;

							for (int i1 = 0; i1 < VEC_NUM; i1++)
							{
								for (int i2 = 0; i2 < VEC_NUM; i2++)
								{
									fp[i][d].at(*it0) += func_d2(u, i, d, i1, i2) * ux[i1] * uy[i2];
								}

							}

							for (int i1 = 0; i1 < dgsolution_ptr->VEC_NUM; i1++)
							{
								fp[i][d].at(*it0) += func_d1(u, i, d, i1) * up[i1].at(*it0);
							}
						}


						// fp[i][d].at(*it0) = 0.;

						// for (int i1 = 0; i1 < VEC_NUM; i1++)
						// {
						// 	for (int i2 = 0; i2 < VEC_NUM; i2++)
						// 	{
						// 		fp[i][d].at(*it0) += func_d2(u, i, d, i1, i2) * ux[i1] * uy[i2];
						// 	}

						// }

						// for (int i1 = 0; i1 < dgsolution_ptr->VEC_NUM; i1++)
						// {
						// 	fp[i][d].at(*it0) += func_d1(u, i, d, i1) * up[i1].at(*it0);
						// }

					}
				}

			}
			/////////////////////////////

		}

	}


}


//compute numerical solution for f(u) at Hermite basis interpolation points
// func[i][j]: 
// i: the number of vector
// j: j-th dgsolution_ptr->DIMension

// func_d1[i][j][k]: 
// i: the number of vector
// j: j-th dgsolution_ptr->DIMension
// k=0: 1st_order derivative ------> fu

// func_d2[i][j][k][l]: 
// i: the number of vector
// j: j-th dgsolution_ptr->DIMension
// k=0, l=0: 2nd_order derivative ------> fuu

// func_d3[i][j][k][l][m]: 
// i: the number of vector
// j: j-th dgsolution_ptr->DIMension
// k=0, l=0, m=0: 3rd_order derivative ------> fuuu

// func_d4[i][j][k][l][m][n]: 
// i: the number of vector
// j: j-th dgsolution_ptr->DIMension
// k=0, l=0, m=0, n=0: 4th_order derivative ------> fuuuu



void HermInterpolation::eval_fp_Her_2D_PMAX5_scalar(std::function<double(std::vector<double>, int, int)> func,
	std::function<double(std::vector<double>, int, int, int)> func_d1,
	std::function<double(std::vector<double>, int, int, int, int)> func_d2,
	std::function<double(std::vector<double>, int, int, int, int, int)> func_d3,
	std::function<double(std::vector<double>, int, int, int, int, int, int)> func_d4,
	std::vector< std::vector<bool> > is_intp)
{
	assert(dgsolution_ptr->DIM == 2 && HermBasis::PMAX == 5 && Interpolation::VEC_NUM == 1);

	//////////////// compute value of function or derivative for f(u) in 2D

	for (auto it = dgsolution_ptr->dg.begin(); it != dgsolution_ptr->dg.end(); it++)
	{
		const std::vector<int> & l = it->second.level;
		const std::vector<int> & j = it->second.suppt;

		std::vector<std::vector<int>> & order = it->second.order_local_intp;

		std::vector<VecMultiD<double> > & up = it->second.up_intp;
		// VecMultiD<double> & fp = it->second.fp_intp;
		std::vector< std::vector< VecMultiD<double> > > & fp = it->second.fp_intp;

		std::vector<double> pos(DIM);

		std::vector<int> p(DIM);

		// index for point
		std::vector<int> q(DIM);

		// index for function or derivative
		std::vector<int> k(DIM);

		////////////// 

		for (auto it0 = order.begin(); it0 != order.end(); it0++)
		{

			for (int d = 0; d < DIM; ++d) {

				deg_pt_deri_1d((*it0)[d], q[d], k[d]);
			}

			//////////// case by case

			if (k[0] == 0 && k[1] == 0) // f
			{

				// store the vector of solution
				std::vector<double> u(VEC_NUM, 0.);

				for (int i = 0; i < VEC_NUM; i++)
				{
					u[i] = up[i].at(*it0);
				}

				for (int i = 0; i < VEC_NUM; i++)
				{
					for (int d = 0; d < DIM; d++)
					{
						if (is_intp[i][d] == true)
						{
							fp[i][d].at(*it0) = func(u, i, d);
						}

						// fp[i][d].at(*it0) = func(u, i, d);
					}
				}

			}
			else if (k[0] == 1 && k[1] == 0) // fx
			{
				// find u
				p[0] = (*it0)[0] - 2;
				p[1] = (*it0)[1];

				std::vector<double> u(VEC_NUM, 0.);

				for (int i = 0; i < VEC_NUM; i++)
				{
					u[i] = up[i].at(p);
				}

				// double u = up.at(p);

				//////
				for (int i = 0; i < VEC_NUM; i++)
				{
					for (int d = 0; d < DIM; d++)
					{
						if (is_intp[i][d] == true)
						{
							fp[i][d].at(*it0) = 0.;

							// for (int i1 = 0; i1 < VEC_NUM; i1++) // fu, fv, fw, ...
							// {
							// 	fp[i][d].at(*it0) += func_d1(u, i, d, i1) * up[i1].at(*it0);
							// }

							fp[i][d].at(*it0) += func_d1(u, i, d, 0) * up[0].at(*it0);

						}

					}
				}

			}
			else if (k[0] == 0 && k[1] == 1) // fy
			{
				// find u
				p[0] = (*it0)[0];
				p[1] = (*it0)[1] - 2;

				std::vector<double> u(VEC_NUM, 0.);

				for (int i = 0; i < VEC_NUM; i++)
				{
					u[i] = up[i].at(p);
				}

				// double u = up.at(p);

				//////
				for (int i = 0; i < VEC_NUM; i++)
				{
					for (int d = 0; d < DIM; d++)
					{
						if (is_intp[i][d] == true)
						{
							fp[i][d].at(*it0) = 0.;

							// for (int i1 = 0; i1 < VEC_NUM; i1++) // fu, fv, fw, ...
							// {
							// 	fp[i][d].at(*it0) += func_d1(u, i, d, i1) * up[i1].at(*it0);
							// }

							fp[i][d].at(*it0) += func_d1(u, i, d, 0) * up[0].at(*it0);
						}

					}
				}

			}
			else if (k[0] == 1 && k[1] == 1) // fxy
			{

				// find u
				p[0] = (*it0)[0] - 2;
				p[1] = (*it0)[1] - 2;

				std::vector<double> u(VEC_NUM, 0.);

				for (int i = 0; i < VEC_NUM; i++)
				{
					u[i] = up[i].at(p);
				}

				// double u = up.at(p);

				// find ux
				p[0] = (*it0)[0];
				p[1] = (*it0)[1] - 2;

				std::vector<double> ux(VEC_NUM, 0.);

				for (int i = 0; i < VEC_NUM; i++)
				{
					ux[i] = up[i].at(p);
				}

				// double ux = up.at(p);

				// find uy
				p[0] = (*it0)[0] - 2;
				p[1] = (*it0)[1];

				std::vector<double> uy(VEC_NUM, 0.);

				for (int i = 0; i < VEC_NUM; i++)
				{
					uy[i] = up[i].at(p);
				}

				// double uy = up.at(p);

				//////
				for (int i = 0; i < VEC_NUM; i++)
				{
					for (int d = 0; d < DIM; d++)
					{
						if (is_intp[i][d] == true)
						{
							fp[i][d].at(*it0) = 0.;

							// for (int i1 = 0; i1 < VEC_NUM; i1++)
							// {
							// 	for (int i2 = 0; i2 < VEC_NUM; i2++)
							// 	{
							// 		fp[i][d].at(*it0) += func_d2(u, i, d, i1, i2) * ux[i1] * uy[i2];
							// 	}

							// }

							fp[i][d].at(*it0) += func_d2(u, i, d, 0, 0) * ux[0] * uy[0];

							// for (int i1 = 0; i1 < dgsolution_ptr->VEC_NUM; i1++)
							// {
							// 	fp[i][d].at(*it0) += func_d1(u, i, d, i1) * up[i1].at(*it0);
							// }

							fp[i][d].at(*it0) += func_d1(u, i, d, 0) * up[0].at(*it0);

						}

					}
				}

			}
			else if (k[0] == 2 && k[1] == 0) // fxx
			{

				// find u
				p[0] = (*it0)[0] - 4;
				p[1] = (*it0)[1];

				std::vector<double> u(VEC_NUM, 0.);

				for (int i = 0; i < VEC_NUM; i++)
				{
					u[i] = up[i].at(p);
				}

				// find ux
				p[0] = (*it0)[0] - 2;
				p[1] = (*it0)[1];

				std::vector<double> ux(VEC_NUM, 0.);

				for (int i = 0; i < VEC_NUM; i++)
				{
					ux[i] = up[i].at(p);
				}

				//////
				for (int i = 0; i < VEC_NUM; i++)
				{
					for (int d = 0; d < DIM; d++)
					{
						if (is_intp[i][d] == true)
						{
							fp[i][d].at(*it0) = 0.;

							fp[i][d].at(*it0) += func_d2(u, i, d, 0, 0) * ux[0] * ux[0];

							fp[i][d].at(*it0) += func_d1(u, i, d, 0) * up[0].at(*it0);

						}

					}
				}

			}
			else if (k[0] == 0 && k[1] == 2) // fyy
			{

				// find u
				p[0] = (*it0)[0];
				p[1] = (*it0)[1] - 4;

				std::vector<double> u(VEC_NUM, 0.);

				for (int i = 0; i < VEC_NUM; i++)
				{
					u[i] = up[i].at(p);
				}

				// find uy
				p[0] = (*it0)[0];
				p[1] = (*it0)[1] - 2;

				std::vector<double> uy(VEC_NUM, 0.);

				for (int i = 0; i < VEC_NUM; i++)
				{
					uy[i] = up[i].at(p);
				}

				//////
				for (int i = 0; i < VEC_NUM; i++)
				{
					for (int d = 0; d < DIM; d++)
					{
						if (is_intp[i][d] == true)
						{
							fp[i][d].at(*it0) = 0.;

							fp[i][d].at(*it0) += func_d2(u, i, d, 0, 0) * uy[0] * uy[0];

							fp[i][d].at(*it0) += func_d1(u, i, d, 0) * up[0].at(*it0);

						}

					}
				}

			}
			else if (k[0] == 1 && k[1] == 2) // fxyy
			{

				// find u
				p[0] = (*it0)[0] - 2;
				p[1] = (*it0)[1] - 4;

				std::vector<double> u(VEC_NUM, 0.);

				for (int i = 0; i < VEC_NUM; i++)
				{
					u[i] = up[i].at(p);
				}

				// find ux
				p[0] = (*it0)[0];
				p[1] = (*it0)[1] - 4;

				std::vector<double> ux(VEC_NUM, 0.);

				for (int i = 0; i < VEC_NUM; i++)
				{
					ux[i] = up[i].at(p);
				}

				// find uy
				p[0] = (*it0)[0] - 2;
				p[1] = (*it0)[1] - 2;

				std::vector<double> uy(VEC_NUM, 0.);

				for (int i = 0; i < VEC_NUM; i++)
				{
					uy[i] = up[i].at(p);
				}

				// find uxy
				p[0] = (*it0)[0];
				p[1] = (*it0)[1] - 2;

				std::vector<double> uxy(VEC_NUM, 0.);

				for (int i = 0; i < VEC_NUM; i++)
				{
					uxy[i] = up[i].at(p);
				}

				// find uyy
				p[0] = (*it0)[0] - 2;
				p[1] = (*it0)[1];

				std::vector<double> uyy(VEC_NUM, 0.);

				for (int i = 0; i < VEC_NUM; i++)
				{
					uyy[i] = up[i].at(p);
				}

				//////
				for (int i = 0; i < VEC_NUM; i++)
				{
					for (int d = 0; d < DIM; d++)
					{
						if (is_intp[i][d] == true)
						{
							fp[i][d].at(*it0) = 0.;

							fp[i][d].at(*it0) += func_d3(u, i, d, 0, 0, 0) * ux[0] * uy[0] * uy[0];

							fp[i][d].at(*it0) += func_d2(u, i, d, 0, 0) * (ux[0] * uyy[0] + 2.0 * uy[0] * uxy[0]);

							fp[i][d].at(*it0) += func_d1(u, i, d, 0) * up[0].at(*it0);

						}

					}
				}

			}
			else if (k[0] == 2 && k[1] == 1) // fxxy
			{

				// find u
				p[0] = (*it0)[0] - 4;
				p[1] = (*it0)[1] - 2;

				std::vector<double> u(VEC_NUM, 0.);

				for (int i = 0; i < VEC_NUM; i++)
				{
					u[i] = up[i].at(p);
				}

				// find ux
				p[0] = (*it0)[0] - 2;
				p[1] = (*it0)[1] - 2;

				std::vector<double> ux(VEC_NUM, 0.);

				for (int i = 0; i < VEC_NUM; i++)
				{
					ux[i] = up[i].at(p);
				}

				// find uy
				p[0] = (*it0)[0] - 4;
				p[1] = (*it0)[1];

				std::vector<double> uy(VEC_NUM, 0.);

				for (int i = 0; i < VEC_NUM; i++)
				{
					uy[i] = up[i].at(p);
				}

				// find uxy
				p[0] = (*it0)[0] - 2;
				p[1] = (*it0)[1];

				std::vector<double> uxy(VEC_NUM, 0.);

				for (int i = 0; i < VEC_NUM; i++)
				{
					uxy[i] = up[i].at(p);
				}

				// find uxx
				p[0] = (*it0)[0];
				p[1] = (*it0)[1] - 2;

				std::vector<double> uxx(VEC_NUM, 0.);

				for (int i = 0; i < VEC_NUM; i++)
				{
					uxx[i] = up[i].at(p);
				}

				//////
				for (int i = 0; i < VEC_NUM; i++)
				{
					for (int d = 0; d < DIM; d++)
					{
						if (is_intp[i][d] == true)
						{
							fp[i][d].at(*it0) = 0.;

							fp[i][d].at(*it0) += func_d3(u, i, d, 0, 0, 0) * ux[0] * ux[0] * uy[0];

							fp[i][d].at(*it0) += func_d2(u, i, d, 0, 0) * (uy[0] * uxx[0] + 2.0 * ux[0] * uxy[0]);

							fp[i][d].at(*it0) += func_d1(u, i, d, 0) * up[0].at(*it0);

						}

					}
				}

			}
			else if (k[0] == 2 && k[1] == 2) // fxxyy
			{

				// find u
				p[0] = (*it0)[0] - 4;
				p[1] = (*it0)[1] - 4;

				std::vector<double> u(VEC_NUM, 0.);

				for (int i = 0; i < VEC_NUM; i++)
				{
					u[i] = up[i].at(p);
				}

				// find ux
				p[0] = (*it0)[0] - 2;
				p[1] = (*it0)[1] - 4;

				std::vector<double> ux(VEC_NUM, 0.);

				for (int i = 0; i < VEC_NUM; i++)
				{
					ux[i] = up[i].at(p);
				}

				// find uy
				p[0] = (*it0)[0] - 4;
				p[1] = (*it0)[1] - 2;

				std::vector<double> uy(VEC_NUM, 0.);

				for (int i = 0; i < VEC_NUM; i++)
				{
					uy[i] = up[i].at(p);
				}

				// find uxy
				p[0] = (*it0)[0] - 2;
				p[1] = (*it0)[1] - 2;

				std::vector<double> uxy(VEC_NUM, 0.);

				for (int i = 0; i < VEC_NUM; i++)
				{
					uxy[i] = up[i].at(p);
				}

				// find uxx
				p[0] = (*it0)[0];
				p[1] = (*it0)[1] - 4;

				std::vector<double> uxx(VEC_NUM, 0.);

				for (int i = 0; i < VEC_NUM; i++)
				{
					uxx[i] = up[i].at(p);
				}

				// find uyy
				p[0] = (*it0)[0] - 4;
				p[1] = (*it0)[1];

				std::vector<double> uyy(VEC_NUM, 0.);

				for (int i = 0; i < VEC_NUM; i++)
				{
					uyy[i] = up[i].at(p);
				}

				// find uxyy
				p[0] = (*it0)[0] - 2;
				p[1] = (*it0)[1];

				std::vector<double> uxyy(VEC_NUM, 0.);

				for (int i = 0; i < VEC_NUM; i++)
				{
					uxyy[i] = up[i].at(p);
				}

				// find uxxy
				p[0] = (*it0)[0];
				p[1] = (*it0)[1] - 2;

				std::vector<double> uxxy(VEC_NUM, 0.);

				for (int i = 0; i < VEC_NUM; i++)
				{
					uxxy[i] = up[i].at(p);
				}

				//////
				for (int i = 0; i < VEC_NUM; i++)
				{
					for (int d = 0; d < DIM; d++)
					{
						if (is_intp[i][d] == true)
						{
							fp[i][d].at(*it0) = 0.;

							double c2 = uxx[0] * uyy[0] + 2.0 * (uxxy[0] * uy[0] + ux[0] * uxyy[0] + uxy[0] * uxy[0]);

							double c3 = uxx[0] * uy[0] * uy[0] + uyy[0] * ux[0] * ux[0] + 4.0 * uxy[0] * ux[0] * uy[0];

							fp[i][d].at(*it0) += func_d4(u, i, d, 0, 0, 0, 0) * ux[0] * ux[0] * uy[0] * uy[0];

							fp[i][d].at(*it0) += func_d3(u, i, d, 0, 0, 0) * c3;

							fp[i][d].at(*it0) += func_d2(u, i, d, 0, 0) * c2;

							fp[i][d].at(*it0) += func_d1(u, i, d, 0) * up[0].at(*it0);

						}

					}
				}

			}
			/////////////////////////////

		}

	}


}


//compute numerical solution for f(u) at Hermite basis interpolation points
// func[i][j]: 
// i: the number of vector
// j: j-th dgsolution_ptr->DIMension

// func_d1[i][j][k]: 
// i: the number of vector
// j: j-th dgsolution_ptr->DIMension
// k=0: 1st_order derivative ------> fu
// k=1: 1st_order derivative ------> fv


void HermInterpolation::eval_fp_Her_1D(std::function<double(std::vector<double>, int, int)> func,
	std::function<double(std::vector<double>, int, int, int)> func_d1,
	std::vector< std::vector<bool> > is_intp)
{
	assert(dgsolution_ptr->DIM == 1 && HermBasis::PMAX == 3);

	//////////////// compute value of function or derivative for f(u) in 1D

	for (auto it = dgsolution_ptr->dg.begin(); it != dgsolution_ptr->dg.end(); it++)
	{
		const std::vector<int> & l = it->second.level;
		const std::vector<int> & j = it->second.suppt;

		std::vector<std::vector<int>> & order = it->second.order_local_intp;

		std::vector<VecMultiD<double> > & up = it->second.up_intp;
		// VecMultiD<double> & fp = it->second.fp_intp;
		std::vector< std::vector< VecMultiD<double> > > & fp = it->second.fp_intp;

		std::vector<double> pos(DIM);

		std::vector<int> p(DIM);

		// index for point
		std::vector<int> q(DIM);

		// index for function or derivative
		std::vector<int> k(DIM);

		////////////// 

		for (auto it0 = order.begin(); it0 != order.end(); it0++)
		{

			for (int d = 0; d < DIM; ++d) {

				deg_pt_deri_1d((*it0)[d], q[d], k[d]);
			}

			//////////// case by case

			if (k[0] == 0) // f
			{

				// store the vector of solution
				std::vector<double> u(VEC_NUM, 0.);

				for (int i = 0; i < VEC_NUM; i++)
				{
					u[i] = up[i].at(*it0);
				}

				for (int i = 0; i < VEC_NUM; i++)
				{
					for (int d = 0; d < DIM; d++)
					{
						if (is_intp[i][d] == true)
						{
							fp[i][d].at(*it0) = func(u, i, d);
						}

						// fp[i][d].at(*it0) = func(u, i, d);
					}
				}

			}
			else
			{
				// find u
				p[0] = (*it0)[0] - 2;

				std::vector<double> u(VEC_NUM, 0.);

				for (int i = 0; i < VEC_NUM; i++)
				{
					u[i] = up[i].at(p);
				}

				////////////////
				for (int i = 0; i < VEC_NUM; i++)
				{
					for (int d = 0; d < DIM; d++)
					{
						if (is_intp[i][d] == true)
						{
							fp[i][d].at(*it0) = 0.;

							for (int i1 = 0; i1 < VEC_NUM; i1++) // fu, fv, fw, ...
							{
								fp[i][d].at(*it0) += func_d1(u, i, d, i1) * up[i1].at(*it0);
							}
						}

					}
				}

			}

			/////////////////////////////

		}

	}


}


// read function value at Hermite point from exact function directly

void HermInterpolation::eval_up_exact_Her(std::function<double(std::vector<double>, int, std::vector<int>)> func)
{
	for (auto it = dgsolution_ptr->dg.begin(); it != dgsolution_ptr->dg.end(); it++)
	{
		const std::vector<int> & l = it->second.level;
		const std::vector<int> & j = it->second.suppt;

		std::vector<std::vector<int>> & order = it->second.order_local_intp;

		std::vector<VecMultiD<double> > & up = it->second.up_intp;

		std::vector<double> pos(DIM);

		std::vector<int> p(DIM);

		// index for point
		std::vector<int> q(DIM);
		// index for function, 1st-order derivative or 2nd-order derivative
		std::vector<int> m(DIM);

		for (auto it0 = order.begin(); it0 != order.end(); it0++)
		{
			for (int d = 0; d < DIM; d++)
			{
				p[d] = (*it0)[d];

				deg_pt_deri_1d(p[d], q[d], m[d]);

				pos[d] = dgsolution_ptr->all_bas_Her.at(l[d], j[d], p[d]).intep_pt;
			}

			for (int i = 0; i < VEC_NUM; i++)
			{
				up[i].at(*it0) = func(pos, i, m);
			}

		}

	}
}

// read function value at Hermite point from exact function directly for adaptive interpolation initialization

void HermInterpolation::eval_up_exact_ada_Her(std::function<double(std::vector<double>, int, std::vector<int>)> func)
{
	for (auto it = dgsolution_ptr->dg.begin(); it != dgsolution_ptr->dg.end(); it++)
	{
		// just do for new added elements
		if (it->second.new_add == false) continue;

		const std::vector<int> & l = it->second.level;
		const std::vector<int> & j = it->second.suppt;

		std::vector<std::vector<int>> & order = it->second.order_local_intp;

		std::vector<VecMultiD<double> > & up = it->second.up_intp;

		std::vector<double> pos(DIM);

		std::vector<int> p(DIM);

		// index for point
		std::vector<int> q(DIM);
		// index for function, 1st-order derivative or 2nd-order derivative
		std::vector<int> m(DIM);

		for (auto it0 = order.begin(); it0 != order.end(); it0++)
		{
			for (int d = 0; d < DIM; d++)
			{
				p[d] = (*it0)[d];

				deg_pt_deri_1d(p[d], q[d], m[d]);

				pos[d] = dgsolution_ptr->all_bas_Her.at(l[d], j[d], p[d]).intep_pt;
			}

			for (int i = 0; i < VEC_NUM; i++)
			{
				up[i].at(*it0) = func(pos, i, m);
			}

		}

	}
}

// from degree to get the index of point and derivative 

void HermInterpolation::deg_pt_deri_1d(const int pp, int & p, int & l)
{
	// p: index for point
	// l: index for function or derivative

	// transform between pp and p & l
	if (pp == 0)
	{
		p = 0;
		l = 0;
	}
	else if (pp == 1)
	{
		p = 1;
		l = 0;
	}
	else if (pp == 2)
	{
		p = 0;
		l = 1;
	}
	else if (pp == 3)
	{
		p = 1;
		l = 1;
	}
	else if (pp == 4)
	{
		p = 0;
		l = 2;
	}
	else if (pp == 5)
	{
		p = 1;
		l = 2;
	}
	else
	{
		std::cout << "pp out of range in function deg_pt_deri_1d" << std::endl;
		exit(1);
	}

}


// find points and weights which will affact the coefficients of the Hermite interpolation points

void HermInterpolation::set_pts_wts_1d_ada_Her(const int l, const int j)
{

	const int P1 = HermBasis::PMAX + 1;

	double pos1;

	std::vector<double> p_pos(P1);

	std::vector<int> p_k(P1), p_i(P1), p_num(P1);

	double xleft, xright;

	int lj = hash_key1d(l, j);

	xleft = dgsolution_ptr->all_bas_Her.at(l, j, 0).supp_interv[0];

	xright = dgsolution_ptr->all_bas_Her.at(l, j, 0).supp_interv[1];

	pwts a;

	if (l > 0) {

		int ic = 0;

		for (int k = 0; k < l; ++k) {

			if (ic == P1) continue;

			int p2k = std::max(2, pow_int(2, k));

			for (int i = 1; i < p2k; i += 2) {

				for (int q0 = 0; q0 < P1; ++q0) {

					pos1 = dgsolution_ptr->all_bas_Her.at(k, i, q0).intep_pt;

					// double eps = 0.0;

					if (pos1 > xleft && pos1 < xright) {

						if (ic == P1)  std::cout << ic << std::endl;

						p_pos[ic] = pos1;

						p_k[ic] = k;

						p_i[ic] = i;

						p_num[ic] = q0;

						ic++;

					}
				}
				///////////////////
			}

		}

		//////////////////////

		for (int i1 = 0; i1 < P1; ++i1) { //means P1 points will affect

			a.p_k.push_back(p_k[i1]);

			a.p_i.push_back(p_i[i1]);

			a.p_num.push_back(p_num[i1]);
		}

		////////////compute wts

		a.wt.resize(P1);

		std::vector<double> pos(P1);

		for (int ic = 0; ic < P1; ++ic) {

			pos[ic] = p_pos[ic];
		}

		sort(pos.begin(), pos.end());

		double pl = pos[0];

		int i10 = 0;
		int i20 = 1;

		for (int p0 = 0; p0 < P1; ++p0) {

			i10 = 0;
			i20 = 1;

			for (int ic = 0; ic < P1; ++ic) {

				int ic0;

				if (std::fabs(p_pos[ic] - pl) < 1.0e-14)
				{

					ic0 = i10;

					i10 += 2;

				}
				// if(std::abs(p_pos[ic] - pr) < 1.0e-14)
				else
				{

					ic0 = i20;

					i20 += 2;

				}

				///////// find out function, 1st_order derivative or 2nd_order derivative

				// index of point for p0
				int q1;
				// index of function, 1st_order derivative or 2nd_order derivative for p0
				int l1;

				// index of point for ic0
				int q2;
				// index of function, 1st_order derivative or 2nd_order derivative for ic0
				int l2;

				deg_pt_deri_1d(p0, q1, l1);

				deg_pt_deri_1d(ic0, q2, l2);

				///////////////////

				double xt = HermBasis::intp_msh1[p0];

				a.wt[p0].push_back(-1.0 * std::pow(2, (l - 1) * (l1 - l2)) * dgsolution_ptr->all_bas_Her.at(0, 1, ic0).val(xt, l1, 0));

				//////////////////////
			}


		}

		/////////////////////

		pw1d.insert(std::make_pair(lj, a));

	}
}

// u: compute hierarchical coefficients of Hermite basis based on function value at Hermite point

void HermInterpolation::eval_up_to_coe_D_Her()
{
	int P1 = HermBasis::PMAX + 1;

	///////////////////

	for (auto it = dgsolution_ptr->dg.begin(); it != dgsolution_ptr->dg.end(); it++)
	{
		std::vector<std::vector<int>> & order = it->second.order_local_intp;

		std::vector<VecMultiD<double> > & up = it->second.up_intp;

		std::vector<VecMultiD<std::vector<double>> > & ucoe_1 = it->second.ucoe_intp_inter;

		for (auto it0 = order.begin(); it0 != order.end(); it0++)
		{
			for (int i = 0; i < VEC_NUM; i++)
			{
				ucoe_1[i].at(*it0)[0] = up[i].at(*it0);
			}

		}

	}

	////////////////////

	for (int d = 0; d < DIM; d++)
	{

		for (auto it = dgsolution_ptr->dg.begin(); it != dgsolution_ptr->dg.end(); it++)
		{
			const std::vector<int> & l = it->second.level;
			const std::vector<int> & j = it->second.suppt;

			std::vector<std::vector<int>> & order = it->second.order_local_intp;

			std::vector<VecMultiD<std::vector<double>> > & ucoe_1 = it->second.ucoe_intp_inter;

			//////////////////////////

			int lj = (it->second.order_elem)[d];

			auto itpw = pw1d.find(lj);

			if (itpw == pw1d.end() && lj > 0)
			{
				set_pts_wts_1d_ada_Her(l[d], j[d]);

				itpw = pw1d.find(lj);
			}

			for (auto it0 = order.begin(); it0 != order.end(); it0++)
			{
				std::vector<int> k(DIM), i(DIM), p(DIM), q(DIM);

				////////////////
				std::vector<double> coe(VEC_NUM, 0.);

				for (int i = 0; i < VEC_NUM; i++)
				{
					coe[i] = (ucoe_1[i].at(*it0))[d];
				}

				////////////////

				if (l[d] > 0)
				{
					for (int d0 = 0; d0 < DIM; d0++)
					{
						p[d0] = (*it0)[d0];
					}

					for (int ic = 0; ic < P1; ic++)
					{
						std::vector<int> mm;

						for (int d1 = 0; d1 < DIM; d1++)
						{
							if (d1 == d)
							{

								k[d] = itpw->second.p_k[ic];

								i[d] = itpw->second.p_i[ic];

								q[d] = itpw->second.p_num[ic];

								mm.push_back(itpw->second.p_num[ic]);

							}
							else
							{
								k[d1] = l[d1];

								i[d1] = j[d1];

								q[d1] = p[d1];

								mm.push_back(p[d1]);
							}

						}

						//////////////

						int hak = dgsolution_ptr->hash.hash_key(k, i);

						auto its = dgsolution_ptr->dg.find(hak);

						if (its == dgsolution_ptr->dg.end())
						{
							std::cout << "wrong" << std::endl;
						}

						for (int i = 0; i < VEC_NUM; i++)
						{
							coe[i] += itpw->second.wt[p[d]][ic] * ((its->second.ucoe_intp_inter)[i].at(mm))[d];
						}

						// coe += itpw->second.wt[p[d]][ic] * ((its->second.ucoe_intp_inter).at(mm))[d];

					}

				}

				for (int i = 0; i < VEC_NUM; i++)
				{
					(ucoe_1[i].at(*it0))[d + 1] = coe[i];
				}

			}
			/////////////////////////

		}

	}
	/////////////////

	for (auto it = dgsolution_ptr->dg.begin(); it != dgsolution_ptr->dg.end(); it++)
	{
		std::vector<VecMultiD<double> > & ucoe = it->second.ucoe_intp;

		std::vector<VecMultiD<std::vector<double>> > & ucoe_1 = it->second.ucoe_intp_inter;

		std::vector<std::vector<int>> & order = it->second.order_local_intp;

		for (auto it0 = order.begin(); it0 != order.end(); it0++)
		{
			for (int i = 0; i < VEC_NUM; i++)
			{
				ucoe[i].at(*it0) = (ucoe_1[i].at(*it0))[DIM];
			}

		}
	}

}

// u: compute hierarchical coefficients of Hermite basis based on function value at Hermite point
// for adaptive initialization

void HermInterpolation::eval_up_to_coe_D_ada_Her()
{
	int P1 = HermBasis::PMAX + 1;

	///////////////////

	for (auto it = dgsolution_ptr->dg.begin(); it != dgsolution_ptr->dg.end(); it++)
	{
		// just do for new added elements
		if (it->second.new_add == false) continue;

		std::vector<std::vector<int>> & order = it->second.order_local_intp;

		std::vector<VecMultiD<double> > & up = it->second.up_intp;

		std::vector<VecMultiD<std::vector<double>> > & ucoe_1 = it->second.ucoe_intp_inter;

		for (auto it0 = order.begin(); it0 != order.end(); it0++)
		{
			for (int i = 0; i < VEC_NUM; i++)
			{
				ucoe_1[i].at(*it0)[0] = up[i].at(*it0);
			}

		}

	}

	////////////////////

	for (int d = 0; d < DIM; d++)
	{

		for (auto it = dgsolution_ptr->dg.begin(); it != dgsolution_ptr->dg.end(); it++)
		{
			// just do for new added elements
			if (it->second.new_add == false) continue;

			const std::vector<int> & l = it->second.level;
			const std::vector<int> & j = it->second.suppt;

			std::vector<std::vector<int>> & order = it->second.order_local_intp;

			std::vector<VecMultiD<std::vector<double>> > & ucoe_1 = it->second.ucoe_intp_inter;

			//////////////////////////

			int lj = (it->second.order_elem)[d];

			auto itpw = pw1d.find(lj);

			if (itpw == pw1d.end() && lj > 0)
			{
				set_pts_wts_1d_ada_Her(l[d], j[d]);

				itpw = pw1d.find(lj);
			}

			for (auto it0 = order.begin(); it0 != order.end(); it0++)
			{
				std::vector<int> k(DIM), i(DIM), p(DIM), q(DIM);

				////////////////
				std::vector<double> coe(VEC_NUM, 0.);

				for (int i = 0; i < VEC_NUM; i++)
				{
					coe[i] = (ucoe_1[i].at(*it0))[d];
				}

				////////////////

				if (l[d] > 0)
				{
					for (int d0 = 0; d0 < DIM; d0++)
					{
						p[d0] = (*it0)[d0];
					}

					for (int ic = 0; ic < P1; ic++)
					{
						std::vector<int> mm;

						for (int d1 = 0; d1 < DIM; d1++)
						{
							if (d1 == d)
							{

								k[d] = itpw->second.p_k[ic];

								i[d] = itpw->second.p_i[ic];

								q[d] = itpw->second.p_num[ic];

								mm.push_back(itpw->second.p_num[ic]);

							}
							else
							{
								k[d1] = l[d1];

								i[d1] = j[d1];

								q[d1] = p[d1];

								mm.push_back(p[d1]);
							}

						}

						//////////////

						int hak = dgsolution_ptr->hash.hash_key(k, i);

						auto its = dgsolution_ptr->dg.find(hak);

						if (its == dgsolution_ptr->dg.end())
						{
							std::cout << "wrong" << std::endl;
						}

						for (int i = 0; i < VEC_NUM; i++)
						{
							coe[i] += itpw->second.wt[p[d]][ic] * ((its->second.ucoe_intp_inter)[i].at(mm))[d];
						}

						// coe += itpw->second.wt[p[d]][ic] * ((its->second.ucoe_intp_inter).at(mm))[d];

					}

				}

				for (int i = 0; i < VEC_NUM; i++)
				{
					(ucoe_1[i].at(*it0))[d + 1] = coe[i];
				}

			}
			/////////////////////////

		}

	}
	/////////////////

	for (auto it = dgsolution_ptr->dg.begin(); it != dgsolution_ptr->dg.end(); it++)
	{
		// just do for new added elements
		if (it->second.new_add == false) continue;

		std::vector<VecMultiD<double> > & ucoe = it->second.ucoe_intp;

		std::vector<VecMultiD<std::vector<double>> > & ucoe_1 = it->second.ucoe_intp_inter;

		std::vector<std::vector<int>> & order = it->second.order_local_intp;

		for (auto it0 = order.begin(); it0 != order.end(); it0++)
		{
			for (int i = 0; i < VEC_NUM; i++)
			{
				ucoe[i].at(*it0) = (ucoe_1[i].at(*it0))[DIM];
			}

		}
	}

}

// fu: compute hierarchical coefficients of Hermite basis based on function value at Hermite point 

void HermInterpolation::eval_fp_to_coe_D_Her(std::vector< std::vector<bool> > is_intp)
{
	int P1 = HermBasis::PMAX + 1;

	///////////////////

	for (auto it = dgsolution_ptr->dg.begin(); it != dgsolution_ptr->dg.end(); it++)
	{
		std::vector<std::vector<int>> & order = it->second.order_local_intp;

		std::vector< std::vector< VecMultiD<double> > > & fp = it->second.fp_intp;

		std::vector< std::vector< VecMultiD<std::vector<double>> > > & fucoe_1 = it->second.fucoe_intp_inter;

		for (auto it0 = order.begin(); it0 != order.end(); it0++)
		{
			for (int i = 0; i < VEC_NUM; i++)
			{
				for (int d = 0; d < DIM; d++)
				{
					if (is_intp[i][d] == true)
					{
						fucoe_1[i][d].at(*it0)[0] = fp[i][d].at(*it0);
					}
				}
			}

			// fucoe_1.at(*it0)[0] = fp.at(*it0);
		}

	}

	////////////////////

	for (int d = 0; d < DIM; d++)
	{

		for (auto it = dgsolution_ptr->dg.begin(); it != dgsolution_ptr->dg.end(); it++)
		{
			const std::vector<int> & l = it->second.level;
			const std::vector<int> & j = it->second.suppt;

			std::vector<std::vector<int>> & order = it->second.order_local_intp;

			std::vector< std::vector< VecMultiD<std::vector<double>> > > & fucoe_1 = it->second.fucoe_intp_inter;

			//////////////////////////

			int lj = (it->second.order_elem)[d];

			auto itpw = pw1d.find(lj);

			if (itpw == pw1d.end() && lj > 0)
			{
				set_pts_wts_1d_ada_Her(l[d], j[d]);

				itpw = pw1d.find(lj);
			}

			for (auto it0 = order.begin(); it0 != order.end(); it0++)
			{
				std::vector<int> k(DIM), i(DIM), p(DIM), q(DIM);

				////////////////

				std::vector<std::vector<double>> coe;

				coe.resize(VEC_NUM);

				for (int i = 0; i < VEC_NUM; i++)
				{
					coe[i].resize(DIM);

					for (int d2 = 0; d2 < DIM; d2++)
					{
						if (is_intp[i][d2] == true)
						{
							coe[i][d2] = (fucoe_1[i][d2].at(*it0))[d];
						}
					}
				}

				/////////////////////

				if (l[d] > 0)
				{
					for (int d0 = 0; d0 < DIM; d0++)
					{
						p[d0] = (*it0)[d0];
					}

					for (int ic = 0; ic < P1; ic++)
					{
						std::vector<int> mm;

						for (int d1 = 0; d1 < DIM; d1++)
						{
							if (d1 == d)
							{

								k[d] = itpw->second.p_k[ic];

								i[d] = itpw->second.p_i[ic];

								q[d] = itpw->second.p_num[ic];

								mm.push_back(itpw->second.p_num[ic]);

							}
							else
							{
								k[d1] = l[d1];

								i[d1] = j[d1];

								q[d1] = p[d1];

								mm.push_back(p[d1]);
							}

						}

						//////////////

						int hak = dgsolution_ptr->hash.hash_key(k, i);

						auto its = dgsolution_ptr->dg.find(hak);

						if (its == dgsolution_ptr->dg.end())
						{
							std::cout << "wrong" << std::endl;
						}

						for (int i = 0; i < VEC_NUM; i++)
						{
							for (int d2 = 0; d2 < DIM; d2++)
							{
								if (is_intp[i][d2] == true)
								{
									coe[i][d2] += itpw->second.wt[p[d]][ic] * ((its->second.fucoe_intp_inter)[i][d2].at(mm))[d];
								}
							}
						}

						// coe += itpw->second.wt[p[d]][ic] * ((its->second.fucoe_intp_inter).at(mm))[d];

					}

				}

				for (int i = 0; i < VEC_NUM; i++)
				{
					for (int d2 = 0; d2 < DIM; d2++)
					{
						if (is_intp[i][d2] == true)
						{
							(fucoe_1[i][d2].at(*it0))[d + 1] = coe[i][d2];
						}
					}
				}

				// (fucoe_1.at(*it0))[d+1] = coe;

			}
			/////////////////////////

		}

	}
	/////////////////

	for (auto it = dgsolution_ptr->dg.begin(); it != dgsolution_ptr->dg.end(); it++)
	{
		std::vector< std::vector< VecMultiD<double> > > & fucoe = it->second.fucoe_intp;

		std::vector< std::vector< VecMultiD<std::vector<double>> > > & fucoe_1 = it->second.fucoe_intp_inter;

		std::vector<std::vector<int>> & order = it->second.order_local_intp;

		for (auto it0 = order.begin(); it0 != order.end(); it0++)
		{
			for (int i = 0; i < VEC_NUM; i++)
			{
				for (int d2 = 0; d2 < DIM; d2++)
				{
					if (is_intp[i][d2] == true)
					{
						fucoe[i][d2].at(*it0) = (fucoe_1[i][d2].at(*it0))[DIM];
					}
				}
			}

			// fucoe.at(*it0) = (fucoe_1.at(*it0))[dgsolution_ptr->DIM];
		}
	}

}

// compute coefficients of Alpt basis from coefficients of Hermite basis

void HermInterpolation::eval_coe_alpt_from_coe_Her(const OperatorMatrix1D<HermBasis, AlptBasis> & matrix)
{
	for (auto it = dgsolution_ptr->dg.begin(); it != dgsolution_ptr->dg.end(); it++)
	{
		const std::vector<int> & l = it->second.level;
		const std::vector<int> & j = it->second.suppt;

		std::vector<std::vector<int>> & order = it->second.order_local_alpt;

		std::vector<VecMultiD<double> > & ucoe = it->second.ucoe_alpt;

		std::vector<double> & xl = it->second.xl;
		std::vector<double> & xr = it->second.xr;

		for (auto it0 = order.begin(); it0 != order.end(); it0++)
		{
			std::vector<int> p(DIM, 0);

			std::vector<int> m1(DIM, 0);

			std::vector<double> val(VEC_NUM, 0.);

			for (int d = 0; d < DIM; d++)
			{
				p[d] = (*it0)[d];

				m1[d] = hash_key1d(l[d], j[d]) * (AlptBasis::PMAX + 1) + p[d];
			}

			for (auto its = dgsolution_ptr->dg.begin(); its != dgsolution_ptr->dg.end(); its++)
			{
				const std::vector<int> & k = its->second.level;
				const std::vector<int> & i = its->second.suppt;

				std::vector<std::vector<int>> & order_s = its->second.order_local_intp;

				std::vector<VecMultiD<double> > & ucoe_intp = its->second.ucoe_intp;

				std::vector<double> & xl_s = its->second.xl;
				std::vector<double> & xr_s = its->second.xr;

				///////////////////

				int kk = 0;

				for (int d = 0; d < DIM; d++)
				{
					if (xl[d] > xr_s[d] || xl_s[d] > xr[d])
					{
						kk = 1;
					}
				}

				if (kk == 1) continue;

				/////////////////////

				for (auto it1 = order_s.begin(); it1 != order_s.end(); it1++)
				{
					std::vector<int> q(DIM, 0);

					std::vector<int> m2(DIM, 0);

					double tep = 1.0;

					for (int d = 0; d < DIM; d++)
					{
						q[d] = (*it1)[d];

						m2[d] = hash_key1d(k[d], i[d]) * (HermBasis::PMAX + 1) + q[d];

						tep = tep * matrix.u_v.at(m2[d], m1[d]); // since: row: Herm, col: Alpt 
					}

					for (int i = 0; i < VEC_NUM; i++)
					{
						val[i] += ucoe_intp[i].at(*it1) * tep;
					}

				} // it1


			} // its

			for (int i = 0; i < VEC_NUM; i++)
			{
				ucoe[i].at(*it0) = val[i];
			}

		} // it0

	} // it

}


//function to compute point value at pos[D] based on Alpt basis
// system case

void HermInterpolation::eval_point_val_Alpt_Her(std::vector<double> & pos, std::vector<int> & m1, std::vector<double> & val)
{
	for (int i = 0; i < dgsolution_ptr->VEC_NUM; i++)
	{
		val[i] = 0.;
	}

	int k = 0;

	for (auto it = dgsolution_ptr->dg.begin(); it != dgsolution_ptr->dg.end(); it++)
	{
		const std::vector<int> & l = it->second.level;
		const std::vector<int> & j = it->second.suppt;

		std::vector<std::vector<int>> & order = it->second.order_local_alpt;

		std::vector<VecMultiD<double> > & ucoe = it->second.ucoe_alpt;

		std::vector<double> & xl = it->second.xl;
		std::vector<double> & xr = it->second.xr;

		std::vector<int> p(dgsolution_ptr->DIM);

		k = 0;

		// check whether pos[D] is in this element

		for (int d = 0; d < dgsolution_ptr->DIM; d++)
		{
			if (pos[d] < xl[d] || pos[d] > xr[d])
			{
				k = 1;
			}
		}

		// if k=0, this means pos[D] is in this element
		// then do the following step

		if (k == 1) continue;

		/////////////////////

		int m2 = 0;

		for (auto it0 = order.begin(); it0 != order.end(); it0++)
		{
			double tep = 1.0;

			for (int d = 0; d < dgsolution_ptr->DIM; d++)
			{
				p[d] = (*it0)[d];

				m2 = hash_key1d(l[d], j[d]) * (AlptBasis::PMAX + 1) + p[d];

				tep = tep * Her_pt_Alpt_1D[m1[d]][m2];
			}

			for (int i = 0; i < dgsolution_ptr->VEC_NUM; i++)
			{
				val[i] += ucoe[i].at(*it0) * tep;
			}

		}

	}

}


//////------------ Public functions------------------

void LagrInterpolation::nonlinear_Lagr(std::function<double(std::vector<double>, int, int)> func, std::vector< std::vector<bool> > is_intp)
{
	assert(is_intp.size() == VEC_NUM);
	for (size_t num = 0; num < VEC_NUM; num++) { assert(is_intp[num].size() == DIM); }

	eval_up_fp_Lag(func, is_intp); // compute the function values for func at all the Lagrange points; 

	pw1d.clear();

	eval_fp_to_coe_D_Lag(is_intp); // compute the hierachichal coefficients of the Lag interpolation basis from the point values;
}

void LagrInterpolation::nonlinear_Lagr_fast(std::function<double(std::vector<double>, int, int)> func,
	std::vector< std::vector<bool> > is_intp, FastLagrIntp & fastLagr)
{
	assert(is_intp.size() == VEC_NUM);
	for (size_t num = 0; num < VEC_NUM; num++) { assert(is_intp[num].size() == DIM); }

	fastLagr.eval_up_Lagr();

	eval_fp_Lag(func, is_intp);

	pw1d.clear();

	eval_fp_to_coe_D_Lag(is_intp);
}


void LagrInterpolation::nonlinear_Lagr_fast_full(std::function<double(std::vector<double>, int, int, std::vector<double>)> func,
	std::vector< std::vector<bool> > is_intp, FastLagrIntp & fastLagr)
{
	assert(is_intp.size() == VEC_NUM);
	for (size_t num = 0; num < VEC_NUM; num++) { assert(is_intp[num].size() == DIM); }

	fastLagr.eval_up_Lagr();

	eval_fp_Lag_full(func, is_intp);

	pw1d.clear();

	eval_fp_to_coe_D_Lag(is_intp);
}

void LagrInterpolation::nonlinear_Lagr_HJ_fast(std::function<double(std::vector<double>, int, int)> func, FastLagrIntp & fastLagr)
{
	fastLagr.eval_up_Lagr();

	eval_fp_Lag_HJ(func);

	pw1d.clear();

	eval_fp_to_coe_D_Lag_HJ();

}

void LagrInterpolation::init_from_Lagr_to_alpt(std::function<double(std::vector<double>, int)> func,
	const OperatorMatrix1D<LagrBasis, AlptBasis> & matrix)
{

	eval_up_exact_Lag(func);

	pw1d.clear();

	eval_up_to_coe_D_Lag();

	eval_coe_alpt_from_coe_Lag(matrix);

}

void LagrInterpolation::init_from_Lagr_to_alpt_fast(std::function<double(std::vector<double>, int)> func, FastLagrInit & fastLagr_init)
{

	eval_up_exact_Lag(func);

	pw1d.clear();

	eval_up_to_coe_D_Lag();

	fastLagr_init.eval_ucoe_Alpt_Lagr();
}

void LagrInterpolation::source_from_lagr_to_rhs(std::function<double(std::vector<double>, int)> func, FastLagrInit & fastLagr)
{
	// read function value at Lagrange point from exact function, update Element::up_intp
	eval_up_exact_Lag(func);

	pw1d.clear();

	// transform Element::up_intp to Element::ucoe_intp
	eval_up_to_coe_D_Lag();
	
	// temporarily store Element::ucoe_alpt in Element::ucoe_alpt_predict
	dgsolution_ptr->copy_ucoe_to_predict();

	// transform Element::ucoe_intp to Element::ucoe_alpt
	fastLagr.eval_ucoe_Alpt_Lagr();

	// add Element::ucoe_alpt to Element::rhs
	dgsolution_ptr->add_ucoe_to_rhs();

	// copy Element::ucoe_alpt_predict back to Element::ucoe_alpt
	dgsolution_ptr->copy_predict_to_ucoe();
}


void LagrInterpolation::init_from_Lagr_to_alpt_fast_full(std::function<double(std::vector<double>, int)> func, FastLagrFullInit & fastLagr_init)
{
	eval_up_exact_Lag(func);

	pw1d.clear();

	eval_up_to_coe_D_Lag();

	fastLagr_init.eval_ucoe_Alpt_Full();
}

void LagrInterpolation::init_coe_ada_intp_Lag(std::function<double(std::vector<double>, int)> func)
{

	eval_up_exact_ada_Lag(func);

	pw1d.clear();

	eval_up_to_coe_D_ada_Lag();

}

void LagrInterpolation::var_coeff_gradu_Lagr(std::function<double(std::vector<double>, int)> coe_func, std::vector< std::vector<bool> > is_intp)
{
	assert(is_intp.size() == VEC_NUM);
	for (size_t num = 0; num < VEC_NUM; num++) { assert(is_intp[num].size() == DIM); }

	wave_eval_coe_grad_u_Lag(coe_func, is_intp);

	eval_fp_to_coe_D_Lag(is_intp);

}

void LagrInterpolation::var_coeff_gradu_Lagr_fast(std::function<double(std::vector<double>, int)> coe_func,
	std::vector< std::vector<bool> > is_intp, FastLagrIntp & fastLagr)
{
	assert(is_intp.size() == VEC_NUM);
	for (size_t num = 0; num < VEC_NUM; num++) { assert(is_intp[num].size() == DIM); }

	for (int d = 0; d < DIM; d++)
	{
		fastLagr.eval_der_up_Lagr(d);

		eval_coe_grad_u_Lag(coe_func, is_intp, d);
	}

	eval_fp_to_coe_D_Lag(is_intp);

}

void LargViscosInterpolation::support_gradu_Lagr_fast(const std::unordered_set<Element*> & support_element, const double coefficient, std::vector< std::vector<bool> > is_intp, FastLagrIntp & fastLagr, const bool overlap)
{
	dgsolution_ptr->set_fp_intp_zero();

	dgsolution_ptr->update_viscosity_intersect_element();

	for (int d = 0; d < DIM; d++)
	{
		fastLagr.eval_der_up_Lagr(d);

		eval_artificial_viscos_grad_u_Lag(support_element, coefficient, is_intp, d, overlap);
	}

	eval_fp_to_coe_D_Lag(is_intp);
}

void LagrInterpolation::var_coeff_u_Lagr(std::function<double(std::vector<double>, int)> coe_func, std::vector< std::vector<bool> > is_intp)
{
	assert(is_intp.size() == VEC_NUM);
	for (size_t num = 0; num < VEC_NUM; num++) { assert(is_intp[num].size() == DIM); }

	wave_eval_coe_u_Lag(coe_func, is_intp);

	eval_fp_to_coe_D_Lag(is_intp);
}

void LagrInterpolation::var_coeff_u_Lagr_fast(std::function<double(std::vector<double>, int)> coe_func,
	std::vector< std::vector<bool> > is_intp, FastLagrIntp & fastLagr)
{
	assert(is_intp.size() == VEC_NUM);
	for (size_t num = 0; num < VEC_NUM; num++) { assert(is_intp[num].size() == DIM); }

	fastLagr.eval_up_Lagr();

	eval_coe_u_Lag(coe_func, is_intp);

	eval_fp_to_coe_D_Lag(is_intp);
}


void LagrInterpolation::interp_Vlasov_1D1V(DGSolution & E, FastLagrIntp & fastLagr_f, FastLagrIntp & fastLagr_E)
{		
	// step 1: fast transform, f alpert coefficients -> point values (store in Element::up_intp in f)
	fastLagr_f.eval_up_Lagr();

	// step 2: copy Element::ucoe_alpt in f to Element::ucoe_alpt_other in f
	dgsolution_ptr->copy_ucoe_to_other();

	// step 3: copy Element::up_intp in f to Element::up_intp_other in f
	dgsolution_ptr->copy_up_intp_to_other();

	// step 4: copy E alpert coefficients to f (store in Element::ucoe_alpt in f)
	const std::vector<int> num_vec_f{0};
	const std::vector<int> num_vec_E{0};
	const std::vector<int> vel_dim_f{1};
	dgsolution_ptr->copy_ucoe_alpt_to_f(E, num_vec_f, num_vec_E, vel_dim_f);
	
	// step 5: fast transform, f alpert coefficients -> point values (this is actually point value for E)
	fastLagr_f.eval_up_Lagr();

	// step 6:  exchange Element::ucoe_alpt in f and Element::ucoe_alpt_other in f
	// 			exchange Element::up_intp in f and Element::up_intp_other in f
	dgsolution_ptr->exchange_ucoe_and_other();
	dgsolution_ptr->exchange_up_intp_and_other();

	// step 7: compute v * f and E * f point value
	for (auto it = dgsolution_ptr->dg.begin(); it != dgsolution_ptr->dg.end(); it++)
	{
		const std::vector<int> & l = it->second.level;
		const std::vector<int> & j = it->second.suppt;

		std::vector<std::vector<int>> & order = it->second.order_local_intp;

		std::vector< VecMultiD<double> > & up = it->second.up_intp;
		std::vector< VecMultiD<double> > & up_other = it->second.up_intp_other;
		std::vector< std::vector< VecMultiD<double> > > & fp = it->second.fp_intp;

		std::vector<double> pos(DIM);
		std::vector<int> p(DIM);

		for (auto it0 = order.begin(); it0 != order.end(); it0++)
		{
			for (int d = 0; d < DIM; d++)
			{
				p[d] = (*it0)[d];	// polynomial degree of interpolation basis
				pos[d] = dgsolution_ptr->all_bas_Lag.at(l[d], j[d], p[d]).intep_pt;	// coordinate of interpolation point
			}

			for (int i = 0; i < VEC_NUM; i++)
			{
				// dimension 0: v * f
				int d = 0;
				fp[i][d].at(*it0) = pos[1] * up[i].at(*it0);
				
				// dimension 1: E * f
				d = 1;
				fp[i][d].at(*it0) = up_other[i].at(*it0) * up[i].at(*it0);
			}
		}
	}

	// step 8: transform point value to interpolation coefficients
	std::vector< std::vector<bool> > is_intp;
	is_intp.push_back(std::vector<bool>(DIM, true));
	eval_fp_to_coe_D_Lag(is_intp);
}


void LagrInterpolation::interp_Vlasov_1D1V(DGSolution & E, std::function<double(double)> coe_v, std::function<double(double)> coe_E, FastLagrIntp & fastLagr_f, FastLagrIntp & fastLagr_E)
{
	// step 1: fast transform, f alpert coefficients -> point values (store in Element::up_intp in f)
	fastLagr_f.eval_up_Lagr();

	// step 2: copy Element::ucoe_alpt in f to Element::ucoe_alpt_other in f
	dgsolution_ptr->copy_ucoe_to_other();

	// step 3: copy Element::up_intp in f to Element::up_intp_other in f
	dgsolution_ptr->copy_up_intp_to_other();

	// step 4: copy E alpert coefficients to f (store in Element::ucoe_alpt in f)
	const std::vector<int> num_vec_f{0};
	const std::vector<int> num_vec_E{0};
	const std::vector<int> vel_dim_f{1};
	dgsolution_ptr->copy_ucoe_alpt_to_f(E, num_vec_f, num_vec_E, vel_dim_f);
	
	// step 5: fast transform, f alpert coefficients -> point values (this is actually point value for E)
	fastLagr_f.eval_up_Lagr();

	// step 6:  exchange Element::ucoe_alpt in f and Element::ucoe_alpt_other in f
	// 			exchange Element::up_intp in f and Element::up_intp_other in f
	dgsolution_ptr->exchange_ucoe_and_other();
	dgsolution_ptr->exchange_up_intp_and_other();

	// step 7: compute v * f and E * f point value
	for (auto it = dgsolution_ptr->dg.begin(); it != dgsolution_ptr->dg.end(); it++)
	{
		const std::vector<int> & l = it->second.level;
		const std::vector<int> & j = it->second.suppt;

		std::vector<std::vector<int>> & order = it->second.order_local_intp;

		std::vector< VecMultiD<double> > & up = it->second.up_intp;
		std::vector< VecMultiD<double> > & up_other = it->second.up_intp_other;
		std::vector< std::vector< VecMultiD<double> > > & fp = it->second.fp_intp;

		std::vector<double> pos(DIM);
		std::vector<int> p(DIM);

		for (auto it0 = order.begin(); it0 != order.end(); it0++)
		{
			for (int d = 0; d < DIM; d++)
			{
				p[d] = (*it0)[d];	// polynomial degree of interpolation basis
				pos[d] = dgsolution_ptr->all_bas_Lag.at(l[d], j[d], p[d]).intep_pt;	// coordinate of interpolation point
			}

			for (int i = 0; i < VEC_NUM; i++)
			{
				// dimension 0: v * f
				int d = 0;
				fp[i][d].at(*it0) = coe_v(pos[1]) * up[i].at(*it0);
				
				// dimension 1: E * f
				d = 1;
				fp[i][d].at(*it0) = coe_E(up_other[i].at(*it0)) * up[i].at(*it0);
			}
		}
	}

	// step 8: transform point value to interpolation coefficients
	std::vector< std::vector<bool> > is_intp;
	is_intp.push_back(std::vector<bool>(DIM, true));
	eval_fp_to_coe_D_Lag(is_intp);
}

// f_t + v2 * f_x2 + (E1 + v2 * B3) * f_v1 + (E2 - v1 * B3) * f_v2 = 0
void LagrInterpolation::interp_Vlasov_1D2V(DGSolution & dg_BE, FastLagrIntp & fastLagr_f, FastLagrIntp & fastLagr_BE)
{
	// step 1: fast transform, f alpert coefficients -> point values (store in Element::up_intp in f)
	// only do transformation in the first component
	fastLagr_f.eval_up_Lagr(0);

	// step 2: fast transform, E alpert coefficients -> point values (store in Element::up_intp in E)
	fastLagr_BE.eval_up_Lagr();

	// step 3: copy Element::up_intp in E to Element::up_intp_other in f
	const std::vector<int> num_vec_f{0, 1, 2};
	const std::vector<int> num_vec_E{0, 1, 2};	// dg_BE = {B3, E1, E2}
	const std::vector<int> vel_dim_f{1, 2};
	dgsolution_ptr->copy_up_intp_to_f(dg_BE, num_vec_f, num_vec_E, vel_dim_f);
	
	// step 4: compute v * f and E * f point value
	for (auto it = dgsolution_ptr->dg.begin(); it != dgsolution_ptr->dg.end(); it++)
	{
		const std::vector<int> & l = it->second.level;
		const std::vector<int> & j = it->second.suppt;

		std::vector<std::vector<int>> & order = it->second.order_local_intp;

		std::vector< VecMultiD<double> > & up = it->second.up_intp;
		std::vector< VecMultiD<double> > & up_other = it->second.up_intp_other;
		std::vector< std::vector< VecMultiD<double> > > & fp = it->second.fp_intp;

		std::vector<double> pos(DIM);
		std::vector<int> p(DIM);

		for (auto it0 = order.begin(); it0 != order.end(); it0++)
		{
			for (int d = 0; d < DIM; d++)
			{
				p[d] = (*it0)[d];	// polynomial degree of interpolation basis
				pos[d] = dgsolution_ptr->all_bas_Lag.at(l[d], j[d], p[d]).intep_pt;	// coordinate of interpolation point
			}

			// f_t + v2 * f_x2 + (E1 + v2 * B3) * f_v1 + (E2 - v1 * B3) * f_v2 = 0
			
			// pos = (x2, v1, v2)
			double x2 = pos[0]; double v1 = pos[1]; double v2 = pos[2];
			
			double f = up[0].at(*it0);

			// dg_BE = {B3, E1, E2}
			double B3 = up_other[0].at(*it0);
			double E1 = up_other[1].at(*it0);
			double E2 = up_other[2].at(*it0);

			// dimension 0: v2 * f
			int d = 0;
			fp[0][d].at(*it0) = v2 * f;
			
			// dimension 1: (E1 + v2 * B3) * f
			d = 1;
			fp[0][d].at(*it0) = (E1 + v2 * B3) * f;

			// dimension 2: (E2 - v1 * B3) * f
			d = 2;
			fp[0][d].at(*it0) = (E2 - v1 * B3) * f;
		}
	}

	// step 5: transform point value to interpolation coefficients
	std::vector< std::vector<bool> > is_intp;
	is_intp.push_back(std::vector<bool>(DIM, true));	// interpolation for the 1st vec_num in all dimensions
	is_intp.push_back(std::vector<bool>(DIM, false));	// no interpolation for the 2nd vec_num in all dimensions
	is_intp.push_back(std::vector<bool>(DIM, false));	// no interpolation for the 2nd vec_num in all dimensions
	eval_fp_to_coe_D_Lag(is_intp);
}

void LagrInterpolation::interp_Vlasov_1D2V(DGSolution & dg_BE, std::function<double(double)> coe_x2, std::function<double(double, double, double)> coe_v1, std::function<double(double, double, double)> coe_v2, FastLagrIntp & fastLagr_f, FastLagrIntp & fastLagr_BE)
{
	// step 1: fast transform, f alpert coefficients -> point values (store in Element::up_intp in f)
	fastLagr_f.eval_up_Lagr();

	// step 2: copy Element::ucoe_alpt in f to Element::ucoe_alpt_other in f
	dgsolution_ptr->copy_ucoe_to_other();

	// step 3: copy Element::up_intp in f to Element::up_intp_other in f
	dgsolution_ptr->copy_up_intp_to_other();

	// step 4: copy E alpert coefficients to f (store in Element::ucoe_alpt in f)
	const std::vector<int> num_vec_f{0, 1, 2};
	const std::vector<int> num_vec_E{0, 1, 2};	// dg_BE = {B3, E1, E2}
	const std::vector<int> vel_dim_f{1, 2};
	dgsolution_ptr->copy_ucoe_alpt_to_f(dg_BE, num_vec_f, num_vec_E, vel_dim_f);
	
	// step 5: fast transform, f alpert coefficients -> point values (this is actually point value for E)
	fastLagr_f.eval_up_Lagr();

	// step 6:  exchange Element::ucoe_alpt in f and Element::ucoe_alpt_other in f
	// 			exchange Element::up_intp in f and Element::up_intp_other in f
	dgsolution_ptr->exchange_ucoe_and_other();
	dgsolution_ptr->exchange_up_intp_and_other();

	// step 7: compute v * f and E * f point value
	for (auto it = dgsolution_ptr->dg.begin(); it != dgsolution_ptr->dg.end(); it++)
	{
		const std::vector<int> & l = it->second.level;
		const std::vector<int> & j = it->second.suppt;

		std::vector<std::vector<int>> & order = it->second.order_local_intp;

		std::vector< VecMultiD<double> > & up = it->second.up_intp;
		std::vector< VecMultiD<double> > & up_other = it->second.up_intp_other;
		std::vector< std::vector< VecMultiD<double> > > & fp = it->second.fp_intp;

		std::vector<double> pos(DIM);
		std::vector<int> p(DIM);

		for (auto it0 = order.begin(); it0 != order.end(); it0++)
		{
			for (int d = 0; d < DIM; d++)
			{
				p[d] = (*it0)[d];	// polynomial degree of interpolation basis
				pos[d] = dgsolution_ptr->all_bas_Lag.at(l[d], j[d], p[d]).intep_pt;	// coordinate of interpolation point
			}

			// f_t + v2 * f_x2 + (E1 + v2 * B3) * f_v1 + (E2 - v1 * B3) * f_v2 = 0
			
			// pos = (x2, v1, v2)
			double x2 = pos[0]; double v1 = pos[1]; double v2 = pos[2];
			
			double f = up[0].at(*it0);

			// dg_BE = {B3, E1, E2}
			double B3 = up_other[0].at(*it0);
			double E1 = up_other[1].at(*it0);
			double E2 = up_other[2].at(*it0);

			// dimension 0: v2 * f
			int d = 0;
			fp[0][d].at(*it0) = coe_x2(v2) * f;
			
			// dimension 1: (E1 + v2 * B3) * f
			d = 1;
			fp[0][d].at(*it0) = coe_v1(v2, E1, B3) * f;

			// dimension 2: (E2 - v1 * B3) * f
			d = 2;
			fp[0][d].at(*it0) = coe_v2(v1, E2, B3) * f;
		}
	}

	// step 8: transform point value to interpolation coefficients
	std::vector< std::vector<bool> > is_intp;
	is_intp.push_back(std::vector<bool>(DIM, true));	// interpolation for the 1st vec_num in all dimensions
	is_intp.push_back(std::vector<bool>(DIM, false));	// no interpolation for the 2nd vec_num in all dimensions
	is_intp.push_back(std::vector<bool>(DIM, false));	// no interpolation for the 2nd vec_num in all dimensions
	eval_fp_to_coe_D_Lag(is_intp);
}

void HermInterpolation::nonlinear_Herm_1D(std::function<double(std::vector<double>, int, int)> func,
	std::function<double(std::vector<double>, int, int, int)> func_d1,
	std::vector< std::vector<bool> > is_intp)
{
	eval_up_Her();

	eval_fp_Her_1D(func, func_d1, is_intp);

	pw1d.clear();

	eval_fp_to_coe_D_Her(is_intp);
}

void HermInterpolation::nonlinear_Herm_2D(std::function<double(std::vector<double>, int, int)> func,
	std::function<double(std::vector<double>, int, int, int)> func_d1,
	std::function<double(std::vector<double>, int, int, int, int)>  func_d2,
	std::vector< std::vector<bool> > is_intp)
{
	eval_up_Her();

	eval_fp_Her_2D(func, func_d1, func_d2, is_intp);

	pw1d.clear();

	eval_fp_to_coe_D_Her(is_intp);
}

void HermInterpolation::nonlinear_Herm_2D_fast(std::function<double(std::vector<double>, int, int)> func,
	std::function<double(std::vector<double>, int, int, int)> func_d1,
	std::function<double(std::vector<double>, int, int, int, int)>  func_d2,
	std::vector< std::vector<bool> > is_intp, FastHermIntp & fastHerm)
{
	fastHerm.eval_up_Herm();

	eval_fp_Her_2D(func, func_d1, func_d2, is_intp);

	pw1d.clear();

	eval_fp_to_coe_D_Her(is_intp);
}


void HermInterpolation::nonlinear_Herm_2D_PMAX5_scalar_fast(std::function<double(std::vector<double>, int, int)> func,
	std::function<double(std::vector<double>, int, int, int)> func_d1,
	std::function<double(std::vector<double>, int, int, int, int)>  func_d2,
	std::function<double(std::vector<double>, int, int, int, int, int)>  func_d3,
	std::function<double(std::vector<double>, int, int, int, int, int, int)>  func_d4,
	std::vector< std::vector<bool> > is_intp, FastHermIntp & fastHerm)
{
	fastHerm.eval_up_Herm();

	// eval_fp_Her_2D(func, func_d1, func_d2, is_intp);

	eval_fp_Her_2D_PMAX5_scalar(func, func_d1, func_d2, func_d3, func_d4, is_intp);

	pw1d.clear();

	eval_fp_to_coe_D_Her(is_intp);
}

void HermInterpolation::init_from_Herm_to_alpt(std::function<double(std::vector<double>, int, std::vector<int>)> func,
	const OperatorMatrix1D<HermBasis, AlptBasis> & matrix)
{

	eval_up_exact_Her(func);

	pw1d.clear();

	eval_up_to_coe_D_Her();

	eval_coe_alpt_from_coe_Her(matrix);
}


void HermInterpolation::init_from_Herm_to_alpt_fast(std::function<double(std::vector<double>, int, std::vector<int>)> func,
	FastHermInit & fastHerm_init)
{

	eval_up_exact_Her(func);

	pw1d.clear();

	eval_up_to_coe_D_Her();

	fastHerm_init.eval_ucoe_Alpt_Herm();


}

void HermInterpolation::init_coe_ada_intp_Herm(std::function<double(std::vector<double>, int, std::vector<int>)> func)
{

	eval_up_exact_ada_Her(func);

	pw1d.clear();

	eval_up_to_coe_D_ada_Her();

}


double FluxFunction::linear_flux_scalar(const double u, int dim, const std::vector<double> & coefficient)
{
	return u * coefficient[dim];
}

double FluxFunction::linear_flux_1st_derivative_scalar(const double u, const int dim, const std::vector<double> & coefficient)
{
	return coefficient[dim];
}

double FluxFunction::burgers_flux_scalar(const double u)
{
	return u * u * 0.5;
}

double FluxFunction::burgers_flux_1st_derivative_scalar(const double u)
{
	return u;
}

double FluxFunction::burgers_flux_2nd_derivative_scalar(const double u)
{
	return 1.;
}

double FluxFunction::sin_flux(const double u)
{
	return sin(u);
}

double FluxFunction::sin_flux_1st_derivative(const double u)
{
	return cos(u);
}

double FluxFunction::sin_flux_2nd_derivative(const double u)
{
	return -sin(u);
}

double FluxFunction::cos_flux(const double u)
{
	return cos(u);
}

double FluxFunction::cos_flux_1st_derivative(const double u)
{
	return -sin(u);
}

double FluxFunction::cos_flux_2nd_derivative(const double u)
{
	return -cos(u);
}

double FluxFunction::buckley_leverett_dim_x(const double u)
{
	const double u_pow2 = u * u;
	const double denom = 1 - 2 * u + 2 * u_pow2;
	return u_pow2 / (1 - 2 * u + 2 * u_pow2);
}

double FluxFunction::buckley_leverett_dim_x_1st_derivative(const double u)
{
	const double u_pow2 = u * u;
	const double denom = 1 - 2 * u + 2 * u_pow2;
	return -2 * (-1 + u)*u / std::pow(denom, 2.);
}

double FluxFunction::buckley_leverett_dim_x_2nd_derivative(const double u)
{
	const double u_pow2 = u * u;
	const double u_pow3 = u_pow2 * u;
	const double denom = 1 - 2 * u + 2 * u_pow2;
	return 2 * (1 - 6 * u_pow2 + 4 * u_pow3) / std::pow(denom, 3.);
}

double FluxFunction::buckley_leverett_dim_y(const double u)
{
	const double u_pow2 = u * u;
	const double denom = 1 - 2 * u + 2 * u_pow2;
	return 	-u_pow2 * (4 - 10 * u + 5 * u_pow2) / denom;
}

double FluxFunction::buckley_leverett_dim_y_1st_derivative(const double u)
{
	const double u_pow2 = u * u;
	const double u_pow3 = u * u_pow2;
	const double u_pow4 = u * u_pow3;
	const double denom = 1 - 2 * u + 2 * u_pow2;
	return -2 * u*(4 - 19 * u + 30 * u_pow2 - 25 * u_pow3 + 10 * u_pow4) / std::pow(denom, 2.);
}

double FluxFunction::buckley_leverett_dim_y_2nd_derivative(const double u)
{
	const double u_pow2 = u * u;
	const double u_pow3 = u * u_pow2;
	const double u_pow4 = u * u_pow3;
	const double u_pow5 = u * u_pow4;
	const double u_pow6 = u * u_pow5;
	const double denom = 1 - 2 * u + 2 * u_pow2;
	return -4 * (2 - 15 * u + 33 * u_pow2 - 42 * u_pow3 + 45 * u_pow4 - 30 * u_pow5 + 10 * u_pow6) / std::pow(denom, 3.);
}

bool LargViscosInterpolation::is_pt_in_interval(const double pt, const double xl, const double xr)
{
	if ((pt >= xl) && (pt <= xr)) { return true; }
	return false;
}

bool LargViscosInterpolation::is_pt_in_interval(const double pt, const std::vector<double> & interval)
{
	if ((pt >= interval[0]) && (pt <= interval[1])) { return true; }
	return false;
}

bool LargViscosInterpolation::is_pt_in_interval(const std::vector<double> & pt, const std::vector<double> & xl, const std::vector<double> & xr)
{
	for (size_t d = 0; d < pt.size(); d++)
	{
		if (!(is_pt_in_interval(pt[d], xl[d], xr[d]))) { return false; }
	}
	return true;
}

bool LargViscosInterpolation::is_pt_in_set_element(const std::vector<double> & pt, const std::unordered_set<Element*> & element_set)
{
	for (auto const & elem : element_set)
	{
		if (is_pt_in_interval(pt, elem->xl, elem->xr)) { return true; }
	}
	return false;
}

int LargViscosInterpolation::num_pt_in_set_element(const std::vector<double> & pt, const std::unordered_set<Element*> & element_set)
{
	int num = 0;
	for (auto const & elem : element_set)
	{
		if (is_pt_in_interval(pt, elem->xl, elem->xr)) { num++; }
	}
	return num;
}