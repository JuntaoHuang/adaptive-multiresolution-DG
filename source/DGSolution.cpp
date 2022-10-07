#include "DGSolution.h"
#include "subs.h"
#include "Quad.h"

int DGSolution::DIM;
int DGSolution::VEC_NUM;
std::string DGSolution::prob;
std::vector<int> DGSolution::ind_var_vec;

DGSolution::DGSolution(const bool sparse_, const int level_init_, const int NMAX_, AllBasis<AlptBasis> & all_bas_, AllBasis<LagrBasis> all_bas_Lag_, AllBasis<HermBasis> all_bas_Her_, Hash & hash_):
	sparse(sparse_), level_init(level_init_), NMAX(NMAX_), all_bas(all_bas_), all_bas_Lag(all_bas_Lag_), all_bas_Her(all_bas_Her_), hash(hash_)
{
	assert(ind_var_vec.size() != 0);
	assert(Element::is_intp.size() == VEC_NUM);
	for (size_t num = 0; num < VEC_NUM; num++) { assert(Element::is_intp[num].size() == DIM); }

	// generate vector narray as mesh level index, each element is a vector of size dim: ( n1, n2, ..., n_(dim) ), 0 <= n_i <= NMAX for any 1 <= i <= dim
	std::vector<std::vector<int>> narray;
	const std::vector<int> narray_max(DIM, level_init + 1); // 
	IterativeNestedLoop(narray, DIM, narray_max); // generate a full-grid mesh level index; avoid loop of the dimension

	for (auto const & lev_n : narray) 
	{
		// loop over all mesh level array of size dim

		// case 1: standard sparse grid, only account for basis with sum of index n <= NMAX 
		if (sparse == 1 && (std::accumulate(lev_n.begin(), lev_n.end(), 0) > level_init))  continue;
		
		// case 3: standard sparse grid, only account for basis with sum of index n <= NMAX + DIM - 1 
		//if (sparse == 2 && (std::accumulate(lev_n.begin(), lev_n.end(), 0) > level_init + DIM - 1))	continue;
		
		// case 2： full grid, generate index j: 0, 1, ..., 2^(n-1)-1, for n >= 2. Otherwise j=0 for n=0,1
		std::vector<int> jarray_max;
		for (auto const & n : lev_n)
		{
			int jmax = 1;
			if (n != 0) jmax = pow_int(2, n - 1);

			jarray_max.push_back(jmax);
		}

		std::vector<std::vector<int>> jarray;
		IterativeNestedLoop(jarray, DIM, jarray_max);

		for (auto const & sup_j : jarray)
		{
			// transform to odd index
			std::vector<int> odd_j(sup_j.size());
			for (size_t i = 0; i < sup_j.size(); i++) { odd_j[i] = 2 * sup_j[i] + 1; }
			
			Element elem(lev_n, odd_j, all_bas, hash);	
			dg.insert({ elem.hash_key, elem });
		}
	}

	update_order_all_basis_in_dgmap();
}

DGSolution::DGSolution(const bool sparse_, const int level_init_, const int NMAX_, const int auxiliary_dim_,
	AllBasis<AlptBasis> & all_bas_, AllBasis<LagrBasis> all_bas_Lag_, AllBasis<HermBasis> all_bas_Her_, Hash & hash_):
	sparse(sparse_), level_init(level_init_), NMAX(NMAX_), 
	auxiliary_dim(auxiliary_dim_),
	all_bas(all_bas_), all_bas_Lag(all_bas_Lag_), all_bas_Her(all_bas_Her_), hash(hash_)
{
	assert(ind_var_vec.size() != 0);
	assert(Element::is_intp.size() == VEC_NUM);	
	for (size_t num = 0; num < VEC_NUM; num++) { assert(Element::is_intp[num].size() == DIM); }
	
	// the first real_dim use full grid, the remaining auxiliary dimension use only zero level grid
	const int real_dim = DIM - auxiliary_dim;
	assert(sparse == 0);	// only full grid (in the first real_dim dimension) is allowed in this contructor

	// generate vector narray as mesh level index
	// each element is a vector of size dim: ( n1, n2, ..., n_(dim) )
	// 0 <= n_d <= level_init,	for any 1 <= d <= real_dim
	// n_d = 0					for any real_dim < d <= dim
	std::vector<std::vector<int>> narray;
	// const std::vector<int> narray_max(DIM, level_init + 1);
	std::vector<int> narray_max(real_dim, level_init + 1);
	for (size_t d = 0; d < auxiliary_dim; d++) { narray_max.push_back(1); }

	IterativeNestedLoop(narray, DIM, narray_max); // generate a full-grid mesh level index; avoid loop of the dimension

	for (auto const & lev_n : narray) 
	{
		// loop over all mesh level array of size dim

		// // case 1: standard sparse grid, only account for basis with sum of index n <= NMAX		
		// if (sparse == 1 && (std::accumulate(lev_n.begin(), lev_n.end(), 0) > level_init))  continue;

		// case 2： full grid, generate index j: 0, 1, ..., 2^(n-1)-1, for n >= 2. Otherwise j=0 for n=0,1
		std::vector<int> jarray_max;
		for (auto const & n : lev_n)
		{
			int jmax = 1;
			if (n != 0) jmax = pow_int(2, n - 1);

			jarray_max.push_back(jmax);
		}

		std::vector<std::vector<int>> jarray;
		IterativeNestedLoop(jarray, DIM, jarray_max);

		for (auto const & sup_j : jarray)
		{
			// transform to odd index
			std::vector<int> odd_j(sup_j.size());
			for (size_t i = 0; i < sup_j.size(); i++) { odd_j[i] = 2 * sup_j[i] + 1; }
			
			Element elem(lev_n, odd_j, all_bas, hash);	
			dg.insert({ elem.hash_key, elem });
		}
	}

	update_order_all_basis_in_dgmap();
}

std::vector<double> DGSolution::max_abs_value(const std::vector<int> & sample_max_mesh_level) const
{	
	// number of sampling points in each dimension
	std::vector<int> num_grid_point(DIM, 0);
	std::vector<double> dx(DIM, 0.);
	for (int d = 0; d < DIM; d++)
	{ 
		// here we use +1 to avoid sampling points in the cell interfaces
		num_grid_point[d] = pow_int(2, sample_max_mesh_level[d]) + 1;
		dx[d] = 1./num_grid_point[d];
	}

	std::vector<double> max_abs_solution(VEC_NUM, 0.);
	const std::vector<int> zero_derivative(DIM, 0);
	
	std::vector<std::vector<int>> narray;	
	IterativeNestedLoop(narray, DIM, num_grid_point);
	for (auto const & n : narray)
	{
		std::vector<double> x(DIM, 0.);
		for (int d = 0; d < DIM; d++) { x[d] = dx[d] * (n[d] + 0.5); }

		std::vector<double> value = this->val(x, zero_derivative);
		for (int vec = 0; vec < VEC_NUM; vec++)
		{
			max_abs_solution[vec] = std::max(std::abs(max_abs_solution[vec]), std::abs(value[vec]));
		}
	}

	return max_abs_solution;
}

VecMultiD<double> DGSolution::seperable_project(std::function<double(double, int)> func) const
{
	const std::vector<int> size_coeff{DIM, all_bas.size()};
	VecMultiD<double> coeff(2, size_coeff);

	// project function in each dim to basis in 1D
	for (size_t d = 0; d < DIM; d++)
	{
		auto func_xd = [&](double x) {return func(x,d); };
		std::vector<double> coeff_d = all_bas.projection(func_xd);
		
		for (size_t bas = 0; bas < all_bas.size(); bas++)
		{
			coeff.at(d, bas) = coeff_d[bas];
		}
	}
	return coeff;
}

void DGSolution::init_elem_separable(Element & elem, const VecMultiD<double> & coefficient_1D, const int index_var)
{
	// the default value of index_var = 0;
	// loop over each Alpert basis function
	for (size_t i = 0; i < elem.size_alpt(); i++) // elem.size_alpt() denotes the number of all the alpt basis used 
	{
		double coe = 1.;
		const std::vector<int> & order_global = elem.order_global_alpt[i];
		for (size_t d = 0; d < DIM; d++)
		{
			coe *= coefficient_1D.at(d, order_global[d]);
		}			
		//elem.ucoe_alpt[index_var].at(elem.order_local_alpt[i]) += coe;
		elem.ucoe_alpt[index_var].at(i) += coe;
	}
}

void DGSolution::source_elem_separable(Element & elem, const VecMultiD<double> & coefficient_1D, const int index_var)
{
	// the default value of index_var = 0;
	// loop over each Alpert basis function
	for (size_t i = 0; i < elem.size_alpt(); i++) // elem.size_alpt() denotes the number of all the alpt basis used 
	{
		double coe = 1.;
		const std::vector<int> & order_global = elem.order_global_alpt[i];
		for (size_t d = 0; d < DIM; d++)
		{
			coe *= coefficient_1D.at(d, order_global[d]);
		}			
		elem.source[index_var].at(i) += coe;
	}
}

std::vector<double> DGSolution::get_error_separable_scalar(std::function<double(double, int)> func, const int gauss_points) const
{
	// write function in non-separable form
	auto f = [&](std::vector<double> x) 
		{	
			double y = 1.;
			for (size_t d = 0; d < DIM; d++)
			{
				y *= func(x[d], d);
			}			
			return y;
		};
	return get_error_no_separable_scalar(f, gauss_points); 
}

double DGSolution::get_L2_error_split_separable_scalar(std::function<double(double, int)> func, const double l2_norm_exact_soln) const
{
	// Add integral term of u^2
	double err2 = l2_norm_exact_soln * l2_norm_exact_soln;

	// Add integral term of u_h^2 and 2 * u * u_h

	// project function in each dim to basis in 1D
	// coeff is a two dim vector with size (dim, # all_basis_1D)
	// coeff.at(d, order) denote the inner product of order-th basis function with initial function in d-th dim
	const VecMultiD<double> coeff = seperable_project(func);

	for (auto iter : dg)
	{
		std::vector< VecMultiD<double> > & ucoe_alpt = (iter.second).ucoe_alpt;
		
		for (size_t it0 = 0; it0 < iter.second.size_alpt(); it0++) // size_alpt() denotes the number of all the alpt basis used 
		{
			// Add integral term of u_h^2
			err2 += ucoe_alpt[0].at(it0) * ucoe_alpt[0].at(it0);

			// Add integral term of 2*u*u_h
			double coe = 1.;

			const std::vector<int> & order_global = iter.second.order_global_alpt[it0];

			for (size_t d = 0; d < DIM; d++)
			{
				coe *= coeff.at(d, order_global[d]);
			}	

			err2 -= 2.0 * ucoe_alpt[0].at(it0) * coe;			
		}		
	}

	if (err2 < 0.0) { std::cout << "warning: sqrt of negative value due to round off error in DGSolution::get_L2_error_split_separable_scalar()" << std::endl; }

	return std::sqrt(err2);

}

std::vector<double> DGSolution::get_error_separable_system(std::vector<std::function<double(double, int)>> func, const int gauss_points) const
{
	// write function in non-separable form
	std::vector<std::function<double(std::vector<double>)>> vector_func;
	for (auto & f_separable : func)
	{
		auto f = [&](std::vector<double> x) -> double
			{	
				double y = 1.;
				for (size_t d = 0; d < DIM; d++)
				{
					y *= f_separable(x[d], d);
				}			
				return y;
			};
		vector_func.push_back(f);
	}
	return get_error_no_separable_system(vector_func, gauss_points);
}

std::vector<double> DGSolution::get_error_separable_scalar_sum(std::vector<std::function<double(double, int)>> func, const int gauss_points) const
{
	const int num_separable_func = func.size();

	auto f = [&](std::vector<double> x) -> double
		{	
			double y = 0.;
			for (size_t i = 0; i < num_separable_func; i++)
			{
				double y_i = 1.;
				for (size_t d = 0; d < DIM; d++)
				{
					y_i *= func[i](x[d], d);
				}
				y += y_i;
			}
			return y;
		};
	return get_error_no_separable_scalar(f, gauss_points);
}

std::vector<double> DGSolution::get_error_separable_system_sum(std::vector<std::vector<std::function<double(double, int)>>> func, const int gauss_points) const
{
	// write function in non-separable form
	std::vector<std::function<double(std::vector<double>)>> vector_func;

	// loop over unknow variable of each component
	for (auto & f_separable_sum : func)
	{
		auto f = [&](std::vector<double> x) -> double
			{	
				double f_val = 0.;
				for (auto & f_separable : f_separable_sum)
				{
					double y = 1.;
					for (size_t d = 0; d < DIM; d++)
					{
						y *= f_separable(x[d], d);
					}
					f_val += y;
				}
				return f_val;
			};
		vector_func.push_back(f);
	}
	return get_error_no_separable_system(vector_func, gauss_points);
}

std::vector<double> DGSolution::get_error_no_separable_scalar(std::function<double(std::vector<double>)> func, const int gauss_points) const
{
	std::vector<int> zero_derivative(DIM, 0);
	auto err_fun = [&](std::vector<double> x) { return std::abs( val(x, zero_derivative)[0] - func(x) ); };

	// take the larger value of max mesh level of current active elements and 5
	const int max_mesh_quad = std::max(this->max_mesh_level(), 0);
	Quad quad(DIM);	
	return quad.norm_multiD(err_fun, max_mesh_quad, gauss_points);
}

std::vector<double> DGSolution::get_error_no_separable_system(std::vector<std::function<double(std::vector<double>)>> func, const int gauss_points) const
{
	std::vector<int> zero_derivative(DIM, 0);
	auto err_fun = [&](std::vector<double> x)->double 
		{
			double err = 0.;
			for (size_t num_var = 0; num_var < VEC_NUM; num_var++)
			{
				err += std::pow(val(x, zero_derivative)[num_var] - func[num_var](x), 2.);
			}
			return std::pow(err, 0.5);
		};

	const int max_mesh_quad = std::max(this->max_mesh_level(), 0);
	Quad quad(DIM);	
	return quad.norm_multiD(err_fun, max_mesh_quad, gauss_points);	
}

std::vector<double> DGSolution::get_error_no_separable_system_each(std::vector<std::function<double(std::vector<double>)>> func, const int gauss_points, int ind_var) const
{
	std::vector<int> zero_derivative(DIM, 0);
	auto err_fun = [&](std::vector<double> x) { return val(x, zero_derivative)[ind_var] - func[ind_var](x); };

	const int max_mesh_quad = std::max(this->max_mesh_level(), 0);
	Quad quad(DIM);	
	return quad.norm_multiD(err_fun, max_mesh_quad, gauss_points);	
}

std::vector<double> DGSolution::get_error_no_separable_system(std::function<double(std::vector<double>)> func, const int gauss_points, int ind_var) const
{
	std::vector<int> zero_derivative(DIM, 0);
	auto err_fun = [&](std::vector<double> x)->double
	{
		double err = 0.;
		err += std::pow(val(x, zero_derivative)[ind_var] - func(x), 2.);

		return std::pow(err, 0.5);
	};

	Quad quad(DIM);
	return quad.norm_multiD(err_fun, NMAX, gauss_points);
}


std::vector<double> DGSolution::get_error_no_separable_system_omp(std::function<double(std::vector<double>)> func, const int gauss_points, int ind_var) const
{
	std::vector<int> zero_derivative(DIM, 0);
	auto err_fun = [&](std::vector<double> x)->double
	{
		double err = 0.;
		err += std::pow(val(x, zero_derivative)[ind_var] - func(x), 2.);

		return std::pow(err, 0.5);
	};

	Quad quad(DIM);
	return quad.norm_multiD_omp(err_fun, NMAX, gauss_points);
}

std::vector<double> DGSolution::get_error_Lag_scalar(std::function<double(std::vector<double>)> func, const int gauss_points) const
{
	assert(VEC_NUM == 1);

	auto err_fun = [&](std::vector<double> x) ->double
	{
		return std::abs(val_Lag(x)[0] - func(x));
	};

	Quad quad(DIM);
	const int n = max_mesh_level();
	std::vector<double> norm = quad.norm_multiD(err_fun, n, gauss_points);

	return norm;
}

std::vector<double> DGSolution::get_error_Lag_scalar_random_points(std::function<double(std::vector<double>)> func, const int num_points) const
{
	assert(VEC_NUM == 1);

	auto err_fun = [&](std::vector<double> x) ->double
	{
		return std::abs(val_Lag(x)[0] - func(x));
	};

	double err_1 = 0.;
	double err_2 = 0.;
	double err_inf = 0.;

	// this is copied from https://en.cppreference.com/w/cpp/numeric/random/uniform_real_distribution
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    for (int n = 0; n < num_points; ++n)
	{
		std::vector<double> x(DIM, 0.);
		for (int d = 0; d < DIM; d++)
		{ 
			x[d] = dis(gen);
		}

		double err = err_fun(x);
		err_1 += err;
		err_2 += err * err;
		err_inf = std::max(err_inf, err);
    }

	err_1 = err_1 / num_points;
	err_2 = std::sqrt(err_2 / num_points);
	
	std::vector<double> norm{err_1, err_2, err_inf};
	return norm;
}

// compute the error for all flux functions
// std::vector<double> DGSolution::get_error_Lag(std::vector< std::vector< std::function<double(std::vector<double>)> > >func, const int gauss_points) const

std::vector<double> DGSolution::get_error_Lag(std::function<double(std::vector<double>, int, int)> func, const int gauss_points, std::vector< std::vector<bool> > is_intp) const
{

	auto err_fun = [&](std::vector<double> x, int ii, int dd) ->double
	{
		return std::abs(val_Lag(x, ii, dd) - func(x, ii, dd));
	};

	////////////

	double err1 = 0.;
	double err2 = 0.;
	double err8 = 0.;

	std::vector<double> normtot;

	for (int i = 0; i < VEC_NUM; i++)
	{
		for (int d = 0; d < DIM; d++)
		{
			if (is_intp[i][d] == true)
			{
				Quad quad(DIM);
				std::vector<double> norm = quad.norm_multiD(err_fun, i, d, NMAX, gauss_points);

				err1 += norm[0];
				err2 += norm[1] * norm[1];
				err8 = std::max(err8, norm[2]);
			}

		}

	}

	normtot.push_back(err1);
	normtot.push_back(std::sqrt(err2));
	normtot.push_back(err8);

	return normtot;

}


std::vector<double> DGSolution::get_error_Lag(std::vector< std::vector< std::function<double(std::vector<double>)> > >func, const int gauss_points) const
{

	auto err_fun = [&](std::vector<double> x, int ii, int dd) ->double
	{
		return std::abs(val_Lag(x, ii, dd) - func[ii][dd](x));
	};

	////////////

	double err1 = 0.;
	double err2 = 0.;
	double err8 = 0.;

	std::vector<double> normtot;

	for (int i = 0; i < VEC_NUM; i++)
	{
		for (int d = 0; d < DIM; d++)
		{
			Quad quad(DIM);
			std::vector<double> norm = quad.norm_multiD(err_fun, i, d, NMAX, gauss_points);

			err1 += norm[0];
			err2 += norm[1] * norm[1];
			err8 = std::max(err8, norm[2]);

		}

	}

	normtot.push_back(err1);
	normtot.push_back(std::sqrt(err2));
	normtot.push_back(err8);

	return normtot;

}

// compute the error for all flux functions
std::vector<double> DGSolution::get_error_Her(std::vector< std::vector< std::function<double(std::vector<double>)> > > func, const int gauss_points) const
{
	auto err_fun = [&](std::vector<double> x, int ii, int dd) ->double
	{
		return std::abs(val_Her(x, ii, dd) - func[ii][dd](x));
	};

	////////////

	double err1 = 0.;
	double err2 = 0.;
	double err8 = 0.;

	for (int i = 0; i < VEC_NUM; i++)
	{
		for (int d = 0; d < DIM; d++)
		{
			Quad quad(DIM);

			std::vector<double> norm = quad.norm_multiD(err_fun, i, d, NMAX, gauss_points);

			err1 += norm[0];
			err2 += norm[1] * norm[1];
			err8 = std::max(err8, norm[2]);
		}

	}

	std::vector<double> normtot;

	normtot.push_back(err1);
	normtot.push_back(std::sqrt(err2));
	normtot.push_back(err8);

	return normtot;
}

std::vector<double> DGSolution::get_error_Her(std::function<double(std::vector<double>, int, int)> func, const int gauss_points, std::vector< std::vector<bool> > is_intp) const
{
	auto err_fun = [&](std::vector<double> x, int ii, int dd) ->double
	{
		return std::abs(val_Her(x, ii, dd) - func(x, ii, dd));
	};

	////////////

	double err1 = 0.;
	double err2 = 0.;
	double err8 = 0.;

	for (int i = 0; i < VEC_NUM; i++)
	{
		for (int d = 0; d < DIM; d++)
		{
			if (is_intp[i][d] == true)
			{
				Quad quad(DIM);

				std::vector<double> norm = quad.norm_multiD(err_fun, i, d, NMAX, gauss_points);

				err1 += norm[0];
				err2 += norm[1] * norm[1];
				err8 = std::max(err8, norm[2]);
			}
		}

	}

	std::vector<double> normtot;

	normtot.push_back(err1);
	normtot.push_back(std::sqrt(err2));
	normtot.push_back(err8);

	return normtot;
}

std::vector<double> DGSolution::val(const std::vector<double> & x, const std::vector<int> & derivative) const
{
	std::vector<double> value(VEC_NUM, 0.);
	for (auto && iter : dg)
	{
		const std::vector<double> & val_elem = iter.second.val(x, all_bas, derivative);
		for (size_t num_var = 0; num_var < VEC_NUM; num_var++)
		{
			value[num_var] += val_elem[num_var];
		}
	}
	return value;
}

std::vector<double> DGSolution::val_Lag(const std::vector<double> & x) const
{
	std::vector<double> value(VEC_NUM, 0.);
	for (auto & iter : dg)
	{
		const std::vector<double> val_elem = iter.second.val_Lag(x, all_bas_Lag);
		for (size_t num_var = 0; num_var < VEC_NUM; num_var++)
		{
			value[num_var] += val_elem[num_var];
		}
	}
	return value;
}

std::vector<double> DGSolution::val_Her(const std::vector<double> & x) const
{
	std::vector<double> value(VEC_NUM, 0.);
	for (auto & iter : dg)
	{
		const std::vector<double> val_elem = iter.second.val_Her(x, all_bas_Her);
		for (size_t num_var = 0; num_var < VEC_NUM; num_var++)
		{
			value[num_var] += val_elem[num_var];
		}
	}
	return value;
}

double DGSolution::val_Lag(const std::vector<double> & x, const int ii, const int dd) const
{
	double value = 0.;
	for (auto const & iter : dg)
	{
		value += iter.second.val_Lag(x, ii, dd, all_bas_Lag);
	}
	return value;
}

double DGSolution::val_Her(const std::vector<double> & x, const int ii, const int dd) const
{
	double value = 0.;
	for (auto & iter : dg)
	{
		value += iter.second.val_Her(x, ii, dd, all_bas_Her);
	}
	return value;
}

void DGSolution::find_ptr_vol_alpt()
{
	// loop over all test elements
	for (auto & iter_test : dg)
	{
		// clear pointers
		iter_test.second.ptr_vol_alpt.clear();

		// loop over each dimension
		for (size_t d = 0; d < DIM; d++)
		{
			// pointers in d-th dimension
			std::unordered_set<Element*> ptr_vol_alpt_d;

			// loop over all solution elements
			for (auto & iter_solu : dg)
			{
				// if vol integral is not zero in Alpert basis
				if (iter_test.second.is_vol_alpt(iter_solu.second, d))
				{
					ptr_vol_alpt_d.insert(&(iter_solu.second));
				}
			}
			iter_test.second.ptr_vol_alpt.push_back(ptr_vol_alpt_d);
		}
	}
}

void DGSolution::find_ptr_flx_alpt()
{
	// loop over all test elements
	for (auto & iter_test : dg)
	{
		// clear pointers
		iter_test.second.ptr_flx_alpt.clear();

		// loop over each dimension
		for (size_t d = 0; d < DIM; d++)
		{
			// pointers in d-th dimension
			std::unordered_set<Element*> ptr_flx_alpt_d;

			// loop over all solution elements
			for (auto & iter_solu : dg)
			{
				if (iter_test.second.is_flx_alpt(iter_solu.second, d))
				{
					ptr_flx_alpt_d.insert(&(iter_solu.second));
				}
			}
			iter_test.second.ptr_flx_alpt.push_back(ptr_flx_alpt_d);
		}
	}
}

void DGSolution::find_ptr_vol_intp()
{
	// loop over all test elements
	for (auto & iter_test : dg)
	{
		// clear pointers
		iter_test.second.ptr_vol_intp.clear();

		// loop over each dimension
		for (size_t d = 0; d < DIM; d++)
		{
			// pointers in d-th dimension
			std::unordered_set<Element*> ptr_vol_intp_d;

			// loop over all solution elements
			for (auto & iter_solu : dg)
			{
				if (iter_test.second.is_vol_intp(iter_solu.second))
				{
					ptr_vol_intp_d.insert(&(iter_solu.second));
				}
			}
			iter_test.second.ptr_vol_intp.push_back(ptr_vol_intp_d);
		}
	}
}

void DGSolution::find_ptr_flx_intp()
{
	// loop over all test elements
	for (auto & iter_test : dg)
	{
		// clear pointers
		iter_test.second.ptr_flx_intp.clear();

		// loop over each dimension
		for (size_t d = 0; d < DIM; d++)
		{
			// pointers in d-th dimension
			std::unordered_set<Element*> ptr_flx_intp_d;

			// loop over all solution elements
			for (auto & iter_solu : dg)
			{
				if (iter_test.second.is_flx_intp(iter_solu.second, d))
				{
					ptr_flx_intp_d.insert(&(iter_solu.second));
				}
			}
			iter_test.second.ptr_flx_intp.push_back(ptr_flx_intp_d);
		}
	}
}

void DGSolution::find_ptr_general()
{
	// loop over all test elements
	for (auto & iter_test : dg)
	{
		// clear pointers
		iter_test.second.ptr_general.clear();

		// loop over all solution elements
		for (auto & iter_solu : dg)
		{
			if (iter_test.second.is_element_multidim_intersect_adjacent(iter_solu.second))
			{
				iter_test.second.ptr_general.insert(&(iter_solu.second));
			}
		}
	}
}

void DGSolution::set_ptr_to_all_elem()
{
	// pointers to all elements
	std::unordered_set<Element*> ptr_all_elem;
	for (auto & iter : dg)
	{
		ptr_all_elem.insert(&(iter.second));
	}

	// loop over each element
	for (auto & iter : dg)
	{
		// first clear pointers
		iter.second.ptr_vol_alpt.clear();
		iter.second.ptr_vol_intp.clear();
		iter.second.ptr_flx_alpt.clear();
		iter.second.ptr_flx_intp.clear();

		iter.second.ptr_general.clear();

		// then put pointers to all elements
		for (size_t dim = 0; dim < DIM; dim++)
		{
			iter.second.ptr_vol_alpt.push_back(ptr_all_elem);
			iter.second.ptr_vol_intp.push_back(ptr_all_elem);
			iter.second.ptr_flx_alpt.push_back(ptr_all_elem);
			iter.second.ptr_flx_intp.push_back(ptr_all_elem);
		}

		iter.second.ptr_general = ptr_all_elem;
	}
}

int DGSolution::size_elem() const
{
	return dg.size();
}

int DGSolution::size_basis_alpt() const
{
	return dg.size() * pow_int(Element::PMAX_alpt + 1, DIM);
}

int DGSolution::get_dof() const
{
	return size_basis_alpt() * ind_var_vec.size();
}
int DGSolution::size_basis_intp() const
{
	return dg.size() * pow_int(Element::PMAX_intp + 1, DIM);
}

void DGSolution::print_rhs() const
{
	for (auto const & iter : dg)
	{
		std::cout << "( ";
		for (size_t d = 0; d < DIM; d++)
		{
			std::cout << iter.second.level[d] << " ";
		}
		std::cout << ") ( ";
		for (size_t d = 0; d < DIM; d++)
		{
			std::cout << iter.second.suppt[d] << " ";
		}
		std::cout << ") ";

		for (size_t i = 0; i < VEC_NUM; i++)
		{
			for (auto const & index : iter.second.rhs[i].get_index_iterator())
			{
				std::cout << iter.second.rhs[i].at(index) << " ";
			}
		}
		std::cout << std::endl;
	}
}

void DGSolution::update_order_all_basis_in_dgmap()
{
	int order_alpt_basis_in_dgmap = 0;
	int order_intp_basis_in_dgmap = 0;
	for (auto & iter : dg)
	{
		for (size_t num_vec = 0; num_vec < ind_var_vec.size(); num_vec++)
		{
			// update order of alpert basis
			for (size_t num_basis = 0; num_basis < iter.second.size_alpt(); num_basis++)
			{
				const std::vector<int> & order_local_basis = iter.second.order_local_alpt[num_basis];
				iter.second.order_alpt_basis_in_dg[num_vec].at(order_local_basis) = order_alpt_basis_in_dgmap;
				order_alpt_basis_in_dgmap++;
			}

			// update order of interpolation basis
			for (size_t num_basis = 0; num_basis < iter.second.size_intp(); num_basis++)
			{
				const std::vector<int> & order_local_basis = iter.second.order_local_intp[num_basis];
				iter.second.order_intp_basis_in_dg[num_vec].at(order_local_basis) = order_intp_basis_in_dgmap;
				order_intp_basis_in_dgmap++;
			}
		}
	}
}




std::vector<int> DGSolution::max_mesh_level_vec() const
{
	std::vector<int> mesh_level(DIM, 0);
	for (auto const & iter : dg)
	{
		for (int d = 0; d < DIM; d++)
		{
			mesh_level[d] = std::max(mesh_level[d], iter.second.level[d]);
		}
	}
	return mesh_level;
}

int DGSolution::max_mesh_level() const
{
	int mesh_level = 0;
	for (auto const & iter : dg)
	{
		for (int d = 0; d < DIM; d++)
		{
			mesh_level = std::max(mesh_level, iter.second.level[d]);
		}
	}
	return mesh_level;
}

void DGSolution::copy_ucoe_alpt_to_f(DGSolution & E, const std::vector<int> & num_vec_f, const std::vector<int> & num_vec_E, const std::vector<int> & vel_dim_f)
{
	// check size of copy number of vector are the same for f and E
	assert(num_vec_f.size() == num_vec_E.size());
	const int copy_num_vec_size = num_vec_f.size();

	// loop over all the elements in f
	for (auto & iter_f : this->dg)
	{
		const std::vector<int> & level_f = iter_f.second.level;
		
		// sum of level of f in all the velocity dimensions
		int sum_level_f_in_vel_dim = 0;
		for (auto const vd : vel_dim_f) { sum_level_f_in_vel_dim += level_f[vd]; }

		// if sum is not zero, then there must be a non-zero index in some velocity dimension
		if (sum_level_f_in_vel_dim != 0)
		{
			for (int i = 0; i < VEC_NUM; i++)
			{
				iter_f.second.ucoe_alpt[i] = 0.;
			}
		}
		else
		{		
			// index (including mesh level and support index)
			const std::array<std::vector<int>,2> & index_f = {iter_f.second.level, iter_f.second.suppt};

			// compute hash key of the element in f and find it in E
			int hash_key_f = hash.hash_key(index_f);
			auto iter_E = E.dg.find(hash_key_f);
		
			// if the corresponding element is in f dg solution
			// then copy point value of E to f
			if (iter_E != E.dg.end())
			{
				for(int i = 0; i < copy_num_vec_size; i++)
				{
					iter_f.second.ucoe_alpt[num_vec_f[i]] = iter_E->second.ucoe_alpt[num_vec_E[i]];
				}
			}
			else
			{
				assert("error: not find element in E");
				exit(1);
			}
		}
	}
}

void DGSolution::copy_up_intp_to_f(DGSolution & E, const std::vector<int> & num_vec_f, const std::vector<int> & num_vec_E, const std::vector<int> & vel_dim_f)
{
	// check size of copy number of vector are the same for f and E
	assert(num_vec_f.size() == num_vec_E.size());
	const int copy_num_vec_size = num_vec_f.size();

	// loop over all the elements in f
	for (auto & iter_f : this->dg)
	{
		std::vector<int> level_E = iter_f.second.level;
		std::vector<int> suppt_E = iter_f.second.suppt;

		// velocity dimension should be zero level for E
		for (auto const vd : vel_dim_f)
		{
			level_E[vd] = 0;
			suppt_E[vd] = 1;
		}

		// index (including mesh level and support index)
		const std::array<std::vector<int>,2> & index_E = {level_E, suppt_E};

		// compute hash key of the element in f and find it in E
		int hash_key_E = hash.hash_key(index_E);
		auto iter_E = E.dg.find(hash_key_E);
	
		// if the corresponding element is in f dg solution
		// then copy point value of E to f
		if (iter_E != E.dg.end())
		{
			for(int i = 0; i < copy_num_vec_size; i++)
			{
				iter_f.second.up_intp_other[num_vec_f[i]] = iter_E->second.up_intp[num_vec_E[i]];
			}
		}
		else
		{
			assert("error: not find element in E");
			exit(1);
		}
	}
}

void DGSolution::set_rhs_zero()
{
	for (auto & iter : dg)
	{
		for (size_t i = 0; i < VEC_NUM; i++)
		{
			iter.second.rhs[i].set_zero();
		}
	}
}

void DGSolution::set_source_zero()
{
	for (auto & iter : dg)
	{
		for (size_t i = 0; i < VEC_NUM; i++)
		{
			iter.second.source[i].set_zero();
		}
	}
}

void DGSolution::set_ucoe_alpt_zero()
{
	for (auto & iter : dg)
	{
		for (size_t i = 0; i < VEC_NUM; i++)
		{
			iter.second.ucoe_alpt[i].set_zero();
		}
	}
}

void DGSolution::set_ucoe_alpt_zero(const int num_vec)
{
	for (auto & iter : dg)
	{
		iter.second.ucoe_alpt[num_vec].set_zero();
	}
}

void DGSolution::set_ucoe_alpt_zero(const int max_mesh, const std::vector<int> dim)
{
	for (auto & iter : dg)
	{
		bool flag = false;
		const std::vector<int> & lev = iter.second.level;
		for (auto & d : dim)
		{
			if (lev[d] > max_mesh) { flag = true; }
		}
		
		if (flag)
		{
			for (size_t i = 0; i < VEC_NUM; i++) { iter.second.ucoe_alpt[i].set_zero(); }
		}
	}
}

void DGSolution::multiply_ucoe_alpt_by_const(const double constant)
{
	for (auto & iter : dg)
	{
		for (size_t i = 0; i < VEC_NUM; i++)
		{
			iter.second.ucoe_alpt[i] *= constant;
		}
	}
}

void DGSolution::set_fp_intp_zero()
{
	for (auto & iter : dg)
	{
		for (size_t i = 0; i < VEC_NUM; i++)
		{
			for (size_t d = 0; d < DIM; d++)
			{
				iter.second.fp_intp[i][d].set_zero();
			}
		}
	}
}

void DGSolution::copy_ucoe_to_predict()
{
	for (auto & iter : dg)
	{
		for (size_t i = 0; i < VEC_NUM; i++)
		{
			iter.second.ucoe_alpt_predict[i] = iter.second.ucoe_alpt[i];
		}
	}
}

void DGSolution::copy_predict_to_ucoe()
{
	for (auto & iter : dg)
	{
		for (size_t i = 0; i < VEC_NUM; i++)
		{
			iter.second.ucoe_alpt[i] = iter.second.ucoe_alpt_predict[i];
		}
	}
}


void DGSolution::copy_ucoe_to_other()
{
	for (auto & iter : dg)
	{
		for (size_t i = 0; i < VEC_NUM; i++)
		{
			iter.second.ucoe_alpt_other[i] = iter.second.ucoe_alpt[i];
		}
	}
}

void DGSolution::copy_up_intp_to_other()
{
	for (auto & iter : dg)
	{
		for (size_t i = 0; i < VEC_NUM; i++)
		{
			iter.second.up_intp_other[i] = iter.second.up_intp[i];
		}
	}
}

void DGSolution::exchange_ucoe_and_other()
{
	// VecMultiD<double> scalar_ucoe_alpt(DIM, PMAX_alpt+1);
	VecMultiD<double> scalar_ucoe_alpt = dg.begin()->second.ucoe_alpt[0];

	for (auto & iter : dg)
	{
		for (size_t i = 0; i < VEC_NUM; i++)
		{	
			scalar_ucoe_alpt = iter.second.ucoe_alpt[i];
			iter.second.ucoe_alpt[i] = iter.second.ucoe_alpt_other[i];
			iter.second.ucoe_alpt_other[i] = scalar_ucoe_alpt;
		}
	}
}

void DGSolution::exchange_up_intp_and_other()
{
	// VecMultiD<double> scalar_up_intp(DIM, PMAX_intp+1);
	VecMultiD<double> scalar_up_intp = dg.begin()->second.up_intp[0];

	for (auto & iter : dg)
	{
		for (size_t i = 0; i < VEC_NUM; i++)
		{	
			scalar_up_intp = iter.second.up_intp[i];
			iter.second.up_intp[i] = iter.second.up_intp_other[i];
			iter.second.up_intp_other[i] = scalar_up_intp;
		}
	}
}

void DGSolution::copy_ucoe_ut_to_predict()
{
	for (auto & iter : dg)
	{
		for (size_t i = 0; i < VEC_NUM; i++)
		{
			iter.second.ucoe_ut_predict[i] = iter.second.ucoe_ut[i];
		}
	}
}

void DGSolution::copy_predict_to_ucoe_ut()
{
	for (auto & iter : dg)
	{
		for (size_t i = 0; i < VEC_NUM; i++)
		{
			iter.second.ucoe_ut[i] = iter.second.ucoe_ut_predict[i];
		}
	}
}


void DGSolution::copy_ucoe_to_predict_t_m1()
{
	for (auto & iter : dg)
	{
		for (size_t i = 0; i < VEC_NUM; i++)
		{
			iter.second.ucoe_alpt_predict_t_m1[i] = iter.second.ucoe_alpt_t_m1[i];
		}
	}
}

void DGSolution::copy_predict_to_ucoe_t_m1()
{
	for (auto & iter : dg)
	{
		for (size_t i = 0; i < VEC_NUM; i++)
		{
			iter.second.ucoe_alpt_t_m1[i] = iter.second.ucoe_alpt_predict_t_m1[i];
		}
	}
}

void DGSolution::copy_ucoe_to_ucoem1()
{
	for (auto & iter : dg)
	{
		for (size_t i = 0; i < VEC_NUM; i++)
		{
			iter.second.ucoe_alpt_t_m1[i] = iter.second.ucoe_alpt[i];
		}
	}
}

void DGSolution::copy_ucoem1_to_ucoe()
{
	for (auto & iter : dg)
	{
		for (size_t i = 0; i < VEC_NUM; i++)
		{
			iter.second.ucoe_alpt[i] = iter.second.ucoe_alpt_t_m1[i];
		}
	}
}

void DGSolution::copy_ucoe_to_ucoe_ut()
{
	for (auto & iter : dg)
	{
		for (size_t i = 0; i < VEC_NUM; i++)
		{
			iter.second.ucoe_ut[i] = iter.second.ucoe_alpt[i];
		}
	}
}

void DGSolution::copy_ucoe_ut_to_ucoe()
{
	for (auto & iter : dg)
	{
		for (size_t i = 0; i < VEC_NUM; i++)
		{
			iter.second.ucoe_alpt[i] = iter.second.ucoe_ut[i];
		}
	}
}

void DGSolution::copy_rhs_to_ucoe()
{
	for (auto & iter : dg)
	{
		for (size_t i = 0; i < VEC_NUM; i++)
		{
			iter.second.ucoe_alpt[i] = iter.second.rhs[i];
		}
	}
}

void DGSolution::add_ucoe_to_rhs()
{
	for (auto & iter : dg)
	{
		for (size_t i = 0; i < VEC_NUM; i++)
		{
			iter.second.rhs[i] += iter.second.ucoe_alpt[i];
		}
	}
}

void DGSolution::update_viscosity_intersect_element()
{
	viscosity_intersect_element.clear();

	for (auto const & visc_elem : viscosity_element)
	{
		viscosity_intersect_element.insert(visc_elem->ptr_vol_intp[0].begin(), visc_elem->ptr_vol_intp[0].end());
	}
}
