#include "DGAdapt.h"

std::vector<int> DGAdapt::indicator_var_adapt;

DGAdapt::DGAdapt(const bool sparse_, const int level_init_, const int NMAX_, AllBasis<AlptBasis> & all_bas_, AllBasis<LagrBasis> & all_bas_Lag_, AllBasis<HermBasis> & all_bas_Her_, Hash & hash_, const double eps_, const double eta_, const bool is_find_ptr_alpt_, const bool is_find_ptr_intp_, const bool is_find_ptr_general_):
		DGSolution(sparse_, level_init_, NMAX_, all_bas_, all_bas_Lag_, all_bas_Her_, hash_), 
		eps(eps_), eta(eta_), 
		is_find_ptr_alpt(is_find_ptr_alpt_), 
		is_find_ptr_intp(is_find_ptr_intp_),
		is_find_ptr_general(is_find_ptr_general_)
{
	assert(indicator_var_adapt.size() != 0);

    init_chd_par();

	update_leaf_zero_child();

    update_leaf();

    // update pointers related to DG operators
    if (is_find_ptr_alpt)
    {
        find_ptr_vol_alpt();
        find_ptr_flx_alpt();
    }

    if (is_find_ptr_intp)
    {
        find_ptr_vol_intp();
        find_ptr_flx_intp();
    }

	if (is_find_ptr_general)
	{
		find_ptr_general();
	}
}

DGAdapt::DGAdapt(const bool sparse_, const int level_init_, const int NMAX_, const int auxiliary_dim_, AllBasis<AlptBasis> & all_bas_, AllBasis<LagrBasis> & all_bas_Lag_, AllBasis<HermBasis> & all_bas_Her_, Hash & hash_, const double eps_, const double eta_, const bool is_find_ptr_alpt_, const bool is_find_ptr_intp_, const bool is_find_ptr_general_):
		DGSolution(sparse_, level_init_, NMAX_, auxiliary_dim_, all_bas_, all_bas_Lag_, all_bas_Her_, hash_), 
		eps(eps_), eta(eta_), 
		is_find_ptr_alpt(is_find_ptr_alpt_), 
		is_find_ptr_intp(is_find_ptr_intp_),
		is_find_ptr_general(is_find_ptr_general_)
{
	assert(indicator_var_adapt.size() != 0);

    init_chd_par();

	update_leaf_zero_child();

    update_leaf();

    // update pointers related to DG operators
    if (is_find_ptr_alpt)
    {
        find_ptr_vol_alpt();
        find_ptr_flx_alpt();
    }

    if (is_find_ptr_intp)
    {
        find_ptr_vol_intp();
        find_ptr_flx_intp();
    }

	if (is_find_ptr_general)
	{
		find_ptr_general();
	}
}

void DGAdapt::init_separable_scalar(std::function<double(double, int)> scalar_func)
{
	// step 1: project function in each dim to basis in 1D
	// coeff is a two dim vector with size (dim, # all_basis_1D)
	// coeff.at(d, order) denote the inner product of order-th basis function with initial function in d-th dim
    const VecMultiD<double> coeff = seperable_project(scalar_func);

	// step 2: update coefficient in each element
	// loop over each element
	for (auto &iter : dg)
	{
        init_elem_separable(iter.second, coeff);
	}
	
    // step 3: refine recursively
    refine_init_separable_scalar(coeff);

    update_order_all_basis_in_dgmap();
}

void DGAdapt::init_separable_system(std::vector<std::function<double(double, int)>> vector_func)
{
    assert(vector_func.size()==VEC_NUM);

	// step 1: project function in each dim to basis in 1D
    std::vector<VecMultiD<double>> coeff;
    for (size_t index_var = 0; index_var < VEC_NUM; index_var++)
    {
        coeff.push_back(seperable_project(vector_func[index_var]));
    }
    
	// step 2: update coefficient in each element
	// loop over each element
	for (auto &iter : dg)
	{
        for (size_t index_var = 0; index_var < VEC_NUM; index_var++)
        {
            init_elem_separable(iter.second, coeff[index_var], index_var);
        }
	}

    // step 3: refine recursively
    refine_init_separable_system(coeff);    

    update_order_all_basis_in_dgmap();
}

void DGAdapt::init_separable_scalar_sum(std::vector<std::function<double(double, int)>> sum_func)
{
    const int num_func = sum_func.size();    
    
	// step 1: project function in each dim to basis in 1D
    std::vector<VecMultiD<double>> coeff;
    for (size_t i_func = 0; i_func < num_func; i_func++)
    {
        // coefficient for i-th function
        const VecMultiD<double> coeff_i_func = seperable_project(sum_func[i_func]);
        coeff.push_back(coeff_i_func);
    }

	// step 2: update coefficient in each element
	// loop over each element
	for (auto &iter : dg)
	{
        for (size_t i_func = 0; i_func < num_func; i_func++)
        {
            init_elem_separable(iter.second, coeff[i_func]);
        }
	}

    // step 3: refine recursively
    refine_init_separable_scalar_sum(coeff);

    update_order_all_basis_in_dgmap();
}

void DGAdapt::init_separable_system_sum(std::vector<std::vector<std::function<double(double, int)>>> sum_vector_func)
{
	assert(sum_vector_func.size() == ind_var_vec.size() && ind_var_vec.size() <= VEC_NUM);

	// step 1: project function in each dim to basis in 1D
	std::vector<std::vector<VecMultiD<double>>> coeff;

	//std::cout << dg.size() << std::endl;

	//for (size_t index_var = 0; index_var < VEC_NUM; index_var++) // the outside loop of the unknown variables
	for (size_t index_var = 0; index_var < ind_var_vec.size(); index_var++) // the outside loop of the unknown variables
	{
		std::vector<VecMultiD<double>> coeff_var;

		const int num_func = sum_vector_func[index_var].size(); // Wei: I think here is confusing; a map may be a better choice
		for (size_t i_func = 0; i_func < num_func; i_func++)
		{
			// coefficient for i-th function
			const VecMultiD<double> coeff_i_func = seperable_project(sum_vector_func[index_var][i_func]);
			coeff_var.push_back(coeff_i_func);
		}
		coeff.push_back(coeff_var);
	}

	// step 2: update coefficient in each element
	// loop over each element
	for (auto &iter : dg)
	{
		for (size_t index_var = 0; index_var < ind_var_vec.size(); index_var++)
		{
			const int num_func = sum_vector_func[index_var].size();
			for (size_t i_func = 0; i_func < num_func; i_func++)
			{
				init_elem_separable(iter.second, coeff[index_var][i_func], ind_var_vec[index_var]);
			}
		}
	}



	// step 3: refine recursively
	refine_init_separable_system_sum(coeff);

	update_order_all_basis_in_dgmap();
}


void DGAdapt::compute_moment_full_grid(DGAdapt & f, const std::vector<int> & moment_order, const std::vector<double> & moment_order_weight, const int num_vec)
{	
	// f should be a scalar function
	assert(f.VEC_NUM == 1);
	assert(moment_order.size() == moment_order_weight.size());

	const int moment_order_size = moment_order.size();
	assert(moment_order_size <= 3);
	for (int i = 0; i < moment_order_size; i++)
	{
		assert((moment_order[i] == i) && ("moment_order in DGAdapt::compute_moment_full_grid() is not monotone"));
	}
	
	// loop over all the elements in the current dg solution (represent E or B in Maxwell equations)
	for (auto & iter : dg)
	{
		// index (including mesh level and support index)
		const std::array<std::vector<int>,2> & index = {iter.second.level, iter.second.suppt};

		// compute hash key and find it in distribution function f
		int hash_key_f = hash.hash_key(index);
		auto iter_f = f.dg.find(hash_key_f);
		
		if (iter_f != f.dg.end())
		{
			for (int deg = 0; deg < AlptBasis::PMAX + 1; deg++)
			{
				if (moment_order_size == 1)
				{
					iter.second.rhs[num_vec].at(deg, 0) += iter_f->second.ucoe_alpt[0].at(deg, 0) * moment_order_weight[0];
				}
				else if (moment_order_size == 2)
				{
					iter.second.rhs[num_vec].at(deg, 0) += iter_f->second.ucoe_alpt[0].at(deg, 0) * (moment_order_weight[0] + 1./2. * moment_order_weight[1])
														 + iter_f->second.ucoe_alpt[0].at(deg, 1) * 1./(2.*sqrt(3.)) * moment_order_weight[1];
				}
				else if (moment_order_size == 3)
				{
					iter.second.rhs[num_vec].at(deg, 0) += iter_f->second.ucoe_alpt[0].at(deg, 0) * (moment_order_weight[0] + 1./2. * moment_order_weight[1] + 1./3. * moment_order_weight[2])
														 + iter_f->second.ucoe_alpt[0].at(deg, 1) * (1./(2.*sqrt(3.)) * moment_order_weight[1] + 1./(2.*sqrt(3.)) * moment_order_weight[2])
														 + iter_f->second.ucoe_alpt[0].at(deg, 2) * 1./(6.*sqrt(5.)) * moment_order_weight[2];
				}
			}
		}
	}
}

void DGAdapt::compute_moment_1D2V(DGAdapt & f, const std::vector<int> & moment_order, const double moment_order_weight, const int num_vec_EB, const int num_vec_f)
{
	// f = f(x2, v1, v2)
	assert(f.DIM == 3);
	assert(moment_order.size() == 2);
	for (int i = 0; i < moment_order.size(); i++) { assert(moment_order[i] <= 1); }

	const int moment_order_v1 = moment_order[0];
	const int moment_order_v2 = moment_order[1];
	
	// loop over all the elements in the current dg solution (represent E or B in Maxwell equations)
	for (auto & iter : dg)
	{
		// index (including mesh level and support index)
		const std::array<std::vector<int>,2> & index = {iter.second.level, iter.second.suppt};

		// compute hash key and find it in distribution function f
		int hash_key_f = hash.hash_key(index);
		auto iter_f = f.dg.find(hash_key_f);
		
		if (iter_f != f.dg.end())
		{
			for (int deg = 0; deg < AlptBasis::PMAX + 1; deg++)
			{
				if (moment_order_v1==0 && moment_order_v2==0)
				{
					iter.second.rhs[num_vec_EB].at(deg, 0, 0) += iter_f->second.ucoe_alpt[num_vec_f].at(deg, 0, 0) * moment_order_weight;
				}
				else if (moment_order_v1==1 && moment_order_v2==0)
				{
					iter.second.rhs[num_vec_EB].at(deg, 0, 0) += ( iter_f->second.ucoe_alpt[num_vec_f].at(deg, 0, 0) * 1./2.
																+ iter_f->second.ucoe_alpt[num_vec_f].at(deg, 1, 0) * 1./(2.*sqrt(3.)) )* moment_order_weight;
				}
				else if (moment_order_v1==0 && moment_order_v2==1)
				{
					iter.second.rhs[num_vec_EB].at(deg, 0, 0) += ( iter_f->second.ucoe_alpt[num_vec_f].at(deg, 0, 0) * 1./2.
																+ iter_f->second.ucoe_alpt[num_vec_f].at(deg, 0, 1) * 1./(2.*sqrt(3.)) )* moment_order_weight;
				}
				else
				{
					std::cout << "error in DGAdapt::compute_moment_1D2V()" << std::endl; exit(1);
				}
			}
		}
	}
}

void DGAdapt::compute_moment_2D2V(DGAdapt & f, const std::vector<int> & moment_order, const double moment_order_weight, const int num_vec_EB, const int num_vec_f)
{
	// f = f(x1, x2, v1, v2)
	assert(f.DIM == 4);
	assert(moment_order.size() == 2);
	for (int i = 0; i < moment_order.size(); i++) { assert(moment_order[i] <= 1); }

	const int moment_order_v1 = moment_order[0];
	const int moment_order_v2 = moment_order[1];
	
	// loop over all the elements in the current dg solution (represent E or B in Maxwell equations)
	for (auto & iter : dg)
	{
		// index (including mesh level and support index)
		const std::array<std::vector<int>,2> & index = {iter.second.level, iter.second.suppt};

		// compute hash key and find it in distribution function f
		int hash_key_f = hash.hash_key(index);
		auto iter_f = f.dg.find(hash_key_f);
		
		if (iter_f != f.dg.end())
		{
			for (int deg_x1 = 0; deg_x1 < AlptBasis::PMAX + 1; deg_x1++)
			{
				for (int deg_x2 = 0; deg_x2 < AlptBasis::PMAX + 1; deg_x2++)
				{
					if (moment_order_v1==0 && moment_order_v2==0)
					{
						iter.second.rhs[num_vec_EB].at(deg_x1, deg_x2, 0, 0) += iter_f->second.ucoe_alpt[num_vec_f].at(deg_x1, deg_x2, 0, 0) * moment_order_weight;
					}
					else if (moment_order_v1==1 && moment_order_v2==0)
					{
						iter.second.rhs[num_vec_EB].at(deg_x1, deg_x2, 0, 0) += ( iter_f->second.ucoe_alpt[num_vec_f].at(deg_x1, deg_x2, 0, 0) * 1./2.
																	+ iter_f->second.ucoe_alpt[num_vec_f].at(deg_x1, deg_x2, 1, 0) * 1./(2.*sqrt(3.)) )* moment_order_weight;
					}
					else if (moment_order_v1==0 && moment_order_v2==1)
					{
						iter.second.rhs[num_vec_EB].at(deg_x1, deg_x2, 0, 0) += ( iter_f->second.ucoe_alpt[num_vec_f].at(deg_x1, deg_x2, 0, 0) * 1./2.
																	+ iter_f->second.ucoe_alpt[num_vec_f].at(deg_x1, deg_x2, 0, 1) * 1./(2.*sqrt(3.)) )* moment_order_weight;
					}
					else
					{
						std::cout << "error in DGAdapt::compute_moment_2D2V()" << std::endl; exit(1);
					}
				}
			}
		}
	}
}

void DGAdapt::adapt_f_base_on_E(DGAdapt & E)
{
	// loop over all the elements in E
	for (auto & iter_E : E.dg)
	{
		// index (including mesh level and support index)
		const std::array<std::vector<int>,2> & index_E = {iter_E.second.level, iter_E.second.suppt};

		// compute hash key of the element in E and find it in f
		int hash_key_E = hash.hash_key(index_E);
		auto iter_f = this->dg.find(hash_key_E);

		// if the corresponding element is not in f dg solution
		// then add new element in f and then copy point value of E to f	
		if (iter_f == this->dg.end())
		{
			Element elem(iter_E.second.level, iter_E.second.suppt, all_bas, hash);
			this->add_elem(elem);
		}
	}

    check_hole();

    update_leaf();

    update_leaf_zero_child();

    update_order_all_basis_in_dgmap();
}

// the key member function in the DGAdapt class
void DGAdapt::refine()
{
    // before refine, set new_add variable to be false in all elements
    // this variable will be used to save computational cost in the first stage of RK2 or RK3 method
    set_all_new_add_false();

    for (auto & iter : leaf)
    {
        if (indicator_norm(*(iter.second)) > eps) // l2 norm
        {
            // loop over all its children
            const std::set<std::array<std::vector<int>,2>> index_chd_elem = index_all_chd(iter.second->level, iter.second->suppt);
            for (auto const & index : index_chd_elem)
            {
                int hash_key = hash.hash_key(index);
                
                // if index is not in current child index set, then add it to dg solution
                if ( iter.second->hash_ptr_chd.find(hash_key) == iter.second->hash_ptr_chd.end())
                {
                    Element elem(index[0], index[1], all_bas, hash);
                    add_elem(elem);
                }                
            }
        }
    }
    
    check_hole();

    update_leaf();

    update_leaf_zero_child();

    update_order_all_basis_in_dgmap();
}

void DGAdapt::damping_leaf(const double damp_coef)
{
	for (auto & iter : this->dg)
	{	
		if ((iter.second.level[0] == NMAX))
		{
			iter.second.ucoe_alpt[0] *= damp_coef;
		}
	}

	for (auto & i : viscosity_element)	
	{
		// auto iter = *i;
		// print(iter.level[0]);
		// iter.ucoe_alpt[0] *= damp_coef;
		// for (auto & iter_par : iter.hash_ptr_par)
		// {
		// 	iter_par.second->ucoe_alpt[0] *= damp_coef;
		// }

		// if ((iter.second.level[0] == NMAX))
		// {
		// 	iter.second.ucoe_alpt[0] *= damp_coef;
		// }
        // // max mesh level of this element in all dimensions
        // const int max_mesh_level = *(std::max_element(std::begin(iter.second->level), std::end(iter.second->level)));

        // // only consider element in finest mesh level
        // if (max_mesh_level == NMAX)
		// {	
		// 	// iter.second->ucoe_alpt[0].print();
		// 	iter.second->ucoe_alpt[0] *= damp_coef;

		// 	// for (auto & iter_par : iter.second->hash_ptr_par)
		// 	// {
		// 	// 	iter_par.second->ucoe_alpt[0] *= damp_coef;
		// 	// }
		// }

		// // loop over all its parent elements
		// const std::set<std::array<std::vector<int>,2>> index_chd_elem = index_all_par(iter.second->level, iter.second->suppt);
		// for (auto const & index : index_chd_elem)
		// {
		// 	int hash_key = hash.hash_key(index);
			
		// 	// if index is not in current child index set, then add it to dg solution
		// 	if ( iter.second->hash_ptr_chd.find(hash_key) == iter.second->hash_ptr_chd.end())
		// 	{
		// 		Element elem(index[0], index[1], all_bas, hash);
		// 		add_elem(elem);
		// 	}                
		// }			
	}
}

void DGAdapt::filter(const double damp_coef, const std::vector<double> & wave_speed, const double dt, const int filter_start_level_sum)
{
	const double coefficient = damp_coef * (dt * dt / 2.0) * (M_PI * M_PI);
	
	for (auto & iter : this->dg)
	{	
		// compute the sum of mesh level
		const int level_sum = std::accumulate(iter.second.level.begin(), iter.second.level.end(), 0);

		if (level_sum >= filter_start_level_sum)
		{
			// compute c_1 * 2^(l_1) + c_2 * 2^(l_2) + ... + c_d * 2^(l_d)
			// where c_i is the wave speed in i-th dimension
			// and l_i is the mesh level in i-th dimension
			double index_sum = 0.0;
			for (int d = 0; d < DIM; d++)
			{
				index_sum += wave_speed[d] * pow(2.0, iter.second.level[d]);
			}
			iter.second.ucoe_alpt[0] *= exp(- coefficient * index_sum * index_sum);
		}
	}
}

void DGAdapt::filter_local(const double damp_coef, std::function<double(std::vector<double>, int)> wave_speed_func, const double dt, const int filter_start_level_sum)
{
	const double coefficient = damp_coef * (dt * dt / 2.0) * (M_PI * M_PI);
	
	for (auto & iter : this->dg)
	{	
		// compute the sum of mesh level
		const int level_sum = std::accumulate(iter.second.level.begin(), iter.second.level.end(), 0);

		if (level_sum >= filter_start_level_sum)
		{
			// local wave speed
			std::vector<double> local_wave_speed(DIM, 0.0);
			
			// use iter.second.xl and iter.second.xr to generate corner points
			std::vector<std::vector<double>> corner_points;
			if (DIM == 2)
			{
				for (int i = 0; i < 2; i++)
				{
					for (int j = 0; j < 2; j++)
					{
						std::vector<double> point(2);
						point[0] = (i == 0) ? iter.second.xl[0] : iter.second.xr[0];
						point[1] = (j == 0) ? iter.second.xl[1] : iter.second.xr[1];
						corner_points.push_back(point);
					}
				}
			}
			else if (DIM == 3)
			{
				for (int i = 0; i < 2; i++)
				{
					for (int j = 0; j < 2; j++)
					{
						for (int k = 0; k < 2; k++)
						{
							std::vector<double> point(3);
							point[0] = (i == 0) ? iter.second.xl[0] : iter.second.xr[0];
							point[1] = (j == 0) ? iter.second.xl[1] : iter.second.xr[1];
							point[2] = (k == 0) ? iter.second.xl[2] : iter.second.xr[2];
							corner_points.push_back(point);
						}
					}
				}
			}
			else
			{
				std::cout << "filter_local() does not support DIM > 3" << std::endl;
				exit(1);
			}

			for (auto const & pt : corner_points)
			{
				for (int d = 0; d < DIM; d++)
				{	
					local_wave_speed[d] = std::max(abs(wave_speed_func(pt, d)), local_wave_speed[d]);
				}
			}

			// compute c_1 * 2^(l_1) + c_2 * 2^(l_2) + ... + c_d * 2^(l_d)
			// where c_i is the wave speed in i-th dimension
			// and l_i is the mesh level in i-th dimension
			double index_sum = 0.0;
			for (int d = 0; d < DIM; d++)
			{
				index_sum += local_wave_speed[d] * pow(2.0, iter.second.level[d]);
			}
			iter.second.ucoe_alpt[0] *= exp(- coefficient * index_sum * index_sum);
		}
	}
}

// void DGAdapt::filter_viscosity(const double dt, const double dx, const int convergence_order, const double l2_norm_previous, const double amplify_factor)
// {	
// 	// compute the coefficient in the filter
// 	double coefficient = 0.0;

// 	const int max_iteration = 100;
// 	int iteration = 0;
// 	while (true)
// 	{
// 		double coefficient_previous = coefficient;

// 		double l2_norm = 0.0;
// 		double l2_norm_derivative = 0.0;
// 		for (auto const & iter : dg)
// 		{
// 			double index_sum = 0.0;
// 			for (int d = 0; d < DIM; d++)			
// 			{
// 				if (iter.second.level[d] != 0)
// 				{
// 					index_sum += M_PI * M_PI * pow(2.0, 2.0 * iter.second.level[d]);
// 				}				
// 			}

// 			double l2_norm_local = exp(- 2.0 * coefficient * dt * pow(dx, convergence_order) * index_sum) * iter.second.get_L2_norm_element()[0];
// 			l2_norm += l2_norm_local;
// 			l2_norm_derivative += (- 2.0 * dt * pow(dx, convergence_order) * index_sum) * l2_norm_local;
// 		}
// 		l2_norm -= l2_norm_previous;

// 		coefficient = coefficient - l2_norm/l2_norm_derivative;

// 		if (abs(coefficient - coefficient_previous) < 1.0e-10) { break; }
// 		iteration += 1;

// 		if (iteration > max_iteration)
// 		{
// 			std::cout << "filter_viscosity() does not converge" << std::endl;
// 			break;
// 		}
// 	}

// 	coefficient *= amplify_factor;

// 	for (auto & iter : this->dg)
// 	{	
// 		double index_sum = 0.0;
// 		for (int d = 0; d < DIM; d++)
// 		{
// 			if (iter.second.level[d] != 0)
// 			{
// 				index_sum += M_PI * M_PI * pow(2.0, 2.0 * iter.second.level[d]);
// 			}
// 		}
// 		iter.second.ucoe_alpt[0] *= exp(- coefficient * dt * pow(dx, convergence_order) * index_sum);
// 	}
// }

// NOT TESTED
void DGAdapt::refine(const int max_mesh, const std::vector<int> dim)
{
	// generate vector narray as mesh level index, each element is a vector of size dim: ( n1, n2, ..., n_(dim) ), 0 <= n_i <= NMAX for any 1 <= i <= dim
	std::vector<std::vector<int>> narray;
	std::vector<int> narray_max(DIM, 0);
	for (int i = 0; i < dim.size(); i++)
	{
		int d = dim[i];
		narray_max[d] = max_mesh + 1;
	}
	IterativeNestedLoop(narray, DIM, narray_max); // generate a full-grid mesh level index; avoid loop of the dimension

	for (auto const & lev_n : narray) 
	{
		int lev_n_max = 0;
		for (auto & l : lev_n) { lev_n_max = std::max(lev_n_max, l); }

		if (lev_n_max < max_mesh) { continue; }

		// loop over all mesh level array of size dim
		// full grid, generate index j: 0, 1, ..., 2^(n-1)-1, for n >= 2. Otherwise j=0 for n=0,1
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

			// index (including mesh level and support index)
			const std::array<std::vector<int>,2> & index = {lev_n, odd_j};

			// compute hash key of the element in E and find it in f
			int hash_key_E = hash.hash_key(index);
			auto iter_E = this->dg.find(hash_key_E);
			
			if (iter_E == this->dg.end())
			{
				Element elem(lev_n, odd_j, all_bas, hash);
				this->dg.insert({ elem.hash_key, elem });
			}
		}
	}
	
	update_order_all_basis_in_dgmap();
}

// the key member function in the DGAdapt class to coarsen the dg solution
void DGAdapt::coarsen()
{
    leaf.clear();

    // update no child leaf
    update_leaf_zero_child();    

    coarsen_no_leaf();

    // update leaf
    update_leaf();

    update_order_all_basis_in_dgmap();
}

// NOT TESTED
void DGAdapt::coarsen(const int max_mesh, const std::vector<int> dim)
{
	for (auto & iter : dg)
	{
		bool flag = false;
		const std::vector<int> & mesh_level = iter.second.level;

		// if mesh level is larger than the given mesh level, then flag is true
		for (int i = 0; i < dim.size(); ++i)
		{	
			int d = dim[i];
			if (mesh_level[d] > max_mesh) { flag = true; }
		}

		if (flag)
		{
			Element* elem = &(iter.second);
			del_elem(*elem);
		}
	}

	update_order_all_basis_in_dgmap();
}

bool DGAdapt::check_total_num_chd_par_equal() const
{
    int total_num_chd = 0;
    int total_num_par = 0;
    for (auto const & iter : dg)
	{
        total_num_chd += iter.second.hash_ptr_chd.size();
        total_num_par += iter.second.hash_ptr_par.size();
	}
    return (total_num_chd==total_num_par);
}

void DGAdapt::update_viscosity_element(const double shock_kappa)
{   
    viscosity_element.clear();

    // loop over all elements that have no child
    for (auto const & iter : leaf_zero_child)
    {   
        // max mesh level of this element in all dimensions
        const int max_mesh_level = *(std::max_element(std::begin(iter.second->level), std::end(iter.second->level)));

        // only consider element in finest mesh level
        if (max_mesh_level != NMAX) { continue; }

        // l2 norm of solution in this element
        const double num_l2_norm = indicator_norm(*(iter.second));
        
        // l1 norm of mesh level
        const int level_l1_norm = std::accumulate(iter.second->level.begin(), iter.second->level.end(), 0);

        // theoretical bound for smooth solutions
        // const double theory_bound = pow(2., -(AlptBasis::PMAX+0.5)*level_l1_norm);
        const double theory_bound = pow(2., (-AlptBasis::PMAX+1.0)*level_l1_norm);

        // if l2 norm of solution in this element is larger than theoretical bound for smooth solutions
        // then add it to viscosity element 
        if (log10(num_l2_norm) - log10(theory_bound) > shock_kappa)
        {
            viscosity_element.insert(iter.second);
        }
    }
}

//------------------------------------------------------------
// 
// below are private member functions
// 
//------------------------------------------------------------

void DGAdapt::refine_init_separable_scalar(const VecMultiD<double> & coeff)
{
    bool flag = false;

    for (auto & iter : leaf)
    {
        if (indicator_norm(*(iter.second)) > eps)
        {
            flag = true;

            // loop over all its children
            const std::set<std::array<std::vector<int>,2>> index_chd_elem = index_all_chd(iter.second->level, iter.second->suppt);
            for (auto const & index : index_chd_elem)
            {
                int hash_key = hash.hash_key(index);
                
                // if index is not in current child index set, then add it to dg solution
                if ( iter.second->hash_ptr_chd.find(hash_key) == iter.second->hash_ptr_chd.end())
                {
                    Element elem(index[0], index[1], all_bas, hash);
                    init_elem_separable(elem, coeff);
                    add_elem(elem);                    
                }
            }
        }
    }

    if (flag)
    {
        check_hole_init_separable_scalar(coeff);

        update_leaf();

        refine_init_separable_scalar(coeff);
    }
    else
    {
        return;
    }
}

void DGAdapt::refine_init_separable_system(const std::vector<VecMultiD<double>> & coeff)
{
    bool flag = false;

    for (auto & iter : leaf)
    {
        if (indicator_norm(*(iter.second)) > eps)
        {
            flag = true;

            // loop over all its children
            const std::set<std::array<std::vector<int>,2>> index_chd_elem = index_all_chd(iter.second->level, iter.second->suppt);
            for (auto const & index : index_chd_elem)
            {
                int hash_key = hash.hash_key(index);
                
                // if index is not in current child index set, then add it to dg solution
                if ( iter.second->hash_ptr_chd.find(hash_key) == iter.second->hash_ptr_chd.end())
                {
                    Element elem(index[0], index[1], all_bas, hash);
                    for (size_t index_var = 0; index_var < VEC_NUM; index_var++)
                    {
                        init_elem_separable(elem, coeff[index_var], index_var);                        
                    }
                    add_elem(elem);
                }
            }
        }
    }

    if (flag)
    {
        check_hole_init_separable_system(coeff);

        update_leaf();

        refine_init_separable_system(coeff);
    }
    else
    {
        return;
    }
}

void DGAdapt::refine_init_separable_scalar_sum(const std::vector<VecMultiD<double>> & coeff)
{
    const int num_func = coeff.size();

    bool flag = false;

    for (auto & iter : leaf)
    {
        if (indicator_norm(*(iter.second)) > eps)
        {
            flag = true;

            // loop over all its children
            const std::set<std::array<std::vector<int>,2>> index_chd_elem = index_all_chd(iter.second->level, iter.second->suppt);
            for (auto const & index : index_chd_elem)
            {
                int hash_key = hash.hash_key(index);
                
                // if index is not in current child index set, then add it to dg solution
                if ( iter.second->hash_ptr_chd.find(hash_key) == iter.second->hash_ptr_chd.end())
                {
                    Element elem(index[0], index[1], all_bas, hash);
                    for (size_t i_func = 0; i_func < num_func; i_func++)
                    {
                        init_elem_separable(elem, coeff[i_func]);
                    }                    
                    add_elem(elem);                    
                }
            }
        }
    }

    if (flag)
    {
        check_hole_init_separable_scalar_sum(coeff);

        update_leaf();

        refine_init_separable_scalar_sum(coeff);
    }
    else
    {
        return;
    }
}

void DGAdapt::refine_init_separable_system_sum(const std::vector<std::vector<VecMultiD<double>>> & coeff)
{
    bool flag = false;

    for (auto & iter : leaf)
    {
        if (indicator_norm(*(iter.second)) > eps)
        {
            flag = true;

            // loop over all its children
            const std::set<std::array<std::vector<int>,2>> index_chd_elem = index_all_chd(iter.second->level, iter.second->suppt);
            for (auto const & index : index_chd_elem)
            {
                int hash_key = hash.hash_key(index);
                
                // if index is not in current child index set, then add it to dg solution
                if ( iter.second->hash_ptr_chd.find(hash_key) == iter.second->hash_ptr_chd.end())
				{
					Element elem(index[0], index[1], all_bas, hash);
					for (size_t index_var = 0; index_var < ind_var_vec.size(); index_var++) // the outside loop of the unknown variables
					{
						for (size_t i_func = 0; i_func < coeff[index_var].size(); i_func++)
						{
							init_elem_separable(elem, coeff[index_var][i_func], index_var);
						}
					}

					add_elem(elem);
					//static int icount = 0;
					//icount++;
					//std::cout << icount << std::endl;
					//if (icount == 384)
					//	std::cout << "debug" << std::endl;
				}
			}
		}
	}

	if (flag)
	{
		check_hole_init_separable_system_sum(coeff);

		update_leaf();

		refine_init_separable_system_sum(coeff);
	}
	else
	{
		return;
	}
}

// initial number and pointers of parents and children
void DGAdapt::init_chd_par()
{
	for (auto & iter : dg)
	{
		iter.second.num_total_chd = num_all_chd(iter.second.level);
		iter.second.num_total_par = num_all_par(iter.second.level);

		const std::set<std::array<std::vector<int>, 2>> index_chd_elem = index_all_chd(iter.second.level, iter.second.suppt);
		for (auto const & index : index_chd_elem)
		{
			const int hash_key_chd_elem = hash.hash_key(index);
			auto it_chd = dg.find(hash_key_chd_elem);
			if (it_chd != dg.end())
			{
				iter.second.hash_ptr_chd.insert({ hash_key_chd_elem, &(it_chd->second) });
			}
		}

		const std::set<std::array<std::vector<int>, 2>> index_par_elem = index_all_par(iter.second.level, iter.second.suppt);
		for (auto const & index : index_par_elem)
		{
			const int hash_key_par_elem = hash.hash_key(index);
			auto it_par = dg.find(hash_key_par_elem);
			if (it_par != dg.end())
			{
				iter.second.hash_ptr_par.insert({ hash_key_par_elem, &(it_par->second) });
			}
		}
	}
}

// the key function to recursively coarsen the DGsolution
void DGAdapt::coarsen_no_leaf()
{
	bool flag = false;


	// loop over all leaf with 0 child
	for (auto it = leaf_zero_child.begin(); it != leaf_zero_child.end(); )
	{
		// if satisfy this criterion and mesh level larger than 0
		// to avoid delete element of level 0 and thus result in zero number of basis when initial value are close to zero
		if ((indicator_norm(*(it->second)) < eta) &&
			(*std::max_element(it->second->level.begin(), it->second->level.end()) > 0))
		{
			flag = true;
			Element* elem = it->second;
			it = leaf_zero_child.erase(it);
			del_elem(*elem);
		}
		else
		{
			++it;
		}
	}

	if (flag)
	{
		update_leaf_zero_child();

		coarsen_no_leaf();
	}
	else
	{
		return;
	}
}

double DGAdapt::indicator_norm(const Element & elem) const
{
	double norm = 0.;

	const int sz = elem.ucoe_alpt[0].size();  // size of ucoefficient for each component
	for (auto const ind_vec : indicator_var_adapt)
	{
		norm += elem.ucoe_alpt[ind_vec].norm();
		if (prob == "wave") { norm += elem.ucoe_ut[ind_vec].norm(); }
	}

	return norm;
}


// update leaf (based on num of children)
// if num of existing children is less than num of total children
// then add it to leaf element
void DGAdapt::update_leaf()
{
	leaf.clear();

	for (auto & iter : dg)
	{
		if (iter.second.hash_ptr_chd.size() < iter.second.num_total_chd)
		{
			leaf.insert({ iter.second.hash_key, &(iter.second) });
		}
	}
}

// update leaf with 0 child (based on num of children)
// if num of existing children is 0
// then add it to leaf with 0 child
void DGAdapt::update_leaf_zero_child()
{
	leaf_zero_child.clear();

	for (auto & iter : dg)
	{
		if (iter.second.hash_ptr_chd.size() == 0)
		{
			leaf_zero_child.insert({ iter.second.hash_key, &(iter.second) });
		}
	}
}

void DGAdapt::set_all_new_add_false()
{
	for (auto & iter : dg)
	{
		iter.second.new_add = false;
	}
}

void DGAdapt::add_elem(Element & elem)
{
	// add it into dg solution, return iterator pointing to new element
	auto new_iter = dg.insert(std::make_pair(elem.hash_key, elem)).first;

	new_iter->second.num_total_par = num_all_par(new_iter->second.level);
	new_iter->second.num_total_chd = num_all_chd(new_iter->second.level);
	new_iter->second.new_add = true;

	// find parent of new add element
	const std::set<std::array<std::vector<int>, 2>> & index_all_par_elem = index_all_par(new_iter->second.level, new_iter->second.suppt);
	for (auto const & index : index_all_par_elem)
	{
		int hash_key = hash.hash_key(index);
		auto it = dg.find(hash_key);
		if (it != dg.end())
		{
			it->second.hash_ptr_chd.insert({ new_iter->second.hash_key, &(new_iter->second) });
			new_iter->second.hash_ptr_par.insert({ it->second.hash_key, &(it->second) });
		}
	}

	// find child of new add element
	const std::set<std::array<std::vector<int>, 2>> & index_all_chd_elem = index_all_chd(new_iter->second.level, new_iter->second.suppt);
	for (auto const & index : index_all_chd_elem)
	{
		int hash_key = hash.hash_key(index);
		auto it = dg.find(hash_key);
		if (it != dg.end())
		{
			new_iter->second.hash_ptr_chd.insert({ it->second.hash_key, &(it->second) });
			it->second.hash_ptr_par.insert({ new_iter->second.hash_key, &(new_iter->second) });
		}
	}

	if (is_find_ptr_alpt)
	{
		// first initialize pointers with empty set
		std::unordered_set<Element*> empty_set;
		for (size_t d = 0; d < DIM; d++)
		{
			new_iter->second.ptr_vol_alpt.push_back(empty_set);
			new_iter->second.ptr_flx_alpt.push_back(empty_set);
		}

		for (auto & iter_solu : dg)
		{
			for (size_t d = 0; d < DIM; d++)
			{
				// vol                
				if (new_iter->second.is_vol_alpt(iter_solu.second, d))
				{
					new_iter->second.ptr_vol_alpt[d].insert(&(iter_solu.second));
					iter_solu.second.ptr_vol_alpt[d].insert(&(new_iter->second));
				}
				// flx
				if (new_iter->second.is_flx_alpt(iter_solu.second, d))
				{
					new_iter->second.ptr_flx_alpt[d].insert(&(iter_solu.second));
					iter_solu.second.ptr_flx_alpt[d].insert(&(new_iter->second));
				}
			}
		}
	}

	if (is_find_ptr_intp)
	{
		// first initialize pointers with empty set
		std::unordered_set<Element*> empty_set;
		for (size_t d = 0; d < DIM; d++)
		{
			new_iter->second.ptr_vol_intp.push_back(empty_set);
			new_iter->second.ptr_flx_intp.push_back(empty_set);
		}

		for (auto & iter_solu : dg)
		{
			// vol, independent of dim
			if (new_iter->second.is_vol_intp(iter_solu.second))
			{
				for (size_t d = 0; d < DIM; d++)
				{
					new_iter->second.ptr_vol_intp[d].insert(&(iter_solu.second));
					iter_solu.second.ptr_vol_intp[d].insert(&(new_iter->second));
				}
			}

			for (size_t d = 0; d < DIM; d++)
			{
				// flx
				if (new_iter->second.is_flx_intp(iter_solu.second, d))
				{
					new_iter->second.ptr_flx_intp[d].insert(&(iter_solu.second));
					iter_solu.second.ptr_flx_intp[d].insert(&(new_iter->second));
				}
			}
		}
	}

	if (is_find_ptr_general)
	{
		for (auto & iter_solu : dg)
		{
			if (new_iter->second.is_element_multidim_intersect_adjacent(iter_solu.second))
			{
				new_iter->second.ptr_general.insert(&(iter_solu.second));
				iter_solu.second.ptr_general.insert(&(new_iter->second));
			}
		}
	}
}

void DGAdapt::del_elem(Element & elem)
{
	// find parent of deleted element and erase hash key of deleted element
	for (auto & iter : elem.hash_ptr_par)
	{
		iter.second->hash_ptr_chd.erase(elem.hash_key);
	}

	if (is_find_ptr_alpt)
	{
		for (size_t d = 0; d < DIM; d++)
		{
			for (auto & ptr : elem.ptr_vol_alpt[d])
			{
				// if pointer points to this element itself, then do not erase it
				// since we do not want to change elem.ptr_vol_intp in this loop                
				if (ptr == &elem) continue;

				ptr->ptr_vol_alpt[d].erase(&elem);
			}
			for (auto & ptr : elem.ptr_flx_alpt[d])
			{
				if (ptr == &elem) continue;

				ptr->ptr_flx_alpt[d].erase(&elem);
			}
		}
	}

	if (is_find_ptr_intp)
	{
		for (size_t d = 0; d < DIM; d++)
		{
			for (auto & ptr : elem.ptr_vol_intp[d])
			{
				// if pointer points to this element itself, then do not erase it
				// since we do not want to change elem.ptr_vol_intp in this loop
				if (ptr == &elem) continue;

				ptr->ptr_vol_intp[d].erase(&elem);
			}
			for (auto & ptr : elem.ptr_flx_intp[d])
			{
				if (ptr == &elem) continue;

				ptr->ptr_flx_intp[d].erase(&elem);
			}
		}
	}

	if (is_find_ptr_general)
	{
		for (auto & ptr : elem.ptr_general)
		{
			// if pointer points to this element itself, then do not erase it
			// since we do not want to change elem.ptr_vol_intp in this loop
			if (ptr == &elem) continue;

			ptr->ptr_general.erase(&elem);
		}
	}

	dg.erase(elem.hash_key);
}

std::set<std::array<std::vector<int>, 2>> DGAdapt::index_all_par(const std::vector<int> & lev_n, const std::vector<int> & sup_j)
{
	std::set<std::array<std::vector<int>, 2>> index_par_elem;

	for (int d = 0; d < DIM; ++d)
	{
		if (lev_n[d] == 0) continue;
		std::vector<int> n = lev_n;
		n[d]--;

		std::vector<int> j = sup_j;
		int j_half = (sup_j[d] + 1) / 2;
		if (j_half % 2 == 1)
		{
			j[d] = j_half;
		}
		else
		{
			j[d] = j_half - 1;
		}

		std::array<std::vector<int>, 2> index{ n, j };
		index_par_elem.insert(index);
	}
	return index_par_elem;
}

// child element of (n1,n2)(j1,j2) (index n and j, in 2D case) is
// x direction (keep y direction unchaged): (n1+1,n2)(2*j1-1,j2) and (n1+1,n2)*(2*j1+1,j2)
// y direction (keep x direction unchaged): (n1,n2+1)*(j1,2*j2-1) and (n1,n2+1)*(j1,2*j2+1)
std::set<std::array<std::vector<int>, 2>> DGAdapt::index_all_chd(const std::vector<int> & lev_n, const std::vector<int> & sup_j)
{
	std::set<std::array<std::vector<int>, 2>> index_chd_elem;

	for (int d = 0; d < DIM; ++d)
	{
		std::vector<int> n = lev_n;
		n[d]++;
		if (lev_n[d] == 0)
		{
			std::vector<int> j = sup_j;

			std::array<std::vector<int>, 2> index{ n, j };
			index_chd_elem.insert(index);
		}
		else if (lev_n[d] == DGSolution::NMAX) // if mesh level exceed the maximum value, then do not add it to childen
		{
			continue;
		}
		else
		{
			for (int l = 0; l < 2; ++l)
			{
				std::vector<int> j = sup_j;
				j[d] = 2 * j[d] + 2 * l - 1;

				std::array<std::vector<int>, 2> index{ n, j };
				index_chd_elem.insert(index);
			}
		}
	}
	return index_chd_elem;
}

int DGAdapt::num_all_par(const std::vector<int> & lev_n) const
{
	int num = 0;
	for (int d = 0; d < DIM; ++d)
	{
		// 0 parent element for mesh level 0
		// 1 parent element for mesh level >=1
		if (lev_n[d] != 0) num++;
	}
	return num;
}


int DGAdapt::num_all_chd(const std::vector<int> & lev_n) const
{
	int num = 0;
	for (int d = 0; d < DIM; ++d)
	{
		// 1 children element for mesh level 0
		// 2 children element for mesh level >=1 and <NMAX
		// 0 children element for mesh level =NMAX
		if (lev_n[d] == 0)
		{
			num++;
		}
		else if (lev_n[d] == DGSolution::NMAX)
		{
			continue;
		}
		else
		{
			num += 2;
		}
	}
	return num;
}

void DGAdapt::check_hole()
{
	bool flag = false;

	for (auto & iter : dg)
	{
		if (iter.second.hash_ptr_par.size() < iter.second.num_total_par)
		{
			flag = true;

			// loop over all its parents
			const std::set<std::array<std::vector<int>, 2>> index_par_elem = index_all_par(iter.second.level, iter.second.suppt);
			for (auto const & index : index_par_elem)
			{
				int hash_key = hash.hash_key(index);

				// if index is not in current parent index set, then add it to dg solution
				if (iter.second.hash_ptr_par.find(hash_key) == iter.second.hash_ptr_par.end())
				{
					Element elem(index[0], index[1], all_bas, hash);
					add_elem(elem);
				}
			}
		}
	}

	if (flag)
	{
		check_hole();
	}
	else
	{
		return;
	}
}

void DGAdapt::check_hole_init_separable_scalar(const VecMultiD<double> & coeff)
{
	bool flag = false;

	for (auto & iter : dg)
	{
		if (iter.second.hash_ptr_par.size() < iter.second.num_total_par)
		{
			flag = true;

			// loop over all its parents
			const std::set<std::array<std::vector<int>, 2>> index_par_elem = index_all_par(iter.second.level, iter.second.suppt);
			for (auto const & index : index_par_elem)
			{
				int hash_key = hash.hash_key(index);

				// if index is not in current parent index set, then add it to dg solution
				if (iter.second.hash_ptr_par.find(hash_key) == iter.second.hash_ptr_par.end())
				{
					Element elem(index[0], index[1], all_bas, hash);
					init_elem_separable(elem, coeff);
					add_elem(elem);
				}
			}
		}
	}

	if (flag)
	{
		check_hole_init_separable_scalar(coeff);
	}
	else
	{
		return;
	}
}

void DGAdapt::check_hole_init_separable_system(const std::vector<VecMultiD<double>> & coeff)
{
	bool flag = false;

	for (auto & iter : dg)
	{
		if (iter.second.hash_ptr_par.size() < iter.second.num_total_par)
		{
			flag = true;

			// loop over all its parents
			const std::set<std::array<std::vector<int>, 2>> index_par_elem = index_all_par(iter.second.level, iter.second.suppt);
			for (auto const & index : index_par_elem)
			{
				int hash_key = hash.hash_key(index);

				// if index is not in current parent index set, then add it to dg solution
				if (iter.second.hash_ptr_par.find(hash_key) == iter.second.hash_ptr_par.end())
				{
					Element elem(index[0], index[1], all_bas, hash);
					for (size_t index_var = 0; index_var < VEC_NUM; index_var++)
					{
						init_elem_separable(elem, coeff[index_var]);
					}
					add_elem(elem);
				}
			}
		}
	}

	if (flag)
	{
		check_hole_init_separable_system(coeff);
	}
	else
	{
		return;
	}
}

void DGAdapt::check_hole_init_separable_scalar_sum(const std::vector<VecMultiD<double>> & coeff)
{
	const int num_func = coeff.size();

	bool flag = false;

	for (auto & iter : dg)
	{
		if (iter.second.hash_ptr_par.size() < iter.second.num_total_par)
		{
			flag = true;

			// loop over all its parents
			const std::set<std::array<std::vector<int>, 2>> index_par_elem = index_all_par(iter.second.level, iter.second.suppt);
			for (auto const & index : index_par_elem)
			{
				int hash_key = hash.hash_key(index);

				// if index is not in current parent index set, then add it to dg solution
				if (iter.second.hash_ptr_par.find(hash_key) == iter.second.hash_ptr_par.end())
				{
					Element elem(index[0], index[1], all_bas, hash);

					for (size_t i_func = 0; i_func < num_func; i_func++)
					{
						init_elem_separable(elem, coeff[i_func]);
					}
					add_elem(elem);
				}
			}
		}
	}

	if (flag)
	{
		check_hole_init_separable_scalar_sum(coeff);
	}
	else
	{
		return;
	}
}

void DGAdapt::check_hole_init_separable_system_sum(const std::vector<std::vector<VecMultiD<double>>> & coeff)
{
	bool flag = false;

	for (auto & iter : dg)
	{
		if (iter.second.hash_ptr_par.size() < iter.second.num_total_par)
		{
			flag = true;

			// loop over all its parents
			const std::set<std::array<std::vector<int>, 2>> index_par_elem = index_all_par(iter.second.level, iter.second.suppt);
			for (auto const & index : index_par_elem)
			{
				int hash_key = hash.hash_key(index);

				// if index is not in current parent index set, then add it to dg solution
				if (iter.second.hash_ptr_par.find(hash_key) == iter.second.hash_ptr_par.end())
				{
					Element elem(index[0], index[1], all_bas, hash);
					for (size_t index_var = 0; index_var < ind_var_vec.size(); index_var++)
					{
						for (size_t i_func = 0; i_func < coeff[index_var].size(); i_func++)
						{
							init_elem_separable(elem, coeff[index_var][i_func], index_var);
						}
					}
					add_elem(elem);
				}
			}
		}
	}

	if (flag)
	{
		check_hole_init_separable_system_sum(coeff);
	}
	else
	{
		return;
	}
}