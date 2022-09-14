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


void DGAdapt::compute_moment_full_grid(DGAdapt f, const std::vector<int> & moment_order, const std::vector<double> & moment_order_weight, const int num_vec)
{	
	// f should be a scalar function
	assert(f.VEC_NUM == 1);
	
	// loop over all the elements in the current dg solution (represent E or B in Maxwell equations)
	for (auto & iter : dg)
	{
		// index (including mesh level and support index)
		const std::array<std::vector<int>,2> & index = {iter.second.level, iter.second.suppt};

		// compute hash key and find it in distribution function f
		int hash_key_f = hash.hash_key(index);
		auto iter_f = f.dg.find(hash_key_f);
		
		// if (iter_f == f.dg.end())
		// // if there is no corresponding element in f, then take coefficients in E or B to be zero
		// {
		// 	for (int deg = 0; deg < AlptBasis::PMAX + 1; deg++)
		// 	{
		// 		iter.second.ucoe_alpt[0].at(deg, 0) = 0.;
		// 	}
		// }
		// else
		if (iter_f != f.dg.end())
		{
			for (int deg = 0; deg < AlptBasis::PMAX + 1; deg++)
			{
				double moment = 0.;
				for (int i = 0; i < moment_order.size(); i++)
				{
					iter.second.rhs[num_vec].at(deg, 0) += iter_f->second.ucoe_alpt[0].at(deg, moment_order[i]) * moment_order_weight[i];
				}
			}
		}
	}
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