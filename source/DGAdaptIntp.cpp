#include "DGAdaptIntp.h"


DGAdaptIntp::DGAdaptIntp(const bool sparse_, const int level_init_, const int NMAX_, AllBasis<AlptBasis> & all_bas_, AllBasis<LagrBasis> & all_bas_Lag_, AllBasis<HermBasis> & all_bas_Her_, 
                         Hash & hash_, const double eps_, const double eta_,
                         const bool is_find_ptr_alpt_, const bool is_find_ptr_intp_,
                         const OperatorMatrix1D<LagrBasis, LagrBasis> & matrix_Lag_,
                         const OperatorMatrix1D<HermBasis, HermBasis> & matrix_Her_):
DGAdapt(sparse_, level_init_, NMAX_, all_bas_, all_bas_Lag_, all_bas_Her_, hash_, eps_, eta_, is_find_ptr_alpt_, is_find_ptr_intp_), matrix_Lag_ptr(& matrix_Lag_), matrix_Her_ptr(& matrix_Her_)
{
    
}


void DGAdaptIntp::init_adaptive_intp_Lag(std::function<double(std::vector<double>, int)> func, LagrInterpolation & interp)
{
    std::cout << "size of dg before initial refine is: " << dg.size() << std::endl;

    interp.init_coe_ada_intp_Lag(func);

    //refine recursively
    refine_init_intp_Lag(func, interp);

    std::cout << "size of dg after initial refine is: " <<  dg.size() << std::endl;

    update_order_all_basis_in_dgmap();
}

void DGAdaptIntp::init_adaptive_intp_Herm(std::function<double(std::vector<double>, int, std::vector<int>)> func, HermInterpolation & interp)
{
    // std::cout << "size of dg before initial refine is: " << dg.size() << std::endl;

    interp.init_coe_ada_intp_Herm(func);

    //refine recursively
    refine_init_intp_Herm(func, interp);

    // std::cout << "size of dg after initial refine is: " <<  dg.size() << std::endl;

    update_order_all_basis_in_dgmap();
}

double DGAdaptIntp::get_L2_error_split_adaptive_intp_scalar(DGAdaptIntp & exact_solu) const
{
    double err2 = 0.0;

    // Add integral term of u^2
	
    for (auto iter : exact_solu.dg)
	{
		std::vector< VecMultiD<double> > & ucoe_alpt = (iter.second).ucoe_alpt;
		
		for (size_t it0 = 0; it0 < iter.second.size_alpt(); it0++) // size_alpt() denotes the number of all the alpt basis used 
		{
			err2 += ucoe_alpt[0].at(it0) * ucoe_alpt[0].at(it0);			
		}		
	}

    // Add integral term of u_h^2
    for (auto iter : dg)
	{
		std::vector< VecMultiD<double> > & ucoe_alpt = (iter.second).ucoe_alpt;
		
		for (size_t it0 = 0; it0 < iter.second.size_alpt(); it0++) // size_alpt() denotes the number of all the alpt basis used 
		{
			err2 += ucoe_alpt[0].at(it0) * ucoe_alpt[0].at(it0);			
		}		
	}

    // Add integral term of -2*u*u_h
    for (auto iter : dg)
	{
        const int hash_key = iter.first;

        std::vector< VecMultiD<double> > & ucoe_alpt = (iter.second).ucoe_alpt;

        // find the same elements in exact_solu.dg
        // using orthogonality
        auto it = exact_solu.dg.find(hash_key);

        if (it == exact_solu.dg.end()) continue;

        std::vector< VecMultiD<double> > & ucoe_alpt_ext = (it->second).ucoe_alpt;

        for (size_t it0 = 0; it0 < iter.second.size_alpt(); it0++) // size_alpt() denotes the number of all the alpt basis used 
		{
			err2 -= 2.0 * ucoe_alpt[0].at(it0) * ucoe_alpt_ext[0].at(it0);			
		}

    }

    ////////////////////////
    if(err2 < 0.0)
    
    {
        std::cout << "negative value in DGAdaptIntp::get_L2_error_split_adaptive_intp_scalar" << std::endl;
        std::cout << "error is: " << err2 << std::endl;
    } 

    return std::sqrt(abs(err2));
}

//------------------------------------------------------------
// 
// below are private member functions
// 
//------------------------------------------------------------

// recursively refine with Lagrange interpolation basis
void DGAdaptIntp::refine_init_intp_Lag(std::function<double(std::vector<double>, int)> func, LagrInterpolation & interp)
{
    bool flag = false;

    // set new_add = false for all old elements 
    set_all_new_add_false();

    for (auto & iter : leaf)
    {
        if (indicator_norm_intp_Lag(*(iter.second)) > eps)
        {
            flag = true;

            // loop over all its children
            const std::set<std::array<std::vector<int>,2>> index_chd_elem = index_all_chd(iter.second->level, iter.second->suppt);
            for (auto const & index : index_chd_elem)
            {
                int hash_key = hash.hash_key(index);
                
                // if index is not in current child index set, then add it to dg solution
                // here, we do not initialize the new added elements
                if ( iter.second->hash_ptr_chd.find(hash_key) == iter.second->hash_ptr_chd.end())
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

        //initialization for new added elements

        interp.init_coe_ada_intp_Lag(func);

        update_leaf();

        refine_init_intp_Lag(func, interp);
    }
    else
    {
        return;
    }
}

// evaluate L2 norm with Lagrange interpolation basis
double DGAdaptIntp::indicator_norm_intp_Lag(Element & elem) const
{	
    double norm = 0.;

    // const int sz = elem.ucoe_intp[0].size();  // size of ucoefficient for each component

    std::vector<std::vector<int>> & order_local = elem.order_local_intp;
    std::vector<std::vector<int>> & order_global = elem.order_global_intp;

    // for (size_t num_vec = 0; num_vec < VEC_NUM; num_vec++)
    for (auto num_vec : indicator_var_adapt)
    {
        auto it_global = order_global.begin();

        for (auto it = order_local.begin(); it != order_local.end(); it++, it_global++)
        {
            double val = 1.0;

            for (int d = 0; d < DIM; d++)
            {
                int num = (*it_global)[d];

                val *= matrix_Lag_ptr->u_v.at(num, num);
            }

            val *= std::pow(elem.ucoe_intp[num_vec].at(*it), 2.);
            
            norm += val;
        }
        
    }

    if(norm < 0.0) std::cout << "wrong in DGAdaptIntp::indicator_norm_intp_Lag" << std::endl;
    
    norm = std::pow(norm, 0.5);
    return norm;
}

// recursively refine with Hermite interpolation basis
void DGAdaptIntp::refine_init_intp_Herm(std::function<double(std::vector<double>, int, std::vector<int>)> func, HermInterpolation & interp)
{
    bool flag = false;

    // set new_add = false for all old elements 
    set_all_new_add_false();

    for (auto & iter : leaf)
    {
        if (indicator_norm_intp_Herm(*(iter.second)) > eps)
        {
            flag = true;

            // loop over all its children
            const std::set<std::array<std::vector<int>,2>> index_chd_elem = index_all_chd(iter.second->level, iter.second->suppt);
            for (auto const & index : index_chd_elem)
            {
                int hash_key = hash.hash_key(index);
                
                // if index is not in current child index set, then add it to dg solution
                // here, we do not initialize the new added elements
                if ( iter.second->hash_ptr_chd.find(hash_key) == iter.second->hash_ptr_chd.end())
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

        //initialization for new added elements

        interp.init_coe_ada_intp_Herm(func);

        update_leaf();

        refine_init_intp_Herm(func, interp);
    }
    else
    {
        return;
    }
}


// evaluate L2 norm with Hermite interpolation basis
double DGAdaptIntp::indicator_norm_intp_Herm(Element & elem) const
{	
    double norm = 0.;

    // const int sz = elem.ucoe_intp[0].size();  // size of ucoefficient for each component

    std::vector<std::vector<int>> & order_local = elem.order_local_intp;
    std::vector<std::vector<int>> & order_global = elem.order_global_intp;

    // for (size_t num_vec = 0; num_vec < VEC_NUM; num_vec++)
    for (auto num_vec : indicator_var_adapt)
    {
        auto it_global = order_global.begin();

        for (auto it = order_local.begin(); it != order_local.end(); it++, it_global++)
        {
            double val = 1.0;

            for (int d = 0; d < DIM; d++)
            {
                int num = (*it_global)[d];

                val *= matrix_Her_ptr->u_v.at(num, num);
            }

            val *= std::pow(elem.ucoe_intp[num_vec].at(*it), 2.);
            
            norm += val;
        }
        
    }

    if(norm < 0.0) std::cout << "wrong in DGAdaptIntp::indicator_norm_intp_Herm" << std::endl;
    
    norm = std::pow(norm, 0.5);
    return norm;
}


// // evaluate Lagrange interpolation coefficients for new added elements in refinement (for initialization) 
// void DGAdaptIntp::init_new_ele_coe_intp(std::function<double(std::vector<double>, int)> func, LagrInterpolation & interp)
// {
//     interp.init_coe_ada_intp(func);
// }