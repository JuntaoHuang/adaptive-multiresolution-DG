#include "FastMultiplyLU.h"
#include <omp.h>

void FastRHS::transform_fucoe_to_rhs(const std::vector<const VecMultiD<double>*> & mat_1D, const std::vector<std::string> & operator_type, const int dim_interp, const double coefficient, const int vec_index)
{
    const int dim = dgsolution_ptr->DIM;

    std::vector<std::vector<int>> dim_order_transform;
    std::vector<std::vector<std::string>> LU_order_transform;
    generate_transform_order(dim, dim_order_transform, LU_order_transform);
    
    for (size_t sum_num = 0; sum_num < dim_order_transform.size(); sum_num++)
    {
        transform_multiD_partial_sum(mat_1D, dim_order_transform[sum_num], LU_order_transform[sum_num], operator_type, dim_interp, vec_index, coefficient);
    }
}

void FastRHS::set_rhs_zero(const int vec_index)
{
    for (auto & iter : dgsolution_ptr->dg)
    {
        iter.second.rhs[vec_index].set_zero();
    }
}

void FastRHS::transform_1D_from_fucoe_to_rhs(const VecMultiD<double> & mat_1D, const std::string & LU, const std::string & operator_type, 
        const std::vector<int> & size_trans_from, const std::vector<int> & size_trans_to, const int dim_transform_1D, 
        const bool is_first_step, const bool is_last_step, const int dim_interp, const int vec_index, const double coefficient)
{
    // transform from interpolation basis to alpert basis
    const int pmax_trans_from = Element::PMAX_intp;
    const int pmax_trans_to = Element::PMAX_alpt;

    resize_ucoe_transfrom(size_trans_from, vec_index);

    // if first step, then copy value in ucoe_alpt to ucoe_trans_from
    // else, copy value ucoe_trans_to to ucoe_trans_from
    if (is_first_step) { copy_fucoe_to_transfrom(dim_interp, vec_index); }
    else { copy_transto_to_transfrom(vec_index); }

    resize_ucoe_transto(size_trans_to, vec_index);

    // do transformation in 1D
    // only multiply coefficient in the first step
    if (is_first_step) { transform_1D(mat_1D, LU, operator_type, dim_transform_1D, pmax_trans_from, pmax_trans_to, coefficient, vec_index, vec_index); }
    else { transform_1D(mat_1D, LU, operator_type, dim_transform_1D, pmax_trans_from, pmax_trans_to, 1.0, vec_index, vec_index); }

    // add transform_to to rhs in the last step
    if (is_last_step) { add_transto_to_rhs(vec_index); }
}

void FastRHS::transform_multiD_partial_sum(const std::vector<const VecMultiD<double>*> & mat_1D_array, const std::vector<int> & dim_order_transform, 
        const std::vector<std::string> & LU_order_transform, const std::vector<std::string> & operator_type, const int dim_interp, const int vec_index, const double coefficient)
{
    // transform from alpert basis to interpolation basis
    const int pmax_trans_from = Element::PMAX_intp;
    const int pmax_trans_to = Element::PMAX_alpt;
    const int dim = Element::DIM;

    std::vector<std::vector<int>> series_vec = series_vec_transform_size(pmax_trans_from+1, pmax_trans_to+1, dim, dim_order_transform);

    // variable control first or last step
    std::vector<bool> is_first_step(dim, false);
    is_first_step[0] = true;
    
    std::vector<bool> is_last_step(dim, false);    
    is_last_step[dim-1] = true;
    
    for (size_t d = 0; d < dim; d++)
    {
        transform_1D_from_fucoe_to_rhs(*(mat_1D_array[dim_order_transform[d]]), LU_order_transform[d], operator_type[dim_order_transform[d]], series_vec[d], series_vec[d+1], dim_order_transform[d], is_first_step[d], is_last_step[d], dim_interp, vec_index, coefficient);
    }
}

void FastRHS::rhs_2D(const VecMultiD<double> & mat_x, const VecMultiD<double> & mat_y, const std::string operator_type_x, const std::string operator_type_y, const int dim_interp, const double coefficient)
{
    assert(dgsolution_ptr->DIM==2);

    rhs_2D_sum_1st(mat_x, mat_y, operator_type_x, operator_type_y, dim_interp, coefficient);

    rhs_2D_sum_2nd(mat_x, mat_y, operator_type_x, operator_type_y, dim_interp, coefficient);
}

void FastRHS::rhs_2D_sum_1st(const VecMultiD<double> & mat_x, const VecMultiD<double> & mat_y, const std::string operator_type_x, const std::string operator_type_y, const int dim_interp, const double coefficient)
{
    // transform from interpolation basis to alpert basis
    const int pmax_trans_from = Element::PMAX_intp;
    const int pmax_trans_to = Element::PMAX_alpt;

    // --- first do 1D transformation in x direction ---
    // resize vector before do transformation in x
    resize_ucoe_transfrom(std::vector<int>{Element::PMAX_intp+1, Element::PMAX_intp+1});

    resize_ucoe_transto(std::vector<int>{Element::PMAX_alpt+1, Element::PMAX_intp+1});

    // copy value in fucoe_intp to ucoe_trans_from
    copy_fucoe_to_transfrom(dim_interp);

    // do transformation in 1D in x direction
    transform_1D(mat_x, "L", operator_type_x, 0, pmax_trans_from, pmax_trans_to, coefficient);

    // --- second do 1D transformation in y direction ---
    // resize ucoe_trans_from
    resize_ucoe_transfrom(std::vector<int>{Element::PMAX_alpt+1, Element::PMAX_intp+1});

    // copy value ucoe_trans_to to ucoe_trans_from
    copy_transto_to_transfrom();

    resize_ucoe_transto(std::vector<int>{Element::PMAX_alpt+1, Element::PMAX_alpt+1});

    // do transformation in 1D in y direction
    transform_1D(mat_y, "full", operator_type_y, 1, pmax_trans_from, pmax_trans_to);
    
    // add value in ucoe_trans_to to rhs
    add_transto_to_rhs();
}

void FastRHS::rhs_2D_sum_2nd(const VecMultiD<double> & mat_x, const VecMultiD<double> & mat_y, const std::string operator_type_x, const std::string operator_type_y, const int dim_interp, const double coefficient)
{
    // transform from interpolation basis to alpert basis
    const int pmax_trans_from = Element::PMAX_intp;
    const int pmax_trans_to = Element::PMAX_alpt;    

    // --- first do 1D transformation in y direction ---
    // resize vector before do transformation in y
    resize_ucoe_transfrom(std::vector<int>{Element::PMAX_intp+1, Element::PMAX_intp+1});        
    
    resize_ucoe_transto(std::vector<int>{Element::PMAX_intp+1, Element::PMAX_alpt+1});

    // copy value in fucoe_intp to ucoe_trans_from
    copy_fucoe_to_transfrom(dim_interp);

    // do transformation in 1D in y direction
    transform_1D(mat_y, "full", operator_type_y, 1, pmax_trans_from, pmax_trans_to, coefficient);

    // --- second do 1D transformation in x direction ---
    // resize ucoe_trans_from
    resize_ucoe_transfrom(std::vector<int>{Element::PMAX_intp+1, Element::PMAX_alpt+1});

    // copy value ucoe_trans_to to ucoe_trans_from
    copy_transto_to_transfrom();

    resize_ucoe_transto(std::vector<int>{Element::PMAX_alpt+1, Element::PMAX_alpt+1});

    // do transformation in 1D in x direction
    transform_1D(mat_x, "U", operator_type_x, 0, pmax_trans_from, pmax_trans_to);

    // add value in ucoe_trans_to to rhs
    add_transto_to_rhs();
}

void FastRHSHamiltonJacobi::rhs_nonlinear()
{
    const int dim = dgsolution_ptr->DIM;
    
    std::vector<const VecMultiD<double>*> oper_matx_1D(dim, &(oper_matx_lagr_ptr->u_v));
    std::vector<std::string> operator_type(dim, "vol");
    
    transform_fucoe_to_rhs(oper_matx_1D, operator_type, 0, 1.0);
}

void FastMultiplyLU::transform_1D(const VecMultiD<double> & mat_1D, const std::string LU, const std::string operator_type, const int dim_transform, const int pmax_trans_from, const int pmax_trans_to, const double coefficient, const int vec_index_trans_from, const int vec_index_trans_to)
{   
    const std::vector<std::vector<int>> & index_set_trans_to = (dgsolution_ptr->dg.begin())->second.ucoe_trans_to[vec_index_trans_to].get_index_iterator();
    const std::vector<std::vector<int>> & index_set_trans_from = (dgsolution_ptr->dg.begin())->second.ucoe_trans_from[vec_index_trans_from].get_index_iterator();
    
    const int size_trans_from_dim = (dgsolution_ptr->dg.begin())->second.ucoe_trans_from[vec_index_trans_from].vec_size()[dim_transform];

    //std::vector<int> index_trans_from(dgsolution_ptr->DIM);

	// iterate over dg map
	//omp_set_num_threads(8);
#pragma omp parallel num_threads(3)	
	{
		for (auto & iter : dgsolution_ptr->dg)
		{
#pragma omp single nowait
			{
				std::vector<int> index_trans_from(dgsolution_ptr->DIM);
				// pointers to related elements
				// related elements should have the same index (level, suppt) in all dimensions except the dimension we make transformation
				// this information is stored in Element::ptr_vol_intp_1D and Element::ptr_flx_intp_1D
				const std::unordered_set<Element*> & ptr_related_elem = (operator_type == "vol") ? iter.second.ptr_vol_alpt[dim_transform] : iter.second.ptr_flx_alpt[dim_transform];
				//if (operator_type == "vol")
				//{
				//    ptr_related_elem = iter.second.ptr_vol_alpt[dim_transform];
				//}
				//else if (operator_type == "flx")
				//{
				//    ptr_related_elem = iter.second.ptr_flx_alpt[dim_transform];
				//}

				auto & tmp = iter.second.ucoe_trans_to[vec_index_trans_to];

				// clear array to be zero
				tmp.set_zero();

				// iterator over each components of ucoe_trans_to (i.e. basis function) for each element
				// here index_trans_to also denotes polynomial degree

				int index_1D_trans_to = 0;
				// 0 here denotes the first unknown and can be up to VEC_NUM-1;
				for (auto const & index_trans_to : index_set_trans_to)
				{
					// calculate order in all basis functions by using polynomial degree in 1D in the dimension which we need to do transformation
					const int order_basis_1D_trans_to = iter.second.order_elem[dim_transform] * (pmax_trans_to + 1) + index_trans_to[dim_transform];

					std::copy(index_trans_to.begin(), index_trans_to.end(), index_trans_from.begin());
					index_trans_from[dim_transform] = 0;

					const int index_1D_trans_from_zero = (*ptr_related_elem.begin())->ucoe_trans_from[vec_index_trans_from].multiD_to_1D(index_trans_from);
					const int interval = (*ptr_related_elem.begin())->ucoe_trans_from[vec_index_trans_from]._vec_accu_size[dim_transform];

					// loop over all related elements
					for (auto const & ptr_elem : ptr_related_elem)
					{
						if ((LU == "L") && (iter.second.level[dim_transform] >= ptr_elem->level[dim_transform])) { continue; }
						if ((LU == "U") && (iter.second.level[dim_transform] < ptr_elem->level[dim_transform])) { continue; }

						int index_1D_trans_from = index_1D_trans_from_zero;
						for (size_t k = 0; k < size_trans_from_dim; k++)
						{
							// transfer this polynomial degree to order in all basis functions by using polynomial degree in 1D in dimension which we need to do transformation
							const int order_basis_1D_trans_from = ptr_elem->order_elem[dim_transform] * (pmax_trans_from + 1) + k;

							tmp.at(index_1D_trans_to) += ptr_elem->ucoe_trans_from[vec_index_trans_from].at(index_1D_trans_from) * mat_1D.at(order_basis_1D_trans_from, order_basis_1D_trans_to);

							index_1D_trans_from += interval;
						}
					}
					index_1D_trans_to++;
				}
				tmp *= coefficient;
			}
		}
	}


}

std::vector<std::vector<int>> FastMultiplyLU::series_vec_transform_size(const int size_trans_from, const int size_trans_to, const int dim, const std::vector<int> & transform_order)
{
	assert(transform_order.size() == dim);

	std::vector<std::vector<int>> series_vector;

	// transformation size initially
	std::vector<int> size_vector(dim, size_trans_from);

	series_vector.push_back(size_vector);
	for (auto const & order_index : transform_order)
	{
		size_vector[order_index] = size_trans_to;
		series_vector.push_back(size_vector);
	}
	return series_vector;
}

void FastMultiplyLU::generate_transform_order(const int dim, std::vector<std::vector<int>> & dim_order_transform, std::vector<std::vector<std::string>> & LU)
{
	dim_order_transform.clear();
	LU.clear();

	if (dim == 1)
	{
		dim_order_transform.push_back(std::vector<int>{0});
		LU.push_back(std::vector<std::string>{"full"});
		return;
	}

	// generate a vector from 0 to (dim - 1)
	std::vector<int> vec_0_to_dim;
	for (size_t d = 0; d < dim; d++) { vec_0_to_dim.push_back(d); }

	const int pow2_dminus1 = pow_int(2, dim - 1);
	dim_order_transform = std::vector<std::vector<int>>(pow2_dminus1, vec_0_to_dim);

	std::vector<std::vector<int>> arr_dminus1;
	IterativeNestedLoop(arr_dminus1, dim - 1, 2);

	std::vector<std::vector<int>> arr_dim;
	for (auto const & vec_dminus1 : arr_dminus1)
	{
		std::vector<int> vec_dim;
		for (auto const & v : vec_dminus1)
		{
			vec_dim.push_back(2 * v);
		}
		vec_dim.push_back(1);
		arr_dim.push_back(vec_dim);
	}

	for (size_t num = 0; num < pow2_dminus1; num++)
	{
		sort_first_by_second_order(dim_order_transform[num], arr_dim[num]);
	}

	for (auto const & vec : arr_dim)
	{
		std::vector<std::string> LU_vec;
		for (auto const & v : vec)
		{
			if (v == 0) { LU_vec.push_back("L"); }
			else if (v == 1) { LU_vec.push_back("full"); }
			else { LU_vec.push_back("U"); }
		}
		LU.push_back(LU_vec);
	}
}

void FastMultiplyLU::resize_ucoe_transfrom(const std::vector<int> & size_trans_from, const int vec_index)
{
	for (auto & iter : dgsolution_ptr->dg)
	{
		iter.second.ucoe_trans_from[vec_index].resize(size_trans_from);
		iter.second.ucoe_trans_from[vec_index] = 0.;
	}
}

void FastMultiplyLU::resize_ucoe_transto(const std::vector<int> & size_trans_to, const int vec_index)
{
	for (auto & iter : dgsolution_ptr->dg)
	{
		iter.second.ucoe_trans_to[vec_index].resize(size_trans_to);
		iter.second.ucoe_trans_to[vec_index] = 0.;
	}
}

void FastRHS::copy_fucoe_to_transfrom(const int dim_interp, const int vec_index)
{
	for (auto & iter : dgsolution_ptr->dg)
	{
		iter.second.ucoe_trans_from[vec_index] = iter.second.fucoe_intp[vec_index][dim_interp];
	}
}

void FastMultiplyLU::copy_transto_to_transfrom(const int vec_index)
{
	for (auto & iter : dgsolution_ptr->dg)
	{
		iter.second.ucoe_trans_from[vec_index] = iter.second.ucoe_trans_to[vec_index];
	}
}

void FastRHS::add_transto_to_rhs(const int vec_index)
{
	for (auto & iter : dgsolution_ptr->dg)
	{
		iter.second.rhs[vec_index] += iter.second.ucoe_trans_to[vec_index];
	}
}

void FastInterpolation::copy_ucoealpt_to_transfrom(const int vec_index)
{
	for (auto & iter : dgsolution_ptr->dg)
	{
		iter.second.ucoe_trans_from[vec_index] = iter.second.ucoe_alpt[vec_index];
	}
}

void FastInterpolation::add_transto_to_upintp(const int vec_index)
{
	for (auto & iter : dgsolution_ptr->dg)
	{
		iter.second.up_intp[vec_index] += iter.second.ucoe_trans_to[vec_index];
	}
}

void FastInterpolation::upintp_set_zero(const int vec_index)
{
	for (auto & iter : dgsolution_ptr->dg)
	{
		iter.second.up_intp[vec_index].set_zero();
	}
}

void FastInterpolation::transform_ucoealpt_to_upintp(const std::vector<const VecMultiD<double>*> & alpt_basis_pt_1D)
{
	const int dim = dgsolution_ptr->DIM;

	// clear Element::up_intp to zero
	for (size_t vec_index = 0; vec_index < Element::VEC_NUM; vec_index++)
	{
		upintp_set_zero(vec_index);
	}

	std::vector<std::vector<int>> dim_order_transform;
	std::vector<std::vector<std::string>> LU_order_transform;
	generate_transform_order(dim, dim_order_transform, LU_order_transform);

	//	omp_set_num_threads(dim_order_transform.size());
	//#pragma omp parallel for
	for (int sum_num = 0; sum_num < dim_order_transform.size(); sum_num++)
	{
		transform_multiD_partial_sum(alpt_basis_pt_1D, dim_order_transform[sum_num], LU_order_transform[sum_num]);
	}
}

void FastInterpolation::transform_ucoealpt_to_upintp(const VecMultiD<double> & alpt_basis_pt_1D)
{
	const int dim = dgsolution_ptr->DIM;
	const std::vector<const VecMultiD<double>*> alpt_basis_pt_1D_vec(dim, &alpt_basis_pt_1D);

	transform_ucoealpt_to_upintp(alpt_basis_pt_1D_vec);
}

void FastInterpolation::transform_1D_from_ucoealpt_to_upintp(const VecMultiD<double> & mat_1D, const std::string LU, const std::vector<int> & size_trans_from, const std::vector<int> & size_trans_to, const int dim, const bool is_first_step, const bool is_last_step)
{
	// transform from alpert basis to interpolation basis
	const int pmax_trans_from = Element::PMAX_alpt;
	const int pmax_trans_to = Element::PMAX_intp;

	//omp_set_num_threads(Element::VEC_NUM);
	//omp_set_num_threads(Element::VEC_NUM);
#pragma omp parallel num_threads(Element::VEC_NUM) 
	{
		//#pragma omp single
		//		{
		//			std::cout << omp_get_num_threads() << std::endl;
		//		}
#pragma omp for
		for (int vec_index = 0; vec_index < Element::VEC_NUM; vec_index++)
		{
			/*if(Element::is_intp[vec_index][])*/
			resize_ucoe_transfrom(size_trans_from, vec_index);

			// if first step, then copy value in ucoe_alpt to ucoe_trans_from
			// else, copy value ucoe_trans_to to ucoe_trans_from
			if (is_first_step) { copy_ucoealpt_to_transfrom(vec_index); }
			else { copy_transto_to_transfrom(vec_index); }

			resize_ucoe_transto(size_trans_to, vec_index);

			// do transformation in 1D
			transform_1D(mat_1D, LU, "vol", dim, pmax_trans_from, pmax_trans_to, 1.0, vec_index, vec_index);

			if (is_last_step) { add_transto_to_upintp(vec_index); }
		}
	}
}

void FastInterpolation::transform_multiD_partial_sum(const std::vector<const VecMultiD<double>*> & mat_1D_array, const std::vector<int> & dim_order_transform, const std::vector<std::string> & LU_order_transform)
{
	// transform from alpert basis to interpolation basis
	const int pmax_trans_from = Element::PMAX_alpt;
	const int pmax_trans_to = Element::PMAX_intp;
	const int dim = Element::DIM;

	std::vector<std::vector<int>> series_vec = series_vec_transform_size(pmax_trans_from + 1, pmax_trans_to + 1, dim, dim_order_transform);

	// variable control first or last step
	std::vector<bool> is_first_step(dim, false);
	is_first_step[0] = true;

	std::vector<bool> is_last_step(dim, false);
	is_last_step[dim - 1] = true;

	for (size_t d = 0; d < dim; d++)
	{
		transform_1D_from_ucoealpt_to_upintp(*(mat_1D_array[dim_order_transform[d]]), LU_order_transform[d], series_vec[d], series_vec[d + 1], dim_order_transform[d], is_first_step[d], is_last_step[d]);
	}
}

void HyperbolicSameFluxHermRHS::rhs_vol_scalar()
{
    const int dim = dgsolution_ptr->DIM;
    
    std::vector<const VecMultiD<double>*> oper_matx_1D;
    std::vector<std::string> operator_type(dim, "vol");
    for (size_t d = 0; d < dim; d++)
    {
        oper_matx_1D.clear();
        for (size_t dim_derivative = 0; dim_derivative < dim; dim_derivative++)
        {
            if (dim_derivative == d) { oper_matx_1D.push_back(&(oper_matx_herm_ptr->u_vx)); }
            else { oper_matx_1D.push_back(&(oper_matx_herm_ptr->u_v)); }
        }
        transform_fucoe_to_rhs(oper_matx_1D, operator_type, 0);
    }
}

void HyperbolicSameFluxHermRHS::rhs_vol_scalar(const int dim)
{
	std::vector<std::string> operator_type(dgsolution_ptr->DIM, "vol");

	std::vector<const VecMultiD<double>*> oper_matx_1D;
	for (size_t dim_derivative = 0; dim_derivative < dgsolution_ptr->DIM; dim_derivative++)
	{
		if (dim_derivative == dim) { oper_matx_1D.push_back(&(oper_matx_herm_ptr->u_vx)); }
		else { oper_matx_1D.push_back(&(oper_matx_herm_ptr->u_v)); }
	}
	transform_fucoe_to_rhs(oper_matx_1D, operator_type, 0);
}

void HyperbolicSameFluxHermRHS::rhs_flx_intp_scalar()
{
    const int dim = dgsolution_ptr->DIM;
    
    const VecMultiD<double> oper_matx_uave_vjp = oper_matx_herm_ptr->ulft_vjp + oper_matx_herm_ptr->urgt_vjp;

    std::vector<const VecMultiD<double>*> oper_matx_1D;
    std::vector<std::string> operator_type;
    for (size_t d = 0; d < dim; d++)
    {
        oper_matx_1D.clear();
        operator_type.clear();

        for (size_t dim_derivative = 0; dim_derivative < dim; dim_derivative++)
        {
            if (dim_derivative == d) 
            { 
                oper_matx_1D.push_back(&oper_matx_uave_vjp);
                operator_type.push_back("flx");
            }
            else 
            { 
                oper_matx_1D.push_back(&oper_matx_herm_ptr->u_v);
                operator_type.push_back("vol");
            }
        }
        transform_fucoe_to_rhs(oper_matx_1D, operator_type, 0, 0.5);
    }
}

void HyperbolicSameFluxHermRHS::rhs_flx_intp_scalar(const int dim)
{    
    const VecMultiD<double> oper_matx_uave_vjp = oper_matx_herm_ptr->ulft_vjp + oper_matx_herm_ptr->urgt_vjp;

    std::vector<const VecMultiD<double>*> oper_matx_1D;
    std::vector<std::string> operator_type;
	for (size_t dim_derivative = 0; dim_derivative < dgsolution_ptr->DIM; dim_derivative++)
	{
		if (dim_derivative == dim)
		{ 
			oper_matx_1D.push_back(&oper_matx_uave_vjp);
			operator_type.push_back("flx");
		}
		else 
		{ 
			oper_matx_1D.push_back(&oper_matx_herm_ptr->u_v);
			operator_type.push_back("vol");
		}
	}
	transform_fucoe_to_rhs(oper_matx_1D, operator_type, 0, 0.5);
}

void HyperbolicDiffFluxHermRHS::rhs_vol_scalar()
{
	// volume integral in x direction
	rhs_2D(oper_matx_herm_ptr->u_vx, oper_matx_herm_ptr->u_v, "vol", "vol", 0);

	// volume integral in y direction
	rhs_2D(oper_matx_herm_ptr->u_v, oper_matx_herm_ptr->u_vx, "vol", "vol", 1);
}

void HyperbolicDiffFluxHermRHS::rhs_flx_intp_scalar()
{
	// flux integral in x direction
	rhs_2D(oper_matx_herm_ptr->ulft_vjp + oper_matx_herm_ptr->urgt_vjp, oper_matx_herm_ptr->u_v, "flx", "vol", 0, 0.5);

	// flux integral in y direction
	rhs_2D(oper_matx_herm_ptr->u_v, oper_matx_herm_ptr->ulft_vjp + oper_matx_herm_ptr->urgt_vjp, "vol", "flx", 1, 0.5);
}

void HyperbolicSameFluxLagrRHS::rhs_vol_scalar()
{
	// volume integral in x direction
	rhs_2D(oper_matx_lagr_ptr->u_vx, oper_matx_lagr_ptr->u_v, "vol", "vol", 0);

	// volume integral in y direction
	rhs_2D(oper_matx_lagr_ptr->u_v, oper_matx_lagr_ptr->u_vx, "vol", "vol", 0);
}

void HyperbolicSameFluxLagrRHS::rhs_flx_intp_scalar()
{
	// flux integral in x direction
	rhs_2D(oper_matx_lagr_ptr->ulft_vjp + oper_matx_lagr_ptr->urgt_vjp, oper_matx_lagr_ptr->u_v, "flx", "vol", 0, 0.5);

	// flux integral in y direction
	rhs_2D(oper_matx_lagr_ptr->u_v, oper_matx_lagr_ptr->ulft_vjp + oper_matx_lagr_ptr->urgt_vjp, "vol", "flx", 0, 0.5);
}

void HyperbolicDiffFluxLagrRHS::rhs_vol_scalar()
{
	// volume integral in x direction
	rhs_2D(oper_matx_lagr_ptr->u_vx, oper_matx_lagr_ptr->u_v, "vol", "vol", 0);

	// volume integral in y direction
	rhs_2D(oper_matx_lagr_ptr->u_v, oper_matx_lagr_ptr->u_vx, "vol", "vol", 1);
}

void HyperbolicDiffFluxLagrRHS::rhs_flx_intp_scalar()
{
	// flux integral in x direction
	rhs_2D(oper_matx_lagr_ptr->ulft_vjp + oper_matx_lagr_ptr->urgt_vjp, oper_matx_lagr_ptr->u_v, "flx", "vol", 0, 0.5);

	// flux integral in y direction
	rhs_2D(oper_matx_lagr_ptr->u_v, oper_matx_lagr_ptr->ulft_vjp + oper_matx_lagr_ptr->urgt_vjp, "vol", "flx", 1, 0.5);
}

void HyperbolicLagrRHS::rhs_vol_scalar()
{
    const int dim = dgsolution_ptr->DIM;
    
    std::vector<const VecMultiD<double>*> oper_matx_1D;
    std::vector<std::string> operator_type(dim, "vol");
    for (size_t d = 0; d < dim; d++)
    {
        oper_matx_1D.clear();
        for (size_t dim_derivative = 0; dim_derivative < dim; dim_derivative++)
        {
            if (dim_derivative == d) { oper_matx_1D.push_back(&(oper_matx_lagr_ptr->u_vx)); }
            else { oper_matx_1D.push_back(&(oper_matx_lagr_ptr->u_v)); }
        }
        transform_fucoe_to_rhs(oper_matx_1D, operator_type, d);
    }
}

void HyperbolicLagrRHS::rhs_flx_intp_scalar()
{
    const int dim = dgsolution_ptr->DIM;

    const VecMultiD<double> oper_matx_uave_vjp = oper_matx_lagr_ptr->ulft_vjp + oper_matx_lagr_ptr->urgt_vjp;

    std::vector<const VecMultiD<double>*> oper_matx_1D;
    std::vector<std::string> operator_type;
    for (size_t d = 0; d < dim; d++)
    {
        oper_matx_1D.clear();
        operator_type.clear();

        for (size_t dim_derivative = 0; dim_derivative < dim; dim_derivative++)
        {
            if (dim_derivative == d) 
            { 
                oper_matx_1D.push_back(&oper_matx_uave_vjp);
                operator_type.push_back("flx");
            }
            else 
            { 
                oper_matx_1D.push_back(&oper_matx_lagr_ptr->u_v);
                operator_type.push_back("vol");
            }
        }
        transform_fucoe_to_rhs(oper_matx_1D, operator_type, d, 0.5);
    }
}

void SourceFastLagr::rhs_source()
{
	std::vector<std::string> operator_type(dgsolution_ptr->DIM, "vol");

	std::vector<const VecMultiD<double>*> oper_matx_1D(dgsolution_ptr->DIM, &(oper_matx_lagr_ptr->u_v));

	for (size_t vec_num = 0; vec_num < dgsolution_ptr->VEC_NUM; vec_num++)
	{
		transform_fucoe_to_rhs(oper_matx_1D, operator_type, 0, 1.0, vec_num);
	}		
}

FastHermIntp::FastHermIntp(DGSolution & dgsolution, const std::vector<std::vector<double>> & Her_pt_Alpt_1D) :
	FastInterpolation(dgsolution)
{
	const int num_herm_basis = Her_pt_Alpt_1D.size();
	const int num_alpt_basis = Her_pt_Alpt_1D[0].size();

	alpt_basis_herm_pt.resize(std::vector<int>{num_alpt_basis, num_herm_basis});

	for (size_t row = 0; row < num_alpt_basis; row++)
	{
		for (size_t col = 0; col < num_herm_basis; col++)
		{
			alpt_basis_herm_pt.at(row, col) = Her_pt_Alpt_1D[col][row];
		}
	}
}

void FastHermIntp::eval_up_Herm()
{
	transform_ucoealpt_to_upintp(alpt_basis_herm_pt);
}


FastLagrIntp::FastLagrIntp(DGSolution & dgsolution, const std::vector<std::vector<double>> & Lag_pt_Alpt_1D,
	const std::vector<std::vector<double>> & Lag_pt_Alpt_1D_d1) :
	FastInterpolation(dgsolution)
{
	const int num_Lagr_basis = Lag_pt_Alpt_1D.size();
	const int num_alpt_basis = Lag_pt_Alpt_1D[0].size();

	alpt_basis_Lagr_pt.resize(std::vector<int>{num_alpt_basis, num_Lagr_basis});

	alpt_basis_der_Lagr_pt.resize(std::vector<int>{num_alpt_basis, num_Lagr_basis});

	for (size_t row = 0; row < num_alpt_basis; row++)
	{
		for (size_t col = 0; col < num_Lagr_basis; col++)
		{
			alpt_basis_Lagr_pt.at(row, col) = Lag_pt_Alpt_1D[col][row];

			alpt_basis_der_Lagr_pt.at(row, col) = Lag_pt_Alpt_1D_d1[col][row];
		}
	}

}

void FastLagrIntp::eval_up_Lagr()
{
	transform_ucoealpt_to_upintp(alpt_basis_Lagr_pt);
}

void FastLagrIntp::eval_der_up_Lagr(const int d0)
{
	const int dim = dgsolution_ptr->DIM;
	assert((d0 >= 0) && (d0 < dim));

	std::vector<const VecMultiD<double>*> oper_matx;
	for (size_t d = 0; d < dim; d++)
	{
		if (d == d0) { oper_matx.push_back(&alpt_basis_der_Lagr_pt); }
		else { oper_matx.push_back(&alpt_basis_Lagr_pt); }
	}

	transform_ucoealpt_to_upintp(oper_matx);
}



void FastInitial::copy_ucoeintp_to_transfrom(const int vec_index)
{
	for (auto & iter : dgsolution_ptr->dg)
	{
		iter.second.ucoe_trans_from[vec_index] = iter.second.ucoe_intp[vec_index];
	}
}

void FastInitial::add_transto_to_ucoealpt(const int vec_index)
{
	for (auto & iter : dgsolution_ptr->dg)
	{
		iter.second.ucoe_alpt[vec_index] += iter.second.ucoe_trans_to[vec_index];
	}
}

void FastInitial::ucoealpt_set_zero(const int vec_index)
{
	for (auto & iter : dgsolution_ptr->dg)
	{
		iter.second.ucoe_alpt[vec_index].set_zero();
	}
}

void FastInitial::transform_ucoeintp_to_ucoealpt(const std::vector<const VecMultiD<double>*> & inner_product_1D)
{
	const int dim = dgsolution_ptr->DIM;

	// clear Element::ucoe_alpt to zero
	for (size_t vec_index = 0; vec_index < Element::VEC_NUM; vec_index++)
	{
		ucoealpt_set_zero(vec_index);
	}

	std::vector<std::vector<int>> dim_order_transform;
	std::vector<std::vector<std::string>> LU_order_transform;
	generate_transform_order(dim, dim_order_transform, LU_order_transform);

	for (size_t sum_num = 0; sum_num < dim_order_transform.size(); sum_num++)
	{
		transform_multiD_partial_sum(inner_product_1D, dim_order_transform[sum_num], LU_order_transform[sum_num]);
	}
}

void FastInitial::transform_ucoeintp_to_ucoealpt(const VecMultiD<double> & inner_product_1D)
{
	const int dim = dgsolution_ptr->DIM;
	const std::vector<const VecMultiD<double>*> inner_product_1D_vec(dim, &inner_product_1D);

	transform_ucoeintp_to_ucoealpt(inner_product_1D_vec);

}

void FastInitial::transform_ucoeintp_to_ucoealpt_2D(const VecMultiD<double> & inner_product_1D)
{
	assert(dgsolution_ptr->DIM == 2);

	// transform from alpert basis to interpolation basis
	const int pmax_trans_from = Element::PMAX_intp;
	const int pmax_trans_to = Element::PMAX_alpt;

	for (size_t vec_index = 0; vec_index < Element::VEC_NUM; vec_index++)
	{
		// clear Element::ucoe_alpt to zero
		ucoealpt_set_zero(vec_index);

		// first multiply L in x direction then full matrix in y direction
		{
			// --- first do 1D transformation in x direction ---
			// resize vector before do transformation in x
			resize_ucoe_transfrom(std::vector<int>{pmax_trans_from + 1, pmax_trans_from + 1}, vec_index);

			resize_ucoe_transto(std::vector<int>{pmax_trans_to + 1, pmax_trans_from + 1}, vec_index);

			// copy value in ucoe_intp to ucoe_trans_from
			copy_ucoeintp_to_transfrom(vec_index);

			// do transformation in 1D in x direction
			transform_1D(inner_product_1D, "L", "vol", 0, pmax_trans_from, pmax_trans_to, 1.0, vec_index, vec_index);

			// --- second do 1D transformation in y direction ---
			// resize ucoe_trans_from
			resize_ucoe_transfrom(std::vector<int>{pmax_trans_to + 1, pmax_trans_from + 1}, vec_index);

			// copy value ucoe_trans_to to ucoe_trans_from
			copy_transto_to_transfrom(vec_index);

			resize_ucoe_transto(std::vector<int>{pmax_trans_to + 1, pmax_trans_to + 1}, vec_index);

			// do transformation in 1D in y direction
			transform_1D(inner_product_1D, "full", "vol", 1, pmax_trans_from, pmax_trans_to, 1.0, vec_index, vec_index);

			// add value in ucoe_trans_to to ucoe_alpt
			add_transto_to_ucoealpt(vec_index);
		}

		// then multiply full matrix in y direction then U in x direction
		{
			// --- first do 1D transformation in y direction ---
			// resize vector before do transformation in y
			resize_ucoe_transfrom(std::vector<int>{pmax_trans_from + 1, pmax_trans_from + 1}, vec_index);

			resize_ucoe_transto(std::vector<int>{pmax_trans_from + 1, pmax_trans_to + 1}, vec_index);

			// copy value in ucoe_intp to ucoe_trans_from
			copy_ucoeintp_to_transfrom(vec_index);

			// do transformation in 1D in y direction
			transform_1D(inner_product_1D, "full", "vol", 1, pmax_trans_from, pmax_trans_to, 1.0, vec_index, vec_index);

			// --- second do 1D transformation in x direction ---
			// resize ucoe_trans_from
			resize_ucoe_transfrom(std::vector<int>{pmax_trans_from + 1, pmax_trans_to + 1}, vec_index);

			// copy value ucoe_trans_to to ucoe_trans_from
			copy_transto_to_transfrom(vec_index);

			resize_ucoe_transto(std::vector<int>{pmax_trans_to + 1, pmax_trans_to + 1}, vec_index);

			// do transformation in 1D in x direction
			transform_1D(inner_product_1D, "U", "vol", 0, pmax_trans_from, pmax_trans_to, 1.0, vec_index, vec_index);

			// add value in ucoe_trans_to to ucoe_alpt
			add_transto_to_ucoealpt(vec_index);
		}

	}
}


void FastInitial::transform_1D_from_ucoeintp_to_ucoealpt(const VecMultiD<double> & mat_1D, const std::string LU, const std::vector<int> & size_trans_from, const std::vector<int> & size_trans_to, const int dim, const bool is_first_step, const bool is_last_step)
{
    // transform from interpolation basis to alpert basis
    const int pmax_trans_from = Element::PMAX_intp;
    const int pmax_trans_to = Element::PMAX_alpt;

	for (size_t vec_index = 0; vec_index < Element::VEC_NUM; vec_index++)
	{
		resize_ucoe_transfrom(size_trans_from, vec_index);

		// if first step, then copy value in ucoe_alpt to ucoe_trans_from
		// else, copy value ucoe_trans_to to ucoe_trans_from
		if (is_first_step) { copy_ucoeintp_to_transfrom(vec_index); }
		else { copy_transto_to_transfrom(vec_index); }

		resize_ucoe_transto(size_trans_to, vec_index);

		// do transformation in 1D
		transform_1D(mat_1D, LU, "vol", dim, pmax_trans_from, pmax_trans_to, 1.0, vec_index, vec_index);

		if (is_last_step) { add_transto_to_ucoealpt(vec_index); }
	}
}

void FastInitial::transform_multiD_partial_sum(const std::vector<const VecMultiD<double>*> & mat_1D_array, const std::vector<int> & dim_order_transform, const std::vector<std::string> & LU_order_transform)
{   
    // transform from interpolation basis to alpert basis
    const int pmax_trans_from = Element::PMAX_intp;
    const int pmax_trans_to = Element::PMAX_alpt;
    const int dim = Element::DIM;

	std::vector<std::vector<int>> series_vec = series_vec_transform_size(pmax_trans_from + 1, pmax_trans_to + 1, dim, dim_order_transform);

	// variable control first or last step
	std::vector<bool> is_first_step(dim, false);
	is_first_step[0] = true;

	std::vector<bool> is_last_step(dim, false);
	is_last_step[dim - 1] = true;

	for (size_t d = 0; d < dim; d++)
	{
		transform_1D_from_ucoeintp_to_ucoealpt(*(mat_1D_array[dim_order_transform[d]]), LU_order_transform[d], series_vec[d], series_vec[d + 1], dim_order_transform[d], is_first_step[d], is_last_step[d]);
	}
}



FastHermInit::FastHermInit(DGSolution & dgsolution, const OperatorMatrix1D<HermBasis, AlptBasis> & matrix) :
	FastInitial(dgsolution)
{

	std::vector<int> size = matrix.u_v.vec_size();

	Herm_basis_alpt_basis_int.resize(size);

	for (size_t row = 0; row < size[0]; row++)
	{
		for (size_t col = 0; col < size[1]; col++)
		{
			Herm_basis_alpt_basis_int.at(row, col) = matrix.u_v.at(row, col); // since: row: Herm, col: Alpt 
		}
	}
}

void FastHermInit::eval_ucoe_Alpt_Herm()
{
	transform_ucoeintp_to_ucoealpt(Herm_basis_alpt_basis_int);
}

void FastHermInit::eval_ucoe_Alpt_Herm_2D()
{
	transform_ucoeintp_to_ucoealpt_2D(Herm_basis_alpt_basis_int);
}


FastLagrInit::FastLagrInit(DGSolution & dgsolution, const OperatorMatrix1D<LagrBasis, AlptBasis> & matrix) :
	FastInitial(dgsolution)
{

	std::vector<int> size = matrix.u_v.vec_size();

	Lagr_basis_alpt_basis_int.resize(size);

	for (size_t row = 0; row < size[0]; row++)
	{
		for (size_t col = 0; col < size[1]; col++)
		{
			Lagr_basis_alpt_basis_int.at(row, col) = matrix.u_v.at(row, col); // since: row: Lag, col: Alpt 
		}
	}
}

void FastLagrInit::eval_ucoe_Alpt_Lagr()
{
	transform_ucoeintp_to_ucoealpt(Lagr_basis_alpt_basis_int);
}

void FastLagrInit::eval_ucoe_Alpt_Lagr_2D()
{
	transform_ucoeintp_to_ucoealpt_2D(Lagr_basis_alpt_basis_int);
}

FastLagrFullInit::FastLagrFullInit(DGSolution & dgsolution, const OperatorMatrix1D<LagrBasis, AlptBasis> & matrix) :
	FastLagrInit(dgsolution, matrix)
{
	// verify that it should be full grid
	bool is_full = is_full_grid();
	if (~is_full) { std::cout << "FastLagrFullInit only works for full grid" << std::endl; exit(1); }
};

void FastLagrFullInit::eval_ucoe_Alpt_Full()
{
	// transform from alpert basis to interpolation basis
	const int pmax_trans_from = Element::PMAX_intp;
	const int pmax_trans_to = Element::PMAX_alpt;
	const int DIM = dgsolution_ptr->DIM;

	// clear Element::ucoe_alpt to zero
	for (size_t vec_index = 0; vec_index < Element::VEC_NUM; vec_index++)
	{
		ucoealpt_set_zero(vec_index);

		for (size_t d = 0; d < DIM; d++)
		{
			// initialize size of ucoe_trans_form and ucoe_trans_to
			std::vector<int> size_transfrom(DIM, pmax_trans_from + 1);
			for (size_t dd = 0; dd < d; dd++) { size_transfrom[dd] = pmax_trans_to + 1; }

			std::vector<int> size_transto = size_transfrom;
			size_transto[d] = pmax_trans_to + 1;

			// resize ucoe_trans_form
			resize_ucoe_transfrom(size_transfrom, vec_index);

			if (d == 0)
			{
				// copy value in ucoe_intp to ucoe_trans_from
				copy_ucoeintp_to_transfrom(vec_index);
			}
			else
			{
				// copy value ucoe_trans_to to ucoe_trans_from
				copy_transto_to_transfrom(vec_index);
			}

			// resize ucoe_trans_to
			resize_ucoe_transto(size_transto, vec_index);

			// do transformation in 1D in d-th direction
			transform_1D(Lagr_basis_alpt_basis_int, "full", "vol", d, pmax_trans_from, pmax_trans_to, 1.0, vec_index, vec_index);
		}

		// add value in ucoe_trans_to to ucoe_alpt
		add_transto_to_ucoealpt(vec_index);

	}

}

bool FastLagrFullInit::is_full_grid()
{
	return (pow_int(2, dgsolution_ptr->DIM * dgsolution_ptr->nmax_level()) == dgsolution_ptr->size_elem());
}



void DiffusionRHS::rhs_vol()
{
	const int dim = dgsolution_ptr->DIM;

	std::vector<const VecMultiD<double>*> oper_matx_1D;
	std::vector<std::string> operator_type(dim, "vol");
	for (size_t d = 0; d < dim; d++)
	{
		oper_matx_1D.clear();
		for (size_t dim_derivative = 0; dim_derivative < dim; dim_derivative++)
		{
			if (dim_derivative == d) { oper_matx_1D.push_back(&(oper_matx_lagr_ptr->u_vx)); }
			else { oper_matx_1D.push_back(&(oper_matx_lagr_ptr->u_v)); }
		}
		transform_fucoe_to_rhs(oper_matx_1D, operator_type, d, -1.0);
	}
}

void DiffusionRHS::rhs_flx_gradu()
{
	const int dim = dgsolution_ptr->DIM;

	std::vector<const VecMultiD<double>*> oper_matx_1D;
	std::vector<std::string> operator_type;
	for (size_t d = 0; d < dim; d++)
	{
		oper_matx_1D.clear();
		operator_type.clear();

		for (size_t dim_derivative = 0; dim_derivative < dim; dim_derivative++)
		{
			if (dim_derivative == d)
			{
				oper_matx_1D.push_back(&(oper_matx_lagr_ptr->uave_vjp));
				operator_type.push_back("flx");
			}
			else
			{
				oper_matx_1D.push_back(&(oper_matx_lagr_ptr->u_v));
				operator_type.push_back("vol");
			}
		}
		transform_fucoe_to_rhs(oper_matx_1D, operator_type, d, -1.0);
	}
}

void DiffusionRHS::rhs_flx_u()
{
	const int dim = dgsolution_ptr->DIM;

	const VecMultiD<double> oper_matx_ujp_vxave = oper_matx_lagr_ptr->ujp_vxlft + oper_matx_lagr_ptr->ujp_vxrgt;

	std::vector<const VecMultiD<double>*> oper_matx_1D;
	std::vector<std::string> operator_type;
	for (size_t d = 0; d < dim; d++)
	{
		oper_matx_1D.clear();
		operator_type.clear();

		for (size_t dim_derivative = 0; dim_derivative < dim; dim_derivative++)
		{
			if (dim_derivative == d)
			{
				oper_matx_1D.push_back(&oper_matx_ujp_vxave);
				operator_type.push_back("flx");
			}
			else
			{
				oper_matx_1D.push_back(&oper_matx_lagr_ptr->u_v);
				operator_type.push_back("vol");
			}
		}
		transform_fucoe_to_rhs(oper_matx_1D, operator_type, 0, -0.5);
	}
}

void DiffusionRHS::rhs_flx_k_minus_u()
{
	const int dim = dgsolution_ptr->DIM;

	std::vector<const VecMultiD<double>*> oper_matx_1D;
	std::vector<std::string> operator_type;
	for (size_t d = 0; d < dim; d++)
	{
		oper_matx_1D.clear();
		operator_type.clear();

		for (size_t dim_derivative = 0; dim_derivative < dim; dim_derivative++)
		{
			if (dim_derivative == d)
			{
				oper_matx_1D.push_back(&(oper_matx_lagr_ptr->ujp_vxlft));
				operator_type.push_back("flx");
			}
			else
			{
				oper_matx_1D.push_back(&(oper_matx_lagr_ptr->u_v));
				operator_type.push_back("vol");
			}
		}
		transform_fucoe_to_rhs(oper_matx_1D, operator_type, 0, -0.5);
	}
}

void DiffusionRHS::rhs_flx_k_plus_u()
{
	const int dim = dgsolution_ptr->DIM;

	std::vector<const VecMultiD<double>*> oper_matx_1D;
	std::vector<std::string> operator_type;
	for (size_t d = 0; d < dim; d++)
	{
		oper_matx_1D.clear();
		operator_type.clear();

		for (size_t dim_derivative = 0; dim_derivative < dim; dim_derivative++)
		{
			if (dim_derivative == d)
			{
				oper_matx_1D.push_back(&(oper_matx_lagr_ptr->ujp_vxrgt));
				operator_type.push_back("flx");
			}
			else
			{
				oper_matx_1D.push_back(&(oper_matx_lagr_ptr->u_v));
				operator_type.push_back("vol");
			}
		}
		transform_fucoe_to_rhs(oper_matx_1D, operator_type, 0, -0.5);
	}
}

