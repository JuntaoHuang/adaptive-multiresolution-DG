#include "LinearForm.h"


LinearForm::LinearForm(DGSolution & dgsolution, AllBasis<AlptBasis> & all_bas_alpt, const int gauss_quad_num_):
	dgsolution_ptr(&dgsolution), all_bas_alpt_ptr(&all_bas_alpt), gauss_quad_1D(1), gauss_quad_2D(2), gauss_quad_3D(3), gauss_quad_num(gauss_quad_num_)
{
	resize_zero_vector();
};

void LinearForm::resize_zero_vector()
{
	vec_b.resize(dgsolution_ptr->size_basis_alpt() * dgsolution_ptr->VEC_NUM);
	vec_b.setZero();
}

void LinearForm::copy_source_to_eigenvec()
{
	int order_alpt_basis_in_dgmap = 0;
	for (auto & iter : dgsolution_ptr->dg)
	{
		for (size_t num_vec = 0; num_vec < dgsolution_ptr->VEC_NUM; num_vec++)
		{
			for (size_t num_basis = 0; num_basis < iter.second.size_alpt(); num_basis++)
			{
				const std::vector<int> & order_local_basis = iter.second.order_local_alpt[num_basis];
				vec_b(order_alpt_basis_in_dgmap) = iter.second.source[num_vec].at(order_local_basis);
				order_alpt_basis_in_dgmap++;
			}
		}
	}
}

void LinearForm::copy_eigenvec_to_source()
{
	int order_alpt_basis_in_dgmap = 0;
	for (auto & iter : dgsolution_ptr->dg)
	{
		for (size_t num_vec = 0; num_vec < dgsolution_ptr->VEC_NUM; num_vec++)
		{
			for (size_t num_basis = 0; num_basis < iter.second.size_alpt(); num_basis++)
			{
				const std::vector<int> & order_local_basis = iter.second.order_local_alpt[num_basis];
				iter.second.source[num_vec].at(order_local_basis) = vec_b(order_alpt_basis_in_dgmap);
				order_alpt_basis_in_dgmap++;
			}
		}
	}
}

double LinearForm::prod_func_bas_1D(std::function<double(double)> func, const int index_alpt_basis, const int derivative_degree) const
{
	return all_bas_alpt_ptr->at(index_alpt_basis).product_function(func, derivative_degree, gauss_quad_num);
}

double LinearForm::prod_func_bas_2D_no_separable(std::function<double(std::vector<double>)> func, const std::vector<int> & index_alpt_basis, const std::vector<int> & derivative_degree) const
{
	assert(index_alpt_basis.size()==2);
	
	// basis function in 1D, in x and y directions
	const Basis & basis_x = all_bas_alpt_ptr->at(index_alpt_basis[0]);
	const Basis & basis_y = all_bas_alpt_ptr->at(index_alpt_basis[1]);

	// set intergral interval to be support of Basis
	const double xl = basis_x.supp_interv[0];
	const double xr = basis_x.supp_interv[1];
	const double xmid = (xl + xr) / 2.;

	const double yl = basis_y.supp_interv[0];
	const double yr = basis_y.supp_interv[1];
	const double ymid = (yl + yr) / 2.;

	auto f = [&](const std::vector<double> & x)->double { return basis_x.val(x[0], derivative_degree[0]) * basis_y.val(x[1], derivative_degree[1]) * func(x); };
	
	// divide [xl, xr] * [yl, yr] into four parts
	// 1. [xl, xmid] * [yl, ymid]
	// 2. [xmid, xr] * [yl, ymid]
	// 3. [xl, xmid] * [ymid, yr]
	// 4. [xmid, xr] * [ymid, yr]
	double product = gauss_quad_2D.GL_multiD(f, std::vector<double>{xl, yl}, std::vector<double>{xmid, ymid}, gauss_quad_num)
					+ gauss_quad_2D.GL_multiD(f, std::vector<double>{xmid, yl}, std::vector<double>{xr, ymid}, gauss_quad_num)
					+ gauss_quad_2D.GL_multiD(f, std::vector<double>{xl, ymid}, std::vector<double>{xmid, yr}, gauss_quad_num)
					+ gauss_quad_2D.GL_multiD(f, std::vector<double>{xmid, ymid}, std::vector<double>{xr, yr}, gauss_quad_num);

	return product;
}

double LinearForm::prod_func_bas_3D_no_separable(std::function<double(std::vector<double>)> func, const std::vector<int> & index_alpt_basis, const std::vector<int> & derivative_degree) const
{
	assert(index_alpt_basis.size()==3 && derivative_degree.size()==3);
	
	// basis function in 1D, in x, y, z directions
	const Basis & basis_x = all_bas_alpt_ptr->at(index_alpt_basis[0]);
	const Basis & basis_y = all_bas_alpt_ptr->at(index_alpt_basis[1]);
	const Basis & basis_z = all_bas_alpt_ptr->at(index_alpt_basis[2]);

	// set intergral interval to be support of Basis
	const double xl = basis_x.supp_interv[0];
	const double xr = basis_x.supp_interv[1];
	const double xmid = (xl + xr) / 2.;

	const double yl = basis_y.supp_interv[0];
	const double yr = basis_y.supp_interv[1];
	const double ymid = (yl + yr) / 2.;

	const double zl = basis_z.supp_interv[0];
	const double zr = basis_z.supp_interv[1];
	const double zmid = (zl + zr) / 2.;

	auto f = [&](const std::vector<double> & x)->double { return basis_x.val(x[0], derivative_degree[0]) * basis_y.val(x[1], derivative_degree[1]) * basis_z.val(x[2], derivative_degree[2]) * func(x); };
	
	// divide [xl, xr] * [yl, yr] * [zl, zr] into 8 parts
	// 1. [xl, xmid] * [yl, ymid] * [zl, zmid]
	// 2. [xmid, xr] * [yl, ymid] * [zl, zmid]
	// 3. [xl, xmid] * [ymid, yr] * [zl, zmid]
	// 4. [xmid, xr] * [ymid, yr] * [zl, zmid]
	// 5. [xl, xmid] * [yl, ymid] * [zmid, zr]
	// 6. [xmid, xr] * [yl, ymid] * [zmid, zr]
	// 7. [xl, xmid] * [ymid, yr] * [zmid, zr]
	// 8. [xmid, xr] * [ymid, yr] * [zmid, zr]	
	double product = gauss_quad_3D.GL_multiD(f, std::vector<double>{xl, yl, zl}, std::vector<double>{xmid, ymid, zmid}, gauss_quad_num)
					+ gauss_quad_3D.GL_multiD(f, std::vector<double>{xmid, yl, zl}, std::vector<double>{xr, ymid, zmid}, gauss_quad_num)
					+ gauss_quad_3D.GL_multiD(f, std::vector<double>{xl, ymid, zl}, std::vector<double>{xmid, yr, zmid}, gauss_quad_num)
					+ gauss_quad_3D.GL_multiD(f, std::vector<double>{xmid, ymid, zl}, std::vector<double>{xr, yr, zmid}, gauss_quad_num)
					+ gauss_quad_3D.GL_multiD(f, std::vector<double>{xl, yl, zmid}, std::vector<double>{xmid, ymid, zr}, gauss_quad_num)
					+ gauss_quad_3D.GL_multiD(f, std::vector<double>{xmid, yl, zmid}, std::vector<double>{xr, ymid, zr}, gauss_quad_num)
					+ gauss_quad_3D.GL_multiD(f, std::vector<double>{xl, ymid, zmid}, std::vector<double>{xmid, yr, zr}, gauss_quad_num)
					+ gauss_quad_3D.GL_multiD(f, std::vector<double>{xmid, ymid, zmid}, std::vector<double>{xr, yr, zr}, gauss_quad_num);

	return product;
}

void Domain1DIntegral::assemble_vector()
{
	// reformulate the function into a vector
	std::vector<std::function<double(double)>> func_x_vec;
	for (size_t num_vec = 0; num_vec < dgsolution_ptr->VEC_NUM; num_vec++)
	{
		auto f = [&](double x)->double { return func_x(x, num_vec); };
		func_x_vec.push_back(f);
	}
	
	int order_alpt_basis_in_dgmap = 0;
	for (auto const & iter : dgsolution_ptr->dg)
	{
		for (size_t num_vec = 0; num_vec < dgsolution_ptr->VEC_NUM; num_vec++)
		{
			for (size_t num_basis = 0; num_basis < iter.second.size_alpt(); num_basis++)
			{
				const std::vector<int> & order_global_basis = iter.second.order_global_alpt[num_basis];				
				vec_b(order_alpt_basis_in_dgmap) = prod_func_bas_1D(func_x_vec[num_vec], order_global_basis[0]);
				order_alpt_basis_in_dgmap++;
			}
		}
	}
}


void Domain2DIntegral::assemble_vector()
{
	resize_zero_vector();

	// reformulate the function into a vector
	std::vector<std::function<double(std::vector<double>)>> func_x_vec;
	for (size_t num_vec = 0; num_vec < dgsolution_ptr->VEC_NUM; num_vec++)
	{
		auto f = [&](std::vector<double> x)->double { return func_x(x, num_vec); };
		func_x_vec.push_back(f);
	}
	
	int order_alpt_basis_in_dgmap = 0;
	for (auto const & iter : dgsolution_ptr->dg)
	{
		for (size_t num_vec = 0; num_vec < dgsolution_ptr->VEC_NUM; num_vec++)
		{
			for (size_t num_basis = 0; num_basis < iter.second.size_alpt(); num_basis++)
			{
				const std::vector<int> & order_global_basis = iter.second.order_global_alpt[num_basis];				
				vec_b(order_alpt_basis_in_dgmap) = prod_func_bas_2D_no_separable(func_x_vec[num_vec], order_global_basis);
				order_alpt_basis_in_dgmap++;
			}
		}
	}
}

void Domain3DIntegral::assemble_vector()
{
	resize_zero_vector();

	// reformulate the function into a vector
	std::vector<std::function<double(std::vector<double>)>> func_x_vec;
	for (size_t num_vec = 0; num_vec < dgsolution_ptr->VEC_NUM; num_vec++)
	{
		auto f = [&](std::vector<double> x)->double { return func_x(x, num_vec); };
		func_x_vec.push_back(f);
	}
	
	int order_alpt_basis_in_dgmap = 0;
	for (auto const & iter : dgsolution_ptr->dg)
	{
		for (size_t num_vec = 0; num_vec < dgsolution_ptr->VEC_NUM; num_vec++)
		{
			for (size_t num_basis = 0; num_basis < iter.second.size_alpt(); num_basis++)
			{
				const std::vector<int> & order_global_basis = iter.second.order_global_alpt[num_basis];				
				vec_b(order_alpt_basis_in_dgmap) = prod_func_bas_3D_no_separable(func_x_vec[num_vec], order_global_basis);
				order_alpt_basis_in_dgmap++;
			}
		}
	}
}

DomainIntegral::DomainIntegral(DGSolution & dgsolution, AllBasis<AlptBasis> & all_bas_alpt, std::vector<std::vector<std::function<double(double, int)>>> func, const int gauss_quad_num):
	LinearForm(dgsolution, all_bas_alpt, gauss_quad_num)
{	
	const int vec_num = dgsolution_ptr->VEC_NUM;
	assert(func.size() == vec_num);
	
	for (size_t i = 0; i < vec_num; i++)
	{
		std::vector<VecMultiD<double>> coeff_i;
		for (auto & func_separable : func[i])
		{
			coeff_i.push_back(dgsolution_ptr->seperable_project(func_separable));
		}
		coeff.push_back(coeff_i);
	}
}

DomainIntegral::DomainIntegral(DGSolution & dgsolution, AllBasis<AlptBasis> & all_bas_alpt, std::function<double(double x, int dim, int num_separable, int var_index)> func, std::vector<int> num_separable, const int gauss_quad_num):
	LinearForm(dgsolution, all_bas_alpt, gauss_quad_num)
{
	const int vec_num = dgsolution_ptr->VEC_NUM;

	for (size_t i = 0; i < vec_num; i++)
	{
		std::vector<VecMultiD<double>> coeff_i;
		for (size_t num_sepa = 0; num_sepa < num_separable[i]; num_sepa++)
		{
			auto separable_function = [&](double x, int d)->double { return func(x, d, num_sepa, i); };
			coeff_i.push_back(dgsolution_ptr->seperable_project(separable_function));
		}
		coeff.push_back(coeff_i);
	}
}

void DomainIntegral::assemble_vector()
{
	const int vec_num = dgsolution_ptr->VEC_NUM;

	// loop over each element
	for (auto & iter : dgsolution_ptr->dg)
	{
		// loop over each unknown variable
		for (size_t var_index = 0; var_index < vec_num; var_index++)
		{	
			// loop over each separable function
			const int num_separable_func = coeff[var_index].size();
			
			for (size_t i = 0; i < num_separable_func; i++)
			{
				dgsolution_ptr->source_elem_separable(iter.second, coeff[var_index][i], var_index);
			}			
		}
	}

	copy_source_to_eigenvec();
}

DomainDeltaSource::DomainDeltaSource(DGSolution & dgsolution, AllBasis<AlptBasis> & all_bas_alpt, const std::vector<double> & x0):
        LinearForm(dgsolution, all_bas_alpt), point_x0(x0) 
{
	value_basis_1D_point_x0.clear();

	for (size_t d = 0; d < dgsolution_ptr->DIM; d++)
	{	
		std::vector<double> value_basis_1D = evaluate_basis_1D(point_x0[d]);
		value_basis_1D_point_x0.push_back(value_basis_1D);
	}	
}

void DomainDeltaSource::assemble_vector(const double coefficient)
{	
	resize_zero_vector();

	int order_alpt_basis_in_dgmap = 0;
	for (auto const & iter : dgsolution_ptr->dg)
	{
		for (size_t num_vec = 0; num_vec < dgsolution_ptr->VEC_NUM; num_vec++)
		{
			for (size_t num_basis = 0; num_basis < iter.second.size_alpt(); num_basis++)
			{
				const std::vector<int> & order_global_basis = iter.second.order_global_alpt[num_basis];
				double value_basis_multidim = 1.;
				for (size_t d = 0; d < dgsolution_ptr->DIM; d++)
				{
					value_basis_multidim *= value_basis_1D_point_x0[d][order_global_basis[d]];
				}				
				vec_b(order_alpt_basis_in_dgmap) = value_basis_multidim;
				order_alpt_basis_in_dgmap++;
			}
		}
	}	
	vec_b *= coefficient;
}

std::vector<double> DomainDeltaSource::evaluate_basis_1D(const double x0)
{	
	const int num_basis_1D = all_bas_alpt_ptr->size();
	std::vector<double> value_basis_1D(num_basis_1D, 0.);

	for (size_t i = 0; i < num_basis_1D; i++)
	{
		value_basis_1D[i] = all_bas_alpt_ptr->at(i).val(x0, 0, 0);
	}	
	return value_basis_1D;
}

double DomainDeltaSource::val_basis_multiD(const std::vector<double> & x0, const std::vector<int> & order_global_basis) const
{	
	const int DIM = dgsolution_ptr->DIM;

	double val = 1.;
	for (int d = 0; d < DIM; d++)
	{
		val *= all_bas_alpt_ptr->at(order_global_basis[d]).val(x0[d], 0, 0);
	}	
	return val;
}

void Boundary1DIntegral::assemble_vector_boundary1D(std::function<double(double)> func_x, const int boundary_dim, const int boundary_sign, const std::vector<int> & derivative)
{
	// make sure that input parameter is in relevant range
	assert((boundary_dim==0)||(boundary_dim==1));
	assert((boundary_sign==-1)||(boundary_sign==1));
	assert(derivative.size()==dgsolution_ptr->DIM);

	// dim of 1D boundary integral
	// if boundary is in x direction, then this 1D boundary integral is in y direction
	// if boundary is in y direction, then this 1D boundary integral is in x direction
	int integral_dim = 0;
	if (boundary_dim==0) { integral_dim = 1; }

	// if left boundary, then boundary point is 0+
	// if right boundary, then boundary point is 1-
	double boundary_x = 0. + Const::ROUND_OFF;
	if (boundary_sign==1) { boundary_x = 1. - Const::ROUND_OFF; }
	
	int order_alpt_basis_in_dgmap = 0;
	for (auto const & iter : dgsolution_ptr->dg)
	{
		for (size_t num_vec = 0; num_vec < dgsolution_ptr->VEC_NUM; num_vec++)
		{
			for (size_t num_basis = 0; num_basis < iter.second.size_alpt(); num_basis++)
			{
				const std::vector<int> & order_global_basis = iter.second.order_global_alpt[num_basis];
				
				double basis_boundary = all_bas_alpt_ptr->at(order_global_basis[boundary_dim]).val(boundary_x, derivative[boundary_dim], 0);

				double product = prod_func_bas_1D(func_x, order_global_basis[integral_dim], derivative[integral_dim]);			
				
				vec_b(order_alpt_basis_in_dgmap) += basis_boundary * product;
				order_alpt_basis_in_dgmap++;
			}
		}
	}	
}

void Boundary2DIntegral::assemble_vector_boundary2D(std::function<double(std::vector<double>)> func_2D, const int boundary_dim, const int boundary_sign, const std::vector<int> & derivative)
{
	// make sure that input parameter is in relevant range
	assert((boundary_dim>=0) && (boundary_dim<dgsolution_ptr->DIM));
	assert((boundary_sign==-1)||(boundary_sign==1));
	assert(derivative.size()==dgsolution_ptr->DIM);

	// dim of 1D boundary integral
	// if boundary is in x direction, then this 1D boundary integral is in y direction
	// if boundary is in y direction, then this 1D boundary integral is in x direction
	std::vector<int> integral_dim;
	if (boundary_dim==0) { integral_dim = std::vector<int>{1, 2}; }
	else if (boundary_dim==1) { integral_dim = std::vector<int>{0, 2}; }
	else if (boundary_dim==2) { integral_dim = std::vector<int>{0, 1}; }

	// if left boundary, then boundary point is 0+
	// if right boundary, then boundary point is 1-
	double boundary_x = 0. + Const::ROUND_OFF;
	if (boundary_sign==1) { boundary_x = 1. - Const::ROUND_OFF; }
	
	int order_alpt_basis_in_dgmap = 0;
	for (auto const & iter : dgsolution_ptr->dg)
	{
		for (size_t num_vec = 0; num_vec < dgsolution_ptr->VEC_NUM; num_vec++)
		{
			for (size_t num_basis = 0; num_basis < iter.second.size_alpt(); num_basis++)
			{
				const std::vector<int> & order_global_basis = iter.second.order_global_alpt[num_basis];
				
				double basis_boundary = all_bas_alpt_ptr->at(order_global_basis[boundary_dim]).val(boundary_x, derivative[boundary_dim], 0);

				std::vector<int> order_global_basis_integral{order_global_basis[integral_dim[0]], order_global_basis[integral_dim[1]]};
				std::vector<int> derivative_basis_integral{derivative[integral_dim[0]], derivative[integral_dim[1]]};

				double product = prod_func_bas_2D_no_separable(func_2D, order_global_basis_integral, derivative_basis_integral);
				
				vec_b(order_alpt_basis_in_dgmap) += basis_boundary * product;
				order_alpt_basis_in_dgmap++;
			}
		}
	}	
}