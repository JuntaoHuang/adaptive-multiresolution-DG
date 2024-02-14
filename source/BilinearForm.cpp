#include "BilinearForm.h"

std::map<std::vector<int>, double> ArtificialViscosity::visc_mass;
std::map<std::vector<int>, double> ArtificialViscosity::visc_stiff;

void BilinearForm::add(const BilinearForm & bilinearform)
{	
	// size should be the same
	assert(row == bilinearform.row);
	assert(col == bilinearform.col);
	
	mat += bilinearform.mat;
}

void BilinearFormAlpt::resize_zero_matrix(const int n)
{
	row = dgsolution_ptr->size_basis_alpt() * n;
	col = row;
	mat.resize(row,col);
	mat.setZero();
}

// a key function that assemble the matrix for alpert basis
// If the matrix becomes very dense, the memory may be not enough
void BilinearFormAlpt::assemble_matrix_alpt(const double operatorCoefficient, const int dim, const VecMultiD<double> & mat_operator, const VecMultiD<double> & mat_mass, const std::string integral_type, const int index_solu_variable, const int index_test_variable)
{
	// first build a list of triplets, and then convert it to a SparseMatrix
	typedef Eigen::Triplet<double> T;
	std::vector<T> tripletList; // may not be sufficient for high-dimensional problem
	const int estimation_of_entries = pow(row, 1.5);
	tripletList.reserve(estimation_of_entries);
	Eigen::SparseMatrix<double> matrix_local(row, col);

	// loop over all elements for test functions    
	for (auto & iter_test : dgsolution_ptr->dg)
	{
		// loop over all hash key related to this element
		const std::unordered_set<Element*> & ptr_solu_elem_set = (integral_type == "vol") ? iter_test.second.ptr_vol_alpt[dim] : iter_test.second.ptr_flx_alpt[dim];

		const std::vector<int> & order_local_test_basis = iter_test.second.order_local_alpt[0];
		for (auto const & ptr_solu_elem : ptr_solu_elem_set)
		{
			int row_i = iter_test.second.order_alpt_basis_in_dg[index_test_variable].at(order_local_test_basis);
			const std::vector<int> & order_local_solu_basis = ptr_solu_elem->order_local_alpt[0];
			// loop over each test basis function (multi dimension) in test element
			for (size_t index_test_basis = 0; index_test_basis < iter_test.second.size_alpt(); index_test_basis++)
			{
				// local and global order of test basis (multi dimension)
				//const std::vector<int> & order_local_test_basis = iter_test.second.order_local_alpt[index_test_basis];
				const std::vector<int> & order_global_test_basis = iter_test.second.order_global_alpt[index_test_basis];

				int col_j = ptr_solu_elem->order_alpt_basis_in_dg[index_solu_variable].at(order_local_solu_basis);
				// loop over each solution basis function in element
				for (size_t index_solu_basis = 0; index_solu_basis < ptr_solu_elem->size_alpt(); index_solu_basis++)
				{
					// local and global order of solution basis (multi dimension)
					//const std::vector<int> & order_local_solu_basis = ptr_solu_elem->order_local_alpt[index_solu_basis];
					const std::vector<int> & order_global_solu_basis = ptr_solu_elem->order_global_alpt[index_solu_basis];

					// row and col of operator matrix
					//const int col_j = ptr_solu_elem->order_alpt_basis_in_dg[index_solu_variable].at(order_local_solu_basis);
					//const int row_i = iter_test.second.order_alpt_basis_in_dg[index_test_variable].at(order_local_test_basis);

					double mat_ij = operatorCoefficient * mat_operator.at(order_global_solu_basis[dim], order_global_test_basis[dim]);
					for (size_t d = 0; d < dgsolution_ptr->DIM; d++)
					{
						if ((d != dim) && (order_global_solu_basis[d] != order_global_test_basis[d]))
						{
							mat_ij = 0.;
							break;
						}
					}
					if (std::abs(mat_ij)>=Const::ROUND_OFF)
					{
						tripletList.push_back(T(row_i, col_j, mat_ij));
					}
					col_j++;
				}
				row_i++;
			}
		}							
	}	

	matrix_local.setFromTriplets(tripletList.begin(), tripletList.end());

	mat += matrix_local;
}

void BilinearFormAlpt::assemble_matrix_alpt_coarse_grid(const double operatorCoefficient, const int dim, const VecMultiD<double> & mat_operator, const VecMultiD<double> & mat_mass, const std::string integral_type, const int mesh_nmax, const int index_solu_variable, const int index_test_variable)
{
	// first build a list of triplets, and then convert it to a SparseMatrix
	typedef Eigen::Triplet<double> T;
	std::vector<T> tripletList; // may not be sufficient for high-dimensional problem
	const int estimation_of_entries = pow(row, 1.5);
	tripletList.reserve(estimation_of_entries);
	Eigen::SparseMatrix<double> matrix_local(row, col);

	// loop over all elements for test functions    
	for (auto & iter_test : dgsolution_ptr->dg)
	{	
		// skip some elements
		int test_sum_level = std::accumulate(iter_test.second.level.begin(), iter_test.second.level.end(), 0);
		if (test_sum_level > mesh_nmax)	{ continue; }

		// loop over all hash key related to this element
		const std::unordered_set<Element*> & ptr_solu_elem_set = (integral_type == "vol") ? iter_test.second.ptr_vol_alpt[dim] : iter_test.second.ptr_flx_alpt[dim];

		const std::vector<int> & order_local_test_basis = iter_test.second.order_local_alpt[0];
		for (auto const & ptr_solu_elem : ptr_solu_elem_set)
		{
			// skip some elements
			int solu_sum_level = std::accumulate(ptr_solu_elem->level.begin(), ptr_solu_elem->level.end(), 0);
			if (solu_sum_level > mesh_nmax)	{ continue; }

			int row_i = iter_test.second.order_alpt_basis_in_dg[index_test_variable].at(order_local_test_basis);
			const std::vector<int> & order_local_solu_basis = ptr_solu_elem->order_local_alpt[0];
			// loop over each test basis function (multi dimension) in test element
			for (size_t index_test_basis = 0; index_test_basis < iter_test.second.size_alpt(); index_test_basis++)
			{
				// local and global order of test basis (multi dimension)
				//const std::vector<int> & order_local_test_basis = iter_test.second.order_local_alpt[index_test_basis];
				const std::vector<int> & order_global_test_basis = iter_test.second.order_global_alpt[index_test_basis];

				int col_j = ptr_solu_elem->order_alpt_basis_in_dg[index_solu_variable].at(order_local_solu_basis);
				// loop over each solution basis function in element
				for (size_t index_solu_basis = 0; index_solu_basis < ptr_solu_elem->size_alpt(); index_solu_basis++)
				{
					// local and global order of solution basis (multi dimension)
					//const std::vector<int> & order_local_solu_basis = ptr_solu_elem->order_local_alpt[index_solu_basis];
					const std::vector<int> & order_global_solu_basis = ptr_solu_elem->order_global_alpt[index_solu_basis];

					// row and col of operator matrix
					//const int col_j = ptr_solu_elem->order_alpt_basis_in_dg[index_solu_variable].at(order_local_solu_basis);
					//const int row_i = iter_test.second.order_alpt_basis_in_dg[index_test_variable].at(order_local_test_basis);

					double mat_ij = operatorCoefficient * mat_operator.at(order_global_solu_basis[dim], order_global_test_basis[dim]);
					for (size_t d = 0; d < dgsolution_ptr->DIM; d++)
					{
						if ((d != dim) && (order_global_solu_basis[d] != order_global_test_basis[d]))
						{
							mat_ij = 0.;
							break;
						}
					}
					if (std::abs(mat_ij)>=Const::ROUND_OFF)
					{
						tripletList.push_back(T(row_i, col_j, mat_ij));
					}
					col_j++;
				}
				row_i++;
			}
		}							
	}	

	matrix_local.setFromTriplets(tripletList.begin(), tripletList.end());

	mat += matrix_local;
}


void BilinearFormAlpt::assemble_matrix_alpt(const double operatorCoefficient, const int dim, const std::vector<const VecMultiD<double>*> & mat_1D_array, const std::string & integral_type, const int index_solu_variable, const int index_test_variable)
{
	assert(mat_1D_array.size() == dgsolution_ptr->DIM);

	// first build a list of triplets, and then convert it to a SparseMatrix
	typedef Eigen::Triplet<double> T;
	std::vector<T> tripletList; // may not be sufficient for high-dimensional problem
	const int estimation_of_entries = pow(row, 1.5);
	tripletList.reserve(estimation_of_entries);
	Eigen::SparseMatrix<double> matrix_local(row, col);

	// loop over all elements for test functions    
	for (auto & iter_test : dgsolution_ptr->dg)
	{
		// loop over all hash key related to this element
		const std::unordered_set<Element*> & ptr_solu_elem_set = (integral_type == "vol") ? iter_test.second.ptr_vol_intp[dim] : iter_test.second.ptr_flx_intp[dim];

		const std::vector<int> & order_local_test_basis = iter_test.second.order_local_alpt[0];
		for (auto const & ptr_solu_elem : ptr_solu_elem_set)
		{
			int row_i = iter_test.second.order_alpt_basis_in_dg[index_test_variable].at(order_local_test_basis);
			const std::vector<int> & order_local_solu_basis = ptr_solu_elem->order_local_alpt[0];
			// loop over each test basis function (multi dimension) in test element
			for (size_t index_test_basis = 0; index_test_basis < iter_test.second.size_alpt(); index_test_basis++)
			{
				// local and global order of test basis (multi dimension)
				//const std::vector<int> & order_local_test_basis = iter_test.second.order_local_alpt[index_test_basis];
				const std::vector<int> & order_global_test_basis = iter_test.second.order_global_alpt[index_test_basis];

				int col_j = ptr_solu_elem->order_alpt_basis_in_dg[index_solu_variable].at(order_local_solu_basis);
				// loop over each solution basis function in element
				for (size_t index_solu_basis = 0; index_solu_basis < ptr_solu_elem->size_alpt(); index_solu_basis++)
				{
					// local and global order of solution basis (multi dimension)
					//const std::vector<int> & order_local_solu_basis = ptr_solu_elem->order_local_alpt[index_solu_basis];
					const std::vector<int> & order_global_solu_basis = ptr_solu_elem->order_global_alpt[index_solu_basis];

					double mat_ij = operatorCoefficient;
					for (size_t d = 0; d < dgsolution_ptr->DIM; d++)
					{
						mat_ij *= mat_1D_array[d]->at(order_global_solu_basis[d], order_global_test_basis[d]);
					}
					if (std::abs(mat_ij)>=Const::ROUND_OFF)
					{
						tripletList.push_back(T(row_i, col_j, mat_ij));
					}
					col_j++;
				}
				row_i++;
			}
		}							
	}	

	matrix_local.setFromTriplets(tripletList.begin(), tripletList.end());

	mat += matrix_local;
}

void BilinearFormAlpt::assemble_matrix_alpt(const double operatorCoefficient, const std::vector<const VecMultiD<double>*> & mat_1D_array, const int index_solu_variable, const int index_test_variable)
{
	assert(mat_1D_array.size() == dgsolution_ptr->DIM);

	// first build a list of triplets, and then convert it to a SparseMatrix
	typedef Eigen::Triplet<double> T;
	std::vector<T> tripletList; // may not be sufficient for high-dimensional problem
	const int estimation_of_entries = pow(row, 1.5);
	tripletList.reserve(estimation_of_entries);
	Eigen::SparseMatrix<double> matrix_local(row, col);

	// loop over all elements for test functions    
	for (auto & iter_test : dgsolution_ptr->dg)
	{
		// loop over all hash key related to this element
		const std::unordered_set<Element*> & ptr_solu_elem_set = iter_test.second.ptr_general;

		const std::vector<int> & order_local_test_basis = iter_test.second.order_local_alpt[0];
		for (auto const & ptr_solu_elem : ptr_solu_elem_set)
		{
			int row_i = iter_test.second.order_alpt_basis_in_dg[index_test_variable].at(order_local_test_basis);
			const std::vector<int> & order_local_solu_basis = ptr_solu_elem->order_local_alpt[0];
			// loop over each test basis function (multi dimension) in test element
			for (size_t index_test_basis = 0; index_test_basis < iter_test.second.size_alpt(); index_test_basis++)
			{
				// local and global order of test basis (multi dimension)
				//const std::vector<int> & order_local_test_basis = iter_test.second.order_local_alpt[index_test_basis];
				const std::vector<int> & order_global_test_basis = iter_test.second.order_global_alpt[index_test_basis];

				int col_j = ptr_solu_elem->order_alpt_basis_in_dg[index_solu_variable].at(order_local_solu_basis);
				// loop over each solution basis function in element
				for (size_t index_solu_basis = 0; index_solu_basis < ptr_solu_elem->size_alpt(); index_solu_basis++)
				{
					// local and global order of solution basis (multi dimension)
					//const std::vector<int> & order_local_solu_basis = ptr_solu_elem->order_local_alpt[index_solu_basis];
					const std::vector<int> & order_global_solu_basis = ptr_solu_elem->order_global_alpt[index_solu_basis];

					double mat_ij = operatorCoefficient;
					for (size_t d = 0; d < dgsolution_ptr->DIM; d++)
					{
						mat_ij *= mat_1D_array[d]->at(order_global_solu_basis[d], order_global_test_basis[d]);
					}
					if (std::abs(mat_ij)>=Const::ROUND_OFF)
					{
						tripletList.push_back(T(row_i, col_j, mat_ij));
					}
					col_j++;
				}
				row_i++;
			}
		}							
	}	

	matrix_local.setFromTriplets(tripletList.begin(), tripletList.end());

	mat += matrix_local;
}

// GENERATE CORRECT RESULTS, BUT NOT OPTIMIZED
void BilinearFormAlpt::assemble_matrix_system_alpt(const std::vector<std::vector<double>> & operatorCoefficient, const int dim, const VecMultiD<double> & mat_operator, const VecMultiD<double> & mat_mass, const std::string integral_type, const double coef)
{
	// first build a list of triplets, and then convert it to a SparseMatrix
	typedef Eigen::Triplet<double> T;
	std::vector<T> tripletList; // may not be sufficient for high-dimensional problem
	const int estimation_of_entries = pow(row, 1.5);
	tripletList.reserve(estimation_of_entries);
	Eigen::SparseMatrix<double> matrix_local(row, col);

	auto OperCoef = operatorCoefficient;
	for (auto i = 0; i < dgsolution_ptr->VEC_NUM; ++i)
	{
		for (auto j = 0; j < dgsolution_ptr->VEC_NUM; ++j)
			OperCoef[i][j] *= coef;
	}


	// loop over all elements for test functions    
	for (auto & iter_test : dgsolution_ptr->dg)
	{
		// loop over all hash key related to this element
		std::unordered_set<Element*> ptr_solu_elem_set;
		if (integral_type == "vol")
		{
			ptr_solu_elem_set = iter_test.second.ptr_vol_alpt[dim];
		}
		else if (integral_type == "flx")
		{
			ptr_solu_elem_set = iter_test.second.ptr_flx_alpt[dim];
		}

		for (auto const & ptr_solu_elem : ptr_solu_elem_set)
		{
			// loop over each test basis function (multi dimension) in test element
			for (size_t index_test_basis = 0; index_test_basis < iter_test.second.size_alpt(); index_test_basis++)
			{
				// local and global order of test basis (multi dimension)
				const std::vector<int> & order_local_test_basis = iter_test.second.order_local_alpt[index_test_basis];
				const std::vector<int> & order_global_test_basis = iter_test.second.order_global_alpt[index_test_basis];

				// loop over each solution basis function in element
				for (size_t index_solu_basis = 0; index_solu_basis < ptr_solu_elem->size_alpt(); index_solu_basis++)
				{
					// local and global order of solution basis (multi dimension)
					const std::vector<int> & order_local_solu_basis = ptr_solu_elem->order_local_alpt[index_solu_basis];
					const std::vector<int> & order_global_solu_basis = ptr_solu_elem->order_global_alpt[index_solu_basis];

					// row and col of operator matrix
					for (auto index_test_variable = 0; index_test_variable < dgsolution_ptr->VEC_NUM; index_test_variable++)
					{
						const int row_i = iter_test.second.order_alpt_basis_in_dg[index_test_variable].at(order_local_test_basis);

						for (auto index_solu_variable = 0; index_solu_variable < dgsolution_ptr->VEC_NUM; index_solu_variable++)
						{
							const int col_j = ptr_solu_elem->order_alpt_basis_in_dg[index_solu_variable].at(order_local_solu_basis);
							// or .at(index_solu_basis);

							auto mat_ij =  OperCoef[index_test_variable][index_solu_variable];
							for (size_t d = 0; d < dgsolution_ptr->DIM; d++)
							{
								if (d == dim)
								{
									mat_ij *= mat_operator.at(order_global_solu_basis[d], order_global_test_basis[d]);
								}
								else
								{
									mat_ij *= mat_mass.at(order_global_solu_basis[d], order_global_test_basis[d]);
								}
							}
							if (std::abs(mat_ij) >= Const::ROUND_OFF)
							{
								tripletList.push_back(T(row_i, col_j, mat_ij));
							}
						}
					}
				}
			}
		}
	}

	matrix_local.setFromTriplets(tripletList.begin(), tripletList.end());

	mat += matrix_local;
}


// GENERATE CORRECT RESULTS, BUT NOT OPTIMIZED
void BilinearFormAlpt::assemble_matrix_system_alpt(const std::vector<double> & operatorCoefficient, const int dim, const VecMultiD<double> & mat_operator, const VecMultiD<double> & mat_mass, const std::string integral_type, const double coef)
{
	// first build a list of triplets, and then convert it to a SparseMatrix
	typedef Eigen::Triplet<double> T;
	std::vector<T> tripletList; // may not be sufficient for high-dimensional problem
	const int estimation_of_entries = pow(row, 1.5);
	tripletList.reserve(estimation_of_entries);
	Eigen::SparseMatrix<double> matrix_local(row, col);

	auto OperCoef = operatorCoefficient;
	for (auto i = 0; i < dgsolution_ptr->VEC_NUM; ++i)
	{
		OperCoef[i] *= coef;
	}


	// loop over all elements for test functions    
	for (auto & iter_test : dgsolution_ptr->dg)
	{
		// loop over all hash key related to this element
		std::unordered_set<Element*> ptr_solu_elem_set;
		if (integral_type == "vol")
		{
			ptr_solu_elem_set = iter_test.second.ptr_vol_alpt[dim];
		}
		else if (integral_type == "flx")
		{
			ptr_solu_elem_set = iter_test.second.ptr_flx_alpt[dim];
		}

		for (auto const & ptr_solu_elem : ptr_solu_elem_set)
		{
			// loop over each test basis function (multi dimension) in test element
			for (size_t index_test_basis = 0; index_test_basis < iter_test.second.size_alpt(); index_test_basis++)
			{
				// local and global order of test basis (multi dimension)
				const std::vector<int> & order_local_test_basis = iter_test.second.order_local_alpt[index_test_basis];
				const std::vector<int> & order_global_test_basis = iter_test.second.order_global_alpt[index_test_basis];

				// loop over each solution basis function in element
				for (size_t index_solu_basis = 0; index_solu_basis < ptr_solu_elem->size_alpt(); index_solu_basis++)
				{
					// local and global order of solution basis (multi dimension)
					const std::vector<int> & order_local_solu_basis = ptr_solu_elem->order_local_alpt[index_solu_basis];
					const std::vector<int> & order_global_solu_basis = ptr_solu_elem->order_global_alpt[index_solu_basis];

					// row and col of operator matrix
					for (auto index_test_variable = 0; index_test_variable < dgsolution_ptr->VEC_NUM; index_test_variable++)
					{
						auto index_solu_variable = index_test_variable;
						const int row_i = iter_test.second.order_alpt_basis_in_dg[index_test_variable].at(order_local_test_basis);

						const int col_j = ptr_solu_elem->order_alpt_basis_in_dg[index_solu_variable].at(order_local_solu_basis);

						auto mat_ij = OperCoef[index_test_variable];
						for (size_t d = 0; d < dgsolution_ptr->DIM; d++)
						{
							if (d == dim)
							{
								mat_ij *= mat_operator.at(order_global_solu_basis[d], order_global_test_basis[d]);
							}
							else
							{
								mat_ij *= mat_mass.at(order_global_solu_basis[d], order_global_test_basis[d]);
							}
						}
						if (std::abs(mat_ij) >= Const::ROUND_OFF)
						{
							tripletList.push_back(T(row_i, col_j, mat_ij));
						}

					}
				}
			}
		}
	}

	matrix_local.setFromTriplets(tripletList.begin(), tripletList.end());

	mat += matrix_local;
}

void KdvAlpt::assemble_matrix_scalar(const std::vector<double> & eqnCoefficient)
{
	assert(eqnCoefficient.size()==dgsolution_ptr->DIM);

	for (int dim = 0; dim < dgsolution_ptr->DIM; dim++)
	{				
		assemble_matrix_alpt(eqnCoefficient[dim], dim, oper_matx_alpt_ptr->u_vxxx, oper_matx_alpt_ptr->u_v, "vol");
		assemble_matrix_alpt(eqnCoefficient[dim], dim, oper_matx_alpt_ptr->uxxrgt_vjp, oper_matx_alpt_ptr->u_v, "flx");
		assemble_matrix_alpt(-eqnCoefficient[dim], dim, oper_matx_alpt_ptr->uxrgt_vxjp, oper_matx_alpt_ptr->u_v, "flx");
		assemble_matrix_alpt(eqnCoefficient[dim], dim, oper_matx_alpt_ptr->ulft_vxxjp, oper_matx_alpt_ptr->u_v, "flx");
	}
}

void KdvAlpt::assemble_matrix_scalar(const double eqnCoefficient)
{
	std::vector<double> eqn_array(dgsolution_ptr->DIM, eqnCoefficient);
	
	assemble_matrix_scalar(eqn_array);
}

void ZKAlpt::assemble_matrix_scalar(const std::vector<double> & eqnCoefficient, const int option)
{
	assert(dgsolution_ptr->DIM==2);
	assert(eqnCoefficient.size()==dgsolution_ptr->DIM);
	assert((option >= 0) && (option <= 2));
	
	// operator for u_xxx
	int dim = 0;
	assemble_matrix_alpt(eqnCoefficient[dim], dim, oper_matx_alpt_ptr->u_vxxx, oper_matx_alpt_ptr->u_v, "vol");
	assemble_matrix_alpt(eqnCoefficient[dim], dim, oper_matx_alpt_ptr->uxxrgt_vjp, oper_matx_alpt_ptr->u_v, "flx");
	assemble_matrix_alpt(-eqnCoefficient[dim], dim, oper_matx_alpt_ptr->uxrgt_vxjp, oper_matx_alpt_ptr->u_v, "flx");
	assemble_matrix_alpt(eqnCoefficient[dim], dim, oper_matx_alpt_ptr->ulft_vxxjp, oper_matx_alpt_ptr->u_v, "flx");
	
	// operator for u_xyy
	dim = 1;

	// vol integral of \iint u * v_xyy dxdy
	std::vector<const VecMultiD<double>*> oper_matx;
	oper_matx.push_back(&(oper_matx_alpt_ptr->u_vx));
	oper_matx.push_back(&(oper_matx_alpt_ptr->u_vxx));
	assemble_matrix_alpt(eqnCoefficient[dim], dim, oper_matx, "vol");

	// flx integral of \int u_xy+ * [v] dx
	oper_matx.clear();
	oper_matx.push_back(&(oper_matx_alpt_ptr->ux_v));
	oper_matx.push_back(&(oper_matx_alpt_ptr->uxrgt_vjp));
	assemble_matrix_alpt(eqnCoefficient[dim], 1, oper_matx, "flx");
	
	if (option == 1)
	{
		// flx integral of \int -[u]/dx/dx * [v] dx
		// for larger dissipation and thus optimal convergence order
		const double dx = 1./pow(2, dgsolution_ptr->max_mesh_level());
		assemble_matrix_alpt(-eqnCoefficient[dim]/dx/dx, dim, oper_matx_alpt_ptr->ujp_vjp, oper_matx_alpt_ptr->u_v, "flx");
	}

	// flx integral of - \int u_y+ * [v_y] dy
	oper_matx.clear();
	oper_matx.push_back(&(oper_matx_alpt_ptr->urgt_vjp));
	oper_matx.push_back(&(oper_matx_alpt_ptr->ux_vx));
	assemble_matrix_alpt(-eqnCoefficient[dim], 0, oper_matx, "flx");

	// flx integral of \int u- * [v_xy] dx
	oper_matx.clear();
	oper_matx.push_back(&(oper_matx_alpt_ptr->u_vx));
	oper_matx.push_back(&(oper_matx_alpt_ptr->ulft_vxjp));
	assemble_matrix_alpt(eqnCoefficient[dim], 1, oper_matx, "flx");

	if (option == 0)
	{	
		// jump term [u_y(x, y+)] * [v(x+, y)], in Liu Yong's formulation
		oper_matx.clear();
		oper_matx.push_back(&(oper_matx_alpt_ptr->ujp_vrgt));
		oper_matx.push_back(&(oper_matx_alpt_ptr->uxrgt_vjp));
		assemble_matrix_alpt(eqnCoefficient[dim], oper_matx);

		// jump term - [u(x+, y)] * [v_y(x, y+)], in Liu Yong's formulation
		oper_matx.clear();
		oper_matx.push_back(&(oper_matx_alpt_ptr->urgt_vjp));
		oper_matx.push_back(&(oper_matx_alpt_ptr->ujp_vxrgt));
		assemble_matrix_alpt(-eqnCoefficient[dim], oper_matx);
	}
	
	// // flx integral of \int -[u_xy]*dx*dx * [v_xy] dx
	// // you can choose add or ignore this term
	// // almost no difference for error and convergence order
	// oper_matx.clear();
	// oper_matx.push_back(&(oper_matx_alpt_ptr->ux_vx));
	// oper_matx.push_back(&(oper_matx_alpt_ptr->uxjp_vxjp));
	// assemble_matrix_alpt(-eqnCoefficient[dim]*dx*dx, 1, oper_matx, "flx");	
}

void SchrodingerAlpt::assemble_matrix(const double eqnCoefficient)
{
	const int index_u = 0;
	const int index_v = 1;

	// conservative flux:
	// we use alternating flux, i.e. \tilde{u_x} = u_x^+ and \hat{u} = u^-
	// alpha_1 = 1/2, alpha_2 = -1/2, beta_1 = beta_2 = 0

	// dissipative flux:
	// alpha_1 = 1/2, alpha_2 = -1/2, beta_1 = 1-i, beta_2 = 1+i
	// in eqn(4), Chen, Li and Cheng, JSC, 2019
	for (size_t d = 0; d < dgsolution_ptr->DIM; d++)
	{
		// assemble matrix for u
		// - \int v * phi_xx
		assemble_matrix_alpt(-eqnCoefficient, d, oper_matx_alpt_ptr->u_vxx, oper_matx_alpt_ptr->u_v, "vol", index_v, index_u);

		// v_x^+ * [phi]
		assemble_matrix_alpt(eqnCoefficient, d, oper_matx_alpt_ptr->uxrgt_vjp, oper_matx_alpt_ptr->u_v, "flx", index_v, index_u);

		// - v^- * [phi_x]
		assemble_matrix_alpt(-eqnCoefficient, d, oper_matx_alpt_ptr->ulft_vxjp, oper_matx_alpt_ptr->u_v, "flx", index_v, index_u);

		// assemble matrix for v
		// \int u * phi_xx
		assemble_matrix_alpt(eqnCoefficient, d, oper_matx_alpt_ptr->u_vxx, oper_matx_alpt_ptr->u_v, "vol", index_u, index_v);

		// - u_x^+ * [phi]
		assemble_matrix_alpt(-eqnCoefficient, d, oper_matx_alpt_ptr->uxrgt_vjp, oper_matx_alpt_ptr->u_v, "flx", index_u, index_v);

		// u_x^- * [phi_x]
		assemble_matrix_alpt(eqnCoefficient, d, oper_matx_alpt_ptr->ulft_vxjp, oper_matx_alpt_ptr->u_v, "flx", index_u, index_v);
	}	
}

void SchrodingerAlpt::assemble_matrix_couple(const double eqnCoefficient)
{
	assert(dgsolution_ptr->VEC_NUM <= 4);

	int num = dgsolution_ptr->VEC_NUM / 2;

	// i=0: variable 1
	// i=1: variable 2
	for (int i = 0; i < num; i++)
	{
		int index_u = 0;
		int index_v = 1;

		if (i == 1)
		{
			index_u = 2;
			index_v = 3;
		}
		

		for (size_t d = 0; d < dgsolution_ptr->DIM; d++)
		{
			// assemble matrix for u
			// - \int v * phi_xx
			assemble_matrix_alpt(-eqnCoefficient, d, oper_matx_alpt_ptr->u_vxx, oper_matx_alpt_ptr->u_v, "vol", index_v, index_u);

			// v_x^+ * [phi]
			assemble_matrix_alpt(eqnCoefficient, d, oper_matx_alpt_ptr->uxrgt_vjp, oper_matx_alpt_ptr->u_v, "flx", index_v, index_u);

			// - v^- * [phi_x]
			assemble_matrix_alpt(-eqnCoefficient, d, oper_matx_alpt_ptr->ulft_vxjp, oper_matx_alpt_ptr->u_v, "flx", index_v, index_u);

			// assemble matrix for v
			// \int u * phi_xx
			assemble_matrix_alpt(eqnCoefficient, d, oper_matx_alpt_ptr->u_vxx, oper_matx_alpt_ptr->u_v, "vol", index_u, index_v);

			// - u_x^+ * [phi]
			assemble_matrix_alpt(-eqnCoefficient, d, oper_matx_alpt_ptr->uxrgt_vjp, oper_matx_alpt_ptr->u_v, "flx", index_u, index_v);

			// u_x^- * [phi_x]
			assemble_matrix_alpt(eqnCoefficient, d, oper_matx_alpt_ptr->ulft_vxjp, oper_matx_alpt_ptr->u_v, "flx", index_u, index_v);
		}

	}	
}


void HyperbolicAlpt::assemble_matrix_schrodinger(const double eqnCoefficient)
{
	assert(dgsolution_ptr->VEC_NUM <= 4);

	for (size_t d = 0; d < dgsolution_ptr->DIM; d++)
	{
		// 0: real part of variable 1
		// 1: imaginary part of variable 1
		// 2: real part of variable 2
		// 3: imaginary part of variable 2
		if (eqnCoefficient >= 0.0)
		{
			for (int i = 0; i < dgsolution_ptr->VEC_NUM; i++) 
			{
				if (i<=1)
				{
					assemble_matrix_alpt(eqnCoefficient, d, oper_matx_alpt_ptr->u_vx, oper_matx_alpt_ptr->u_v, "vol", i, i);

					assemble_matrix_alpt(eqnCoefficient, d, oper_matx_alpt_ptr->ulft_vjp, oper_matx_alpt_ptr->u_v, "flx", i, i);
				}
				else
				{
					assemble_matrix_alpt(-eqnCoefficient, d, oper_matx_alpt_ptr->u_vx, oper_matx_alpt_ptr->u_v, "vol", i, i);

					assemble_matrix_alpt(-eqnCoefficient, d, oper_matx_alpt_ptr->urgt_vjp, oper_matx_alpt_ptr->u_v, "flx", i, i);
				}
			}
		}
		else
		{
			for (int i = 0; i < dgsolution_ptr->VEC_NUM; i++) 
			{
				if (i<=1)
				{
					assemble_matrix_alpt(eqnCoefficient, d, oper_matx_alpt_ptr->u_vx, oper_matx_alpt_ptr->u_v, "vol", i, i);

					assemble_matrix_alpt(eqnCoefficient, d, oper_matx_alpt_ptr->urgt_vjp, oper_matx_alpt_ptr->u_v, "flx", i, i);
				}
				else
				{
					assemble_matrix_alpt(-eqnCoefficient, d, oper_matx_alpt_ptr->u_vx, oper_matx_alpt_ptr->u_v, "vol", i, i);

					assemble_matrix_alpt(-eqnCoefficient, d, oper_matx_alpt_ptr->ulft_vjp, oper_matx_alpt_ptr->u_v, "flx", i, i);
				}
			}
		}
		
	}	
}

void HyperbolicAlpt::assemble_matrix(const VecMultiD<double> & volCoefficient, const VecMultiD<double> & fluxLeftCoefficient, const VecMultiD<double> & fluxRightCoefficient)
{
	for (int dim = 0; dim < dgsolution_ptr->DIM; dim++)
	{
		for (int index_solu_variable = 0; index_solu_variable < dgsolution_ptr->VEC_NUM; index_solu_variable++)
		{
			for (int index_test_variable = 0; index_test_variable < dgsolution_ptr->VEC_NUM; index_test_variable++)
			{
				const std::vector<int> index{ dim, index_solu_variable, index_test_variable };
				assemble_matrix_alpt(volCoefficient.at(index), dim, oper_matx_alpt_ptr->u_vx, oper_matx_alpt_ptr->u_v, "vol", index_solu_variable, index_test_variable);
				assemble_matrix_alpt(fluxLeftCoefficient.at(index), dim, oper_matx_alpt_ptr->ulft_vjp, oper_matx_alpt_ptr->u_v, "flx", index_solu_variable, index_test_variable);
				assemble_matrix_alpt(fluxRightCoefficient.at(index), dim, oper_matx_alpt_ptr->ulft_vjp, oper_matx_alpt_ptr->u_v, "flx", index_solu_variable, index_test_variable);
				std::cout << "check for fluxRight in function HyperbolicAlpt::assemble_matrix" << std::endl; // check the above line, ulft_vjp is right?
			}
		}
	}
}

void HyperbolicAlpt::assemble_matrix_scalar(const std::vector<double> & eqnCoefficient)
{
	assert(eqnCoefficient.size() == dgsolution_ptr->DIM);

	for (int dim = 0; dim < dgsolution_ptr->DIM; dim++)
	{
		assemble_matrix_alpt(eqnCoefficient[dim], dim, oper_matx_alpt_ptr->u_vx, oper_matx_alpt_ptr->u_v, "vol");

		// positive coefficient, then upwind flux use left limit
		if (eqnCoefficient[dim] >= 0)
		{
			assemble_matrix_alpt(eqnCoefficient[dim], dim, oper_matx_alpt_ptr->ulft_vjp, oper_matx_alpt_ptr->u_v, "flx");
		}
		// negative coefficient, then upwind flux use right limit
		else
		{
			assemble_matrix_alpt(eqnCoefficient[dim], dim, oper_matx_alpt_ptr->urgt_vjp, oper_matx_alpt_ptr->u_v, "flx");
		}
	}
}

void HyperbolicAlpt::assemble_matrix_scalar_coarse_grid(const std::vector<double> & eqnCoefficient, const int mesh_nmax)
{
	assert(eqnCoefficient.size() == dgsolution_ptr->DIM);

	for (int dim = 0; dim < dgsolution_ptr->DIM; dim++)
	{
		assemble_matrix_alpt_coarse_grid(eqnCoefficient[dim], dim, oper_matx_alpt_ptr->u_vx, oper_matx_alpt_ptr->u_v, "vol", mesh_nmax);

		// positive coefficient, then upwind flux use left limit
		if (eqnCoefficient[dim] >= 0)
		{
			assemble_matrix_alpt_coarse_grid(eqnCoefficient[dim], dim, oper_matx_alpt_ptr->ulft_vjp, oper_matx_alpt_ptr->u_v, "flx", mesh_nmax);
		}
		// negative coefficient, then upwind flux use right limit
		else
		{
			assemble_matrix_alpt_coarse_grid(eqnCoefficient[dim], dim, oper_matx_alpt_ptr->urgt_vjp, oper_matx_alpt_ptr->u_v, "flx", mesh_nmax);
		}
	}
}

void HyperbolicAlpt::assemble_matrix_vol_scalar(const int dim, const double coefficient)
{
	assemble_matrix_alpt(coefficient, dim, oper_matx_alpt_ptr->u_vx, oper_matx_alpt_ptr->u_v, "vol");
}


void HyperbolicAlpt::assemble_matrix_vol_system(const int dim, const std::vector<std::vector<double>> & coefficient)
{
	// check that the size of coefficient matrix is correct
	assert(coefficient.size() == dgsolution_ptr->VEC_NUM);
	for (int i = 0; i < dgsolution_ptr->VEC_NUM; i++) { assert(coefficient[i].size() == dgsolution_ptr->VEC_NUM); }
	
	for (int i = 0; i < dgsolution_ptr->VEC_NUM; i++)
	{
		for (int j = 0; j < dgsolution_ptr->VEC_NUM; j++)
		{
			if (std::abs(coefficient[i][j]) < 1e-12) { continue; }
			
			assemble_matrix_alpt(coefficient[i][j], dim, oper_matx_alpt_ptr->u_vx, oper_matx_alpt_ptr->u_v, "vol", j, i);
		}
	}

	// assemble_matrix_system_alpt(coefficient, dim, oper_matx_alpt_ptr->u_vx, oper_matx_alpt_ptr->u_v, "vol");
}


void HyperbolicAlpt::assemble_matrix_flx_scalar(const int dim, const int sign, const double coefficient)
{
	if (sign == -1)
	{
		assemble_matrix_alpt(coefficient, dim, oper_matx_alpt_ptr->ulft_vjp, oper_matx_alpt_ptr->u_v, "flx");
	}
	else if (sign == 1)
	{
		assemble_matrix_alpt(coefficient, dim, oper_matx_alpt_ptr->urgt_vjp, oper_matx_alpt_ptr->u_v, "flx");
	}
}

HJOutflowAlpt::HJOutflowAlpt(DGSolution & dgsolution, OperatorMatrix1D<AlptBasis, AlptBasis> & oper_matx_alpt_period, OperatorMatrix1D<AlptBasis, AlptBasis> & oper_matx_alpt_inside, const int n):
	HyperbolicAlpt(dgsolution, oper_matx_alpt_period, n), oper_matx_alpt_inside_ptr(&oper_matx_alpt_inside), mat1D_flx_lft(2, oper_matx_alpt_ptr->size), mat1D_flx_rgt(2, oper_matx_alpt_ptr->size)
{
	assert(oper_matx_alpt_ptr->boundary == "period");
	assert(oper_matx_alpt_inside_ptr->boundary == "inside");

	mat1D_flx_lft = oper_matx_alpt_ptr->ulft_vlft*(-1.) + oper_matx_alpt_inside_ptr->ulft_vrgt + oper_matx_alpt_ptr->u_v_bdrleft;
	mat1D_flx_rgt = oper_matx_alpt_inside_ptr->urgt_vlft*(-1.) + oper_matx_alpt_ptr->urgt_vrgt + (oper_matx_alpt_ptr->u_v_bdrright)*(-1.);
}

void HJOutflowAlpt::assemble_matrix_flx_scalar(const int dim, const int sign, const double coefficient)
{
	if (sign == -1)
	{
		assemble_matrix_alpt(coefficient, dim, mat1D_flx_lft, oper_matx_alpt_ptr->u_v, "flx");
	}
	else if (sign == 1)
	{
		assemble_matrix_alpt(coefficient, dim, mat1D_flx_rgt, oper_matx_alpt_ptr->u_v, "flx");
	}
}

void HyperbolicAlpt::assemble_matrix_flx_system(const int dim, const int sign, const std::vector<std::vector<double>> & coefficient, const double coef)
{
	// check that the size of coefficient matrix is correct
	assert(coefficient.size() == dgsolution_ptr->VEC_NUM);
	for (int i = 0; i < dgsolution_ptr->VEC_NUM; i++) { assert(coefficient[i].size() == dgsolution_ptr->VEC_NUM); }	
	
	for (int i = 0; i < dgsolution_ptr->VEC_NUM; i++)
	{
		for (int j = 0; j < dgsolution_ptr->VEC_NUM; j++)
		{
			if (std::abs(coefficient[i][j] * coef) < 1e-12) { continue; }
			
			if (sign == -1)
			{
				assemble_matrix_alpt(coefficient[i][j] * coef, dim, oper_matx_alpt_ptr->ulft_vjp, oper_matx_alpt_ptr->u_v, "flx", j, i);
			}
			else if (sign == 1)
			{
				assemble_matrix_alpt(coefficient[i][j] * coef, dim, oper_matx_alpt_ptr->urgt_vjp, oper_matx_alpt_ptr->u_v, "flx", j, i);
			}
		}
	}

	// if (sign == -1)
	// {
	// 	assemble_matrix_system_alpt(coefficient, dim, oper_matx_alpt_ptr->ulft_vjp, oper_matx_alpt_ptr->u_v, "flx", coef);
	// }
	// else if (sign == 1)
	// {
	// 	assemble_matrix_system_alpt(coefficient, dim, oper_matx_alpt_ptr->urgt_vjp, oper_matx_alpt_ptr->u_v, "flx", coef);
	// }
}


void HyperbolicAlpt::assemble_matrix_flx_system(const int dim, const int sign, const std::vector<double> & coefficient, const double coef)
{
	assert(coefficient.size() == dgsolution_ptr->VEC_NUM);
	for (int i = 0; i < dgsolution_ptr->VEC_NUM; i++)
	{
		if (std::abs(coefficient[i] * coef) < 1e-12) { continue; }

		if (sign == -1)
		{
			assemble_matrix_alpt(coefficient[i] * coef, dim, oper_matx_alpt_ptr->ulft_vjp, oper_matx_alpt_ptr->u_v, "flx", i, i);
		}
		else if (sign == 1)
		{
			assemble_matrix_alpt(coefficient[i] * coef, dim, oper_matx_alpt_ptr->urgt_vjp, oper_matx_alpt_ptr->u_v, "flx", i, i);
		}
	}
	// if (sign == -1)
	// {
	// 	assemble_matrix_system_alpt(coefficient, dim, oper_matx_alpt_ptr->ulft_vjp, oper_matx_alpt_ptr->u_v, "flx", coef);
	// }
	// else if (sign == 1)
	// {
	// 	assemble_matrix_system_alpt(coefficient, dim, oper_matx_alpt_ptr->urgt_vjp, oper_matx_alpt_ptr->u_v, "flx", coef);
	// }
}

void HyperbolicAlpt::assemble_matrix_flx_jump_system(const int dim, const std::vector<double> & coefficient)
{
	assert(coefficient.size() == dgsolution_ptr->VEC_NUM);

	for (int i = 0; i < dgsolution_ptr->VEC_NUM; i++)
	{
		if (std::abs(coefficient[i]) < 1e-12) { continue; }

		assemble_matrix_alpt(coefficient[i], dim, oper_matx_alpt_ptr->ujp_vjp, oper_matx_alpt_ptr->u_v, "flx", i, i);
	}
}

void DiffusionAlpt::assemble_matrix_scalar(const std::vector<double> & eqnCoefficient)
{
	assert(eqnCoefficient.size() == dgsolution_ptr->DIM);

	assemble_matrix_vol_gradu_gradv(eqnCoefficient);

	assemble_matrix_flx_gradu_vjp(eqnCoefficient);

	assemble_matrix_flx_ujp_gradv(eqnCoefficient);

	assemble_matrix_flx_ujp_vjp();
}

void DiffusionAlpt::assemble_matrix_vol_gradu_gradv(const std::vector<double> & eqnCoefficient)
{
	assert(eqnCoefficient.size() == dgsolution_ptr->DIM);

	for (int dim = 0; dim < dgsolution_ptr->DIM; dim++)
	{
		assemble_matrix_alpt(-eqnCoefficient[dim], dim, oper_matx_alpt_ptr->ux_vx, oper_matx_alpt_ptr->u_v, "vol");
	}
}

void DiffusionAlpt::assemble_matrix_flx_gradu_vjp(const std::vector<double> & eqnCoefficient)
{
	assert(eqnCoefficient.size() == dgsolution_ptr->DIM);

	for (int dim = 0; dim < dgsolution_ptr->DIM; dim++)
	{
		assemble_matrix_alpt(-eqnCoefficient[dim], dim, oper_matx_alpt_ptr->uxave_vjp, oper_matx_alpt_ptr->u_v, "flx");
	}
}

void DiffusionAlpt::assemble_matrix_flx_ujp_gradv(const std::vector<double> & eqnCoefficient)
{
	assert(eqnCoefficient.size() == dgsolution_ptr->DIM);

	for (int dim = 0; dim < dgsolution_ptr->DIM; dim++)
	{
		assemble_matrix_alpt(-eqnCoefficient[dim], dim, oper_matx_alpt_ptr->ujp_vxave, oper_matx_alpt_ptr->u_v, "flx");
	}
}

void DiffusionAlpt::assemble_matrix_flx_ujp_vjp()
{
	const int max_mesh = dgsolution_ptr->max_mesh_level();
	const double dx = 1. / pow(2., max_mesh);

	for (int dim = 0; dim < dgsolution_ptr->DIM; dim++)
	{
		assemble_matrix_alpt(-sigma_ipdg / dx, dim, oper_matx_alpt_ptr->ujp_vjp, oper_matx_alpt_ptr->u_v, "flx");
	}
}

DiffusionZeroDirichletAlpt::DiffusionZeroDirichletAlpt(DGSolution & dgsolution, OperatorMatrix1D<AlptBasis, AlptBasis> & oper_matx_alpt, const double sigma_ipdg_, const double eqnCoefficient_):
	BilinearFormAlpt(dgsolution, oper_matx_alpt), sigma_ipdg(sigma_ipdg_), eqnCoefficient(eqnCoefficient_), mat1D_flx(2, oper_matx_alpt_ptr->size)
{            
	const double dx = 1./std::pow(2., dgsolution_ptr->nmax_level());
	mat1D_flx = (oper_matx_alpt_ptr->uxave_vjp + oper_matx_alpt_ptr->ujp_vxave)*(-eqnCoefficient)
			+ oper_matx_alpt_ptr->ujp_vjp*(-sigma_ipdg/dx)
			+ (oper_matx_alpt_ptr->ux_v_bdrright - oper_matx_alpt_ptr->ux_v_bdrleft + oper_matx_alpt_ptr->u_vx_bdrright - oper_matx_alpt_ptr->u_vx_bdrleft)*(eqnCoefficient)
			+ (oper_matx_alpt_ptr->u_v_bdrleft+oper_matx_alpt_ptr->u_v_bdrright)*(-sigma_ipdg/dx);
}

void DiffusionZeroDirichletAlpt::assemble_matrix_scalar()
{
	for (int dim = 0; dim < dgsolution_ptr->DIM; dim++)
	{
		// domain integral
		assemble_matrix_alpt(-eqnCoefficient, dim, oper_matx_alpt_ptr->ux_vx, oper_matx_alpt_ptr->u_v, "vol");

		// interface integral including zero Dirichlet boundary
		assemble_matrix_alpt(1., dim, mat1D_flx, oper_matx_alpt_ptr->u_v, "flx");
	}
}
// the key function: assemble matrix for intp basis
void BilinearFormIntp::assemble_matrix_intp(const double operatorCoefficient, const int dim, const VecMultiD<double> & mat_operator, const VecMultiD<double> & mat_mass, const std::string integral_type, const int index_solu_variable, const int index_test_variable)
{
	// first build a list of triplets, and then convert it to a SparseMatrix
	typedef Eigen::Triplet<double> T;
	std::vector<T> tripletList;
	const int estimation_of_entries = pow(row, 1.5);
	tripletList.reserve(estimation_of_entries);
	Eigen::SparseMatrix<double> matrix_local(row, col);

	// loop over all elements for test functions    
	for (auto & iter_test : dgsolution_ptr->dg)
	{
		// loop over all hash key related to this element
		std::unordered_set<Element*> ptr_solu_elem_set;
		if (integral_type == "vol")
		{
			ptr_solu_elem_set = iter_test.second.ptr_vol_intp[dim];
		}
		else if (integral_type == "flx")
		{
			ptr_solu_elem_set = iter_test.second.ptr_flx_intp[dim];
		}

		for (auto const & ptr_solu_elem : ptr_solu_elem_set)
		{
			// loop over each test basis function (multi dimension) in test element
			for (size_t index_test_basis = 0; index_test_basis < iter_test.second.size_alpt(); index_test_basis++)
			{
				// local and global order of test basis (multi dimension)
				const std::vector<int> & order_local_test_basis = iter_test.second.order_local_alpt[index_test_basis];
				const std::vector<int> & order_global_test_basis = iter_test.second.order_global_alpt[index_test_basis];

				// loop over each solution basis function in element
				for (size_t index_solu_basis = 0; index_solu_basis < ptr_solu_elem->size_intp(); index_solu_basis++)
				{
					// local and global order of solution basis (multi dimension)
					const std::vector<int> & order_local_solu_basis = ptr_solu_elem->order_local_intp[index_solu_basis];
					const std::vector<int> & order_global_solu_basis = ptr_solu_elem->order_global_intp[index_solu_basis];

					// row and col of operator matrix
					const int col_j = ptr_solu_elem->order_intp_basis_in_dg[index_solu_variable].at(order_local_solu_basis);
					const int row_i = iter_test.second.order_alpt_basis_in_dg[index_test_variable].at(order_local_test_basis);

					double mat_ij = operatorCoefficient;
					for (size_t d = 0; d < dgsolution_ptr->DIM; d++)
					{
						if (d == dim)
						{
							mat_ij *= mat_operator.at(order_global_solu_basis[d], order_global_test_basis[d]);
						}
						else
						{
							mat_ij *= mat_mass.at(order_global_solu_basis[d], order_global_test_basis[d]);
						}
					}
					if (std::abs(mat_ij) >= Const::ROUND_OFF)
					{
						tripletList.push_back(T(row_i, col_j, mat_ij));
					}
				}
			}
		}
	}

	matrix_local.setFromTriplets(tripletList.begin(), tripletList.end());

	mat += matrix_local;
}


void BilinearFormIntp::assemble_matrix_intp(const double operatorCoefficient, const VecMultiD<double> & mat_mass, const int index_solu_variable, const int index_test_variable)
{
	// first build a list of triplets, and then convert it to a SparseMatrix
	typedef Eigen::Triplet<double> T;
	std::vector<T> tripletList;
	const int estimation_of_entries = pow(row, 1.5);
	tripletList.reserve(estimation_of_entries);
	Eigen::SparseMatrix<double> matrix_local(row, col);

	// loop over all elements for test functions    
	for (auto & iter_test : dgsolution_ptr->dg)
	{
		// loop over all hash key related to this element
		std::unordered_set<Element*> ptr_solu_elem_set;
			
		ptr_solu_elem_set = iter_test.second.ptr_vol_intp[0]; // ? are the ptr_vol_intp the same for all dimensions???
	
		for (auto const & ptr_solu_elem : ptr_solu_elem_set)
		{
			// loop over each test basis function (multi dimension) in test element
			for (size_t index_test_basis = 0; index_test_basis < iter_test.second.size_alpt(); index_test_basis++)
			{
				// local and global order of test basis (multi dimension)
				const std::vector<int> & order_local_test_basis = iter_test.second.order_local_alpt[index_test_basis];
				const std::vector<int> & order_global_test_basis = iter_test.second.order_global_alpt[index_test_basis];

				// loop over each solution basis function in element
				for (size_t index_solu_basis = 0; index_solu_basis < ptr_solu_elem->size_intp(); index_solu_basis++)
				{
					// local and global order of solution basis (multi dimension)
					const std::vector<int> & order_local_solu_basis = ptr_solu_elem->order_local_intp[index_solu_basis];
					const std::vector<int> & order_global_solu_basis = ptr_solu_elem->order_global_intp[index_solu_basis];

					// row and col of operator matrix
					const int col_j = ptr_solu_elem->order_intp_basis_in_dg[index_solu_variable].at(order_local_solu_basis);
					const int row_i = iter_test.second.order_alpt_basis_in_dg[index_test_variable].at(order_local_test_basis);

					double mat_ij = operatorCoefficient;
					for (size_t d = 0; d < dgsolution_ptr->DIM; d++)
					{
						mat_ij *= mat_mass.at(order_global_solu_basis[d], order_global_test_basis[d]);
					}
					if (std::abs(mat_ij) >= Const::ROUND_OFF)
					{
						tripletList.push_back(T(row_i, col_j, mat_ij));
					}
				}
			}
		}
	}

	matrix_local.setFromTriplets(tripletList.begin(), tripletList.end());

	mat += matrix_local;


	//std::cout << mat.cols() << " " << mat.rows();
}



void HyperbolicHerm::assemble_matrix_scalar(const int dim)
{
	// f(u) is interpolated by Hermite basis
	// volume intergral of f(u) * v_x
	assemble_matrix_intp(1., dim, oper_matx_herm_ptr->u_vx, oper_matx_herm_ptr->u_v, "vol");

	// flux intergral of 1/2 * f(u^-) * [v]
	assemble_matrix_intp(0.5, dim, oper_matx_herm_ptr->ulft_vjp, oper_matx_herm_ptr->u_v, "flx");

	// flux intergral of 1/2 * f(u^+) * [v]
	assemble_matrix_intp(0.5, dim, oper_matx_herm_ptr->urgt_vjp, oper_matx_herm_ptr->u_v, "flx");
}

void HyperbolicHerm::assemble_matrix_vol_scalar(const int dim, const double coefficient)
{
	assemble_matrix_intp(coefficient, dim, oper_matx_herm_ptr->u_vx, oper_matx_herm_ptr->u_v, "vol");
}

void HyperbolicHerm::assemble_matrix_flx_scalar(const int dim, const int sign, const double coefficient)
{
	if (sign == -1)
	{
		assemble_matrix_intp(coefficient, dim, oper_matx_herm_ptr->ulft_vjp, oper_matx_herm_ptr->u_v, "flx");
	}
	else if (sign == 1)
	{
		assemble_matrix_intp(coefficient, dim, oper_matx_herm_ptr->urgt_vjp, oper_matx_herm_ptr->u_v, "flx");
	}
}


void HamiltonJacobiLagr::assemble_matrix_vol_scalar(const double coefficient)
{
	assemble_matrix_intp(coefficient, oper_matx_lagr_ptr->u_v);
}



void DiffusionIntpGradU::assemble_matrix(const int dim, const std::string & interp_type)
{
	if (interp_type == "lagr")
	{
		assemble_matrix_vol_lagr(dim, -1.);
		assemble_matrix_flx_lagr(dim, -1.);
	}
	else if (interp_type == "herm")
	{
		assemble_matrix_vol_herm(dim, -1.);
		assemble_matrix_flx_herm(dim, -1.);
	}
	else
	{
		std::cout << "parameter not correct in DiffusionIntpGradU::assemble_matrix()" << std::endl; exit(1);
	}
}

void DiffusionIntpGradU::assemble_matrix_vol_lagr(const int dim, const double coefficient)
{
	assemble_matrix_intp(coefficient, dim, oper_matx_lagr_ptr->u_vx, oper_matx_lagr_ptr->u_v, "vol");
}

void DiffusionIntpGradU::assemble_matrix_vol_herm(const int dim, const double coefficient)
{
	assemble_matrix_intp(coefficient, dim, oper_matx_herm_ptr->u_vx, oper_matx_herm_ptr->u_v, "vol");
}

void DiffusionIntpGradU::assemble_matrix_flx_lagr(const int dim, const double coefficient)
{
	assemble_matrix_intp(coefficient, dim, oper_matx_lagr_ptr->uave_vjp, oper_matx_lagr_ptr->u_v, "flx");
}

void DiffusionIntpGradU::assemble_matrix_flx_herm(const int dim, const double coefficient)
{
	assemble_matrix_intp(coefficient, dim, oper_matx_herm_ptr->uave_vjp, oper_matx_herm_ptr->u_v, "flx");
}

void DiffusionIntpU::assemble_matrix(const int dim, const std::string & interp_type)
{
	if (sign == -1)
	{
		if (interp_type == "lagr")
		{
			assemble_matrix_intp(-0.5, dim, oper_matx_lagr_ptr->ujp_vxlft, oper_matx_lagr_ptr->u_v, "flx");
		}
		else
		{
			std::cout << "parameter not correct in DiffusionIntpU::assemble_matrix()" << std::endl; exit(1);
		}
	}
	else if (sign == 1)
	{
		if (interp_type == "lagr")
		{
			assemble_matrix_intp(-0.5, dim, oper_matx_lagr_ptr->ujp_vxrgt, oper_matx_lagr_ptr->u_v, "flx");
		}
		else
		{
			std::cout << "parameter not correct in DiffusionIntpU::assemble_matrix()" << std::endl; exit(1);
		}
	}
	else
	{
		std::cout << "parameter not correct in DiffusionIntpU::assemble_matrix()" << std::endl; exit(1);
	}
}

void ArtificialViscosity::assemble_matrix_one_element(Element* visc_elem, const double viscosity_coefficient, const int viscosity_option, const int index_solu_variable, const int index_test_variable)
{
	// first build a list of triplets, and then convert it to a SparseMatrix
	typedef Eigen::Triplet<double> T;
	std::vector<T> tripletList;
	const int estimation_of_entries = pow(row, 1.5);
	tripletList.reserve(estimation_of_entries);
	Eigen::SparseMatrix<double> matrix_local(row, col);

	// all related elements
	std::unordered_set<Element*> intersect_element;
	// two choice here: which one is better? this need to be tested numerically
	// Note: they are the same in 1D case
	if (viscosity_option == 1)
	{
		// choice 1: find all elements that have intersection with the given viscosity elements
		intersect_element = visc_elem->ptr_vol_intp[0];
	}
	else if (viscosity_option == 2)
	{
		// choice 2: find all parents' elements
		// Note: choice 2 is a subset of choice 1
		find_all_parents(visc_elem, intersect_element);
	}
	else
	{
		std::cout << "error in ArtificialViscosity::assemble_matrix_one_element()" << std::endl; exit(1);
	}


	// get global order of arbitrary one basis function in viscosity element
	const std::vector<int> & global_order_visc = visc_elem->order_global_alpt[0];

	// gauss quadrature with k points is exact for polynomials of degree (2k-1)
	// for mass matrix, we calculate u * v which is at most degree (2k), thus only need quadrature points of number (k+1)
	// for stiff matrix, we calculate u_x * v_x which is at most degree (2k-2), thus only need quadrature points of number k
	const int num_gauss_pt_mass = Element::PMAX_alpt + 1;
	const int num_gauss_pt_stiff = Element::PMAX_alpt;

	// loop over all elements for test functions    
	for (auto & ptr_test : intersect_element)
	{
		// loop over all elements for solution functions
		for (auto & ptr_solu : intersect_element)
		{
			// loop over each test basis function (multi dimension) in test element
			for (size_t index_test_basis = 0; index_test_basis < ptr_test->size_alpt(); index_test_basis++)
			{
				// local and global order of test basis (multi dimension)
				const std::vector<int> & order_local_test_basis = ptr_test->order_local_alpt[index_test_basis];
				const std::vector<int> & order_global_test_basis = ptr_test->order_global_alpt[index_test_basis];

				// loop over each solution basis function in element
				for (size_t index_solu_basis = 0; index_solu_basis < ptr_solu->size_alpt(); index_solu_basis++)
				{
					// local and global order of solution basis (multi dimension)
					const std::vector<int> & order_local_solu_basis = ptr_solu->order_local_alpt[index_solu_basis];
					const std::vector<int> & order_global_solu_basis = ptr_solu->order_global_alpt[index_solu_basis];

					// for these two multidimensional basis, calculate mass integral and stiff integral
					std::vector<double> mass(dgsolution_ptr->DIM, 0.);
					std::vector<double> stiff(dgsolution_ptr->DIM, 0.);
					for (size_t d = 0; d < dgsolution_ptr->DIM; d++)
					{
						const std::vector<int> order_u_v_w{ order_global_test_basis[d], order_global_solu_basis[d], global_order_visc[d] };

						auto iter_mass = visc_mass.find(order_u_v_w);
						auto iter_stiff = visc_stiff.find(order_u_v_w);

						if (iter_mass == visc_mass.end())
						{
							// test and solution basis function in d-th dimension
							const Basis & test_basis_d = all_basis_alpt_ptr->at(order_global_test_basis[d]);
							const Basis & solu_basis_d = all_basis_alpt_ptr->at(order_global_solu_basis[d]);

							mass[d] = test_basis_d.product_volume_interv(solu_basis_d, all_basis_alpt_ptr->at(global_order_visc[d]), 0, 0, num_gauss_pt_mass);
							stiff[d] = test_basis_d.product_volume_interv(solu_basis_d, all_basis_alpt_ptr->at(global_order_visc[d]), 1, 1, num_gauss_pt_stiff);

							visc_mass.insert(std::make_pair(order_u_v_w, mass[d]));
							visc_stiff.insert(std::make_pair(order_u_v_w, stiff[d]));
						}
						else
						{
							mass[d] = iter_mass->second;
							stiff[d] = iter_stiff->second;
						}
					}

					// row and col of operator matrix
					const int col_j = ptr_solu->order_alpt_basis_in_dg[index_solu_variable].at(order_local_solu_basis);
					const int row_i = ptr_test->order_alpt_basis_in_dg[index_test_variable].at(order_local_test_basis);

					double mat_ij = 0.;

					// if dim_derivative = 0, then we compute \int (u_x * v_x)
					// if dim_derivative = 1, then we compute \int (u_y * v_y)
					for (size_t dim_derivative = 0; dim_derivative < dgsolution_ptr->DIM; dim_derivative++)
					{
						double mat_ij_dim = viscosity_coefficient;
						for (size_t d = 0; d < dgsolution_ptr->DIM; d++)
						{
							if (d == dim_derivative)
							{
								mat_ij_dim *= stiff[d];
							}
							else
							{
								mat_ij_dim *= mass[d];
							}
						}
						mat_ij += mat_ij_dim;
					}

					if (std::abs(mat_ij) >= Const::ROUND_OFF)
					{
						tripletList.push_back(T(row_i, col_j, mat_ij));
					}
				}
			}
		}
	}

	matrix_local.setFromTriplets(tripletList.begin(), tripletList.end());

	mat += matrix_local;
}

void ArtificialViscosity::assemble_matrix(const double viscosity_coefficient, const int viscosity_option)
{
	for (auto const & visc_elem : dgsolution_ptr->viscosity_element)
	{
		assemble_matrix_one_element(visc_elem, viscosity_coefficient, viscosity_option);
	}
}

void ArtificialViscosity::find_all_parents(Element* elem, std::unordered_set<Element*> & parent_elem) const
{
	parent_elem.clear();

	parent_elem.insert(elem);

	// store elements that in each iteration we want to find all parents for these elements
	std::unordered_set<Element*> elem_set;
	// initalized with just the given element
	elem_set.insert(elem);

	// store the new find parents in each iteration
	// we will use this set as elem_set in the next iteration
	std::unordered_set<Element*> new_add;
	while (true)
	{
		// loop over each element in element set
		for (auto const & one_elem : elem_set)
		{
			// for each element, loop over its all parents and insert into parent_elem
			for (auto const & iter : one_elem->hash_ptr_par)
			{
				parent_elem.insert(iter.second);
				new_add.insert(iter.second);
			}
		}
		// if new add elements is non-empty, then it means that there is no parents element for current elem_set
		if (new_add.empty()) { break; }

		elem_set = new_add;
		new_add.clear();
	}
}