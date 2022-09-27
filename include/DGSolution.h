#pragma once
#include "libs.h"
#include "Element.h"
#include "AllBasis.h"
#include "AlptBasis.h"
#include "LagrBasis.h"
#include "HermBasis.h"
#include "Hash.h"
#include "OperatorMatrix1D.h"


// DGsolution 
class DGSolution
{
friend class Interpolation;
friend class LagrInterpolation;
friend class HermInterpolation;
friend class LargViscosInterpolation;
friend class IO;

public:
	// sparse: variable control initialization using sparse grid or not
	DGSolution(const bool sparse_, const int level_init_, const int NMAX_, AllBasis<AlptBasis> & all_bas_, AllBasis<LagrBasis> all_bas_Lag_, AllBasis<HermBasis> all_bas_Her_, Hash & hash_);
	// the first real_dim (= DIM - auxiliary_dim) use full grid, the remaining auxiliary dimension use only zero level grid
	// this will be used in Vlasov-Maxwell simulation
	DGSolution(const bool sparse_, const int level_init_, const int NMAX_, const int auxiliary_dim_, AllBasis<AlptBasis> & all_bas_, AllBasis<LagrBasis> all_bas_Lag_, AllBasis<HermBasis> all_bas_Her_, Hash & hash_);	
	~DGSolution() {};

	static int DIM;
	static int VEC_NUM;
	static std::string prob;
	static std::vector<int> ind_var_vec;	///< specify which components of unknown variables involve time evolution
	
	int auxiliary_dim;

	// ------------------------------------------------------------------------
	/// return maximum mesh level of current active elements in each dimension, a vector of size dim
	/// this will be used to determined time step size dt in each time step
	std::vector<int> max_mesh_level_vec() const;

	/// return maximum mesh level of current active elements for all dimensions
	int max_mesh_level() const;

	/// return maximum mesh level we set initially
	int nmax_level() const { return NMAX; };

	// return maximum absolute value of the solution with sampling points from a given maximum mesh level
	std::vector<double> max_abs_value(const std::vector<int> & sample_max_mesh_level) const;

	// ------------------------------------------------------------------------
	// pointers to elements when computing DG operator
	// ------------------------------------------------------------------------
	/// find elements with Alpert basis related to volume and flux terms by loop over all the elements
	void find_ptr_vol_alpt();
	void find_ptr_flx_alpt();

	/// find elements with interpolation basis related to volume and flux terms by loop over all the elements.
	/// This is different from Alpert basis since there is no orthogonal property
	void find_ptr_vol_intp();
	void find_ptr_flx_intp();	

	void find_ptr_general();
	
	/// set all pointers (related to vol and flx terms) to all elements.
	/// this function will only be used for debug, exclude bug of incorrect pointers in vol and flx.
	void set_ptr_to_all_elem();

	// ------------------------------------------------------------------------
	// calculate error between DG solution and a given function
	// ------------------------------------------------------------------------
	// calculate L1, L2 and Linf error between DG solution and a given function
	// function is given in separable form in each dimension
	std::vector<double> get_error_separable_scalar(std::function<double(double, int)> func, const int gauss_points = 1) const;
	std::vector<double> get_error_separable_system(std::vector<std::function<double(double, int)>> func, const int gauss_points = 1) const;

	// function is given in summation of several separable function
	std::vector<double> get_error_separable_scalar_sum(std::vector<std::function<double(double, int)>> func, const int gauss_points = 1) const;
	std::vector<double> get_error_separable_system_sum(std::vector<std::vector<std::function<double(double, int)>>> func, const int gauss_points = 1) const;	

	// function is given in non-separable form in each dimension
	std::vector<double> get_error_no_separable_scalar(std::function<double(std::vector<double>)> func, const int gauss_points = 1) const;
	std::vector<double> get_error_no_separable_system(std::vector<std::function<double(std::vector<double>)>> func, const int gauss_points = 1) const;
	std::vector<double> get_error_no_separable_system_each(std::vector<std::function<double(std::vector<double>)>> func, const int gauss_points, int ind_var) const;
	std::vector<double> get_error_no_separable_system(std::function<double(std::vector<double>)> func, const int gauss_points, int ind_var) const;
	std::vector<double> get_error_no_separable_system_omp(std::function<double(std::vector<double>)> func, const int gauss_points, int ind_var) const;

	/**
	 * @brief calculate the L2 error between numerical solution and a given function using split form:  (u - u_h)^2 = u^2 + u_h^2 - 2 * u * u_h
	 * 
	 * @param func 					function is given in separable form in each dimension
	 * @param l2_norm_exact_soln 	L2 norm of exact solution
	 * @return double				L2 error
	 * 
	 * @note	If the L2 error is less than 1e-6, then the square of this error might be around round-off error. Then sqrt of negative value might happen due to round off error.
	 */
	double get_L2_error_split_separable_scalar(std::function<double(double, int)> func, const double l2_norm_exact_soln) const;

	// ------------------------------------------------------------------------
	// calculate error between value of Lagrange and Hermite basis and a given function
	// this is only used in validation of accuracy of interpolation
	// ------------------------------------------------------------------------
	std::vector<double> get_error_Lag(std::function<double(double, int)> func, const int gauss_points) const;
	std::vector<double> get_error_Her(std::function<double(double, int)> func, const int gauss_points) const;

	std::vector<double> get_error_Lag_scalar(std::function<double(std::vector<double>)> func, const int gauss_points) const;
	std::vector<double> get_error_Lag_scalar_random_points(std::function<double(std::vector<double>)> func, const int num_points) const;

	// compute the error for all flux functions (# of flux = Const::VEC_NUM * Const::DIM)
	std::vector<double> get_error_Lag(std::function<double(std::vector<double>, int, int)>func, const int gauss_points, std::vector< std::vector<bool> > is_intp) const;
	std::vector<double> get_error_Lag(std::vector< std::vector< std::function<double(std::vector<double>)> > >func, const int gauss_points) const;

	std::vector<double> get_error_Lag(std::function<double(std::vector<double>)> func, const int gauss_points) const;

	// compute the error for all flux functions (# of flux = Const::VEC_NUM * Const::DIM)
	std::vector<double> get_error_Her(std::function<double(std::vector<double>, int, int)> func, const int gauss_points, std::vector< std::vector<bool> > is_intp) const;
	std::vector<double> get_error_Her(std::vector< std::vector< std::function<double(std::vector<double>)> > > func, const int gauss_points) const;
	
	std::vector<double> get_error_Her(std::function<double(std::vector<double>)> func, const int gauss_points) const;
	
	// ------------------------------------------------------------------------
	// output information of dgsolution
	// num of elements
	// num of alpert basis
	// num of interpolation basis
	// print rhs in screen
	// ------------------------------------------------------------------------
	int size_elem() const;			///< size of dg map
	int size_basis_alpt() const; 	///< size of dg map * (k+1)^d where k is polynomial degree in Alpert basis
	int size_basis_intp() const;	///< size of dg map * (m+1)^d where k is polynomial degree in interpolation basis
	int get_dof() const;			///< dof is defined as size_basis_alpt() * (size of ind_var_vec)
	void print_rhs() const;

	/**
	 * @brief copy Element::ucoe_alpt in dg_E to Element::ucoe_alpt in dg_f
	 * 
	 * @param E 
	 * @param num_vec_f 
	 * @param num_vec_E 
	 * @param vel_dim_f velocity dimensions in f
	 */
	void copy_ucoe_alpt_to_f(DGSolution & E, const std::vector<int> & num_vec_f, const std::vector<int> & num_vec_E, const std::vector<int> & vel_dim_f);

	// ------------------------------------------------------------------------
	// copy or set zero to coefficients in all the elements
	// ------------------------------------------------------------------------
	/// set Element::rhs in all elements to be zero
	void set_rhs_zero();

	/// set Element::source in all elements to be zero
	void set_source_zero();

	/// set Element::ucoe_alpt in all elements to be zero
	void set_ucoe_alpt_zero();

	/// multiply Element::ucoe_alpt in all elements by a constant
	void multiply_ucoe_alpt_by_const(const double constant);

	/// copy Element::ucoe_alpt to Element::ucoe_alpt_predict
	void copy_ucoe_to_predict();

	/// copy Element::ucoe_alpt_predict to Element::ucoe_alpt
	void copy_predict_to_ucoe();

	/// copy Element::ucoe_alpt to Element::ucoe_alpt_other
	void copy_ucoe_to_other();

	/// copy Element::up_intp to Element::up_intp_other
	void copy_up_intp_to_other();

	/// exchange Element::ucoe_alpt and Element::ucoe_alpt_other
	void exchange_ucoe_and_other();

	/// exchange Element::up_intp and Element::up_intp_other
	void exchange_up_intp_and_other();

	/// copy Element::ucoe_ut to Element::ucoe_ut_predict
	void copy_ucoe_ut_to_predict();

	/// copy Element::ucoe_ut_predict to Element::ucoe_ut
	void copy_predict_to_ucoe_ut();

	///	copy Element::ucoe_alpt_t_m1 to Element::ucoe_alpt_predict_t_m1
	void copy_ucoe_to_predict_t_m1();

	/// copy Element::ucoe_alpt_predict_t_m1 to Element::ucoe_alpt_t_m1
	void copy_predict_to_ucoe_t_m1();

	/// copy Element::ucoe_alpt to Element::ucoe_alpt_t_m1
	void copy_ucoe_to_ucoem1();

	/// copy Element::ucoe_alpt_t_m1 to Element::ucoe_alpt
	void copy_ucoem1_to_ucoe();

	/// copy Element::ucoe_alpt to Element::ucoe_ut
	void copy_ucoe_to_ucoe_ut();

	/// copy Element::ucoe_ut to Element::ucoe_alpt
	void copy_ucoe_ut_to_ucoe();

	/// copy Element::rhs to Element::ucoe_alpt
	void copy_rhs_to_ucoe();

	/// key member in this class, store all the active elements and its hash key
	std::unordered_map<int, Element> dg;

	/// artificial viscosity elements, this will be used in shock capturing when solving conservation laws
	std::unordered_set<Element*> viscosity_element;

	/// return number of elements with non-zero artificial viscosity
	int num_artific_visc_elem() const { return viscosity_element.size(); };
	
	/**
	 * @brief calculate value of DG solution at a given point
	 * 
	 * @param x 					a given point
	 * @param derivative 			order of derivative in each dimension
	 * @return std::vector<double> 	vector of size the number of unknown variables
	 */
	std::vector<double> val(const std::vector<double> & x, const std::vector<int> & derivative) const;
	
	/**
	 * @brief	project a function in seperable form in each dim to basis in 1D
	 * 
	 * @param func 					a multidimensional function in seperable form
	 * @return VecMultiD<double> 	a two dimension vector coefficient with size (dim, # all_basis_1D)
	 * 								coefficient.at(d, order) denote the inner product of order-th basis function with initial function in d-th dim
	 */
	VecMultiD<double> seperable_project(std::function<double(double, int)> func) const;

	/**
	 * @brief update coefficient of source in a element
	 * 
	 * @param elem 
	 * @param coefficient_1D 	a two dimension vector coefficient with size (dim, # all_basis_1D)
	 * 							coefficient.at(d, order) denote the inner product of order-th basis function with initial function in d-th dim
	 * @param index_var 		index of unknown variable
	 */
	void source_elem_separable(Element & elem, const VecMultiD<double> & coefficient_1D, const int index_var = 0);

protected:

	const bool sparse;		///< control sparse grid (true) or full grid (false)
	const int level_init;	///< initial mesh level
	const int NMAX;			///< maximum mesh level

	Hash hash;
	AllBasis<AlptBasis> all_bas;
	AllBasis<LagrBasis> all_bas_Lag;
	AllBasis<HermBasis> all_bas_Her;	

	/**
	 * @brief initialization coefficient of a given element based on coefficient of projection of separable functions
	 * 
	 * @param elem 				given element
	 * @param coefficient_1D 	size of DIM * (num of all 1D Alpert basis)
	 * @param index_var 		index of unknown variables
	 */
	void init_elem_separable(Element & elem, const VecMultiD<double> & coefficient_1D, const int index_var = 0);

	/**
	 * @brief update order of all basis function in dg map
	 * 
	 * 		this will update Element::order_alpt_basis_in_dg and Element::order_intp_basis_in_dg
	 * 		it will be used in assemble matrix for DG operator
	 * 		this function will be called in initialization, refinement and coarsen in class DGAdapt
	 */
	void update_order_all_basis_in_dgmap();

private:

	// calculate value of DG solution at point x with interpolation basis
	std::vector<double> val_Lag(const std::vector<double> & x) const;

	std::vector<double> val_Her(const std::vector<double> & x) const;

	
	/// elements that have intersection with artificial viscosity elements
	std::unordered_set<Element*> viscosity_intersect_element;

	/// update viscosity_intersect_element
	void update_viscosity_intersect_element();

	/// set all Element::fp_intp in all elements to be zero
	void set_fp_intp_zero();

	/**
	 * @brief return value of flux function of collection of all Lagrange basis (with coefficients fucoe_intp) at any point x
	 * 
	 * @param x 
	 * @param ii index of unknown variable
	 * @param dd index of dimension
	 * @return double 
	 */
	double val_Lag(const std::vector<double> & x, const int ii, const int dd) const;

	/**
	 * @brief return value of flux function of collection of all Hermite basis (with coefficients fucoe_intp) at any point x
	 * 
	 * @param x 
	 * @param ii index of unknown variable
	 * @param dd index of dimension
	 * @return double 
	 */
	double val_Her(const std::vector<double> & x, const int ii, const int dd) const;
};

