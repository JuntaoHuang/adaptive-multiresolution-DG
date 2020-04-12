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
	~DGSolution() {};

	static int DIM;
	static int VEC_NUM;
	static std::string prob;
	static std::vector<int> ind_var_vec;	// specify which components of unknown variables involve time evolution
	

	// return maximum mesh level of current active elements in each dimension, a vector of size dim
	// this will be used to determined time step size dt
	std::vector<int> max_mesh_level_vec() const;

	// return maximum mesh level of current active elements for all dimensions
	int max_mesh_level() const;

	// return maximum mesh level we set initially
	int nmax_level() const { return NMAX; };

	// ------------------------------------------------------------------------
	// initialization
	// ------------------------------------------------------------------------
	// update all coefficients, given a function of separable form f(x1,x2,...,xdim) = g_1(x1) * g_2(x2) * ... * g_dim(xdim)
	// input is a functional: (x, d) -> g_d(x)
	virtual void init_separable_scalar(std::function<double(double, int)> scalar_func);
	// overload for systems of equations, input func should have the same size with num of unknown variable (or equations)
	//virtual void init_separable_system(sd::vector<std::function<double(double, int)>> vector_func);

	// given a initial function, which is summation of separable functions
	// size of sum_func is num of separable functions
	virtual void init_separable_scalar_sum(std::vector<std::function<double(double, int)>> sum_scalar_func);
	// overload for systems of equations
	//virtual void init_separable_system_sum(std::vector<std::vector<std::function<double(double, int)>>> sum_vector_func);

	// given a function of non-separable form f=f(x1,x2,...,xdim)
	// ******************************* TO BE COMPLETED *******************************
	virtual void init_no_separable_scalar(std::function<double(std::vector<double>)> scalar_func) {};
	virtual void init_no_separable_system(std::vector<std::function<double(std::vector<double>)>> vector_func) {};

	// ------------------------------------------------------------------------
	// DG operator
	// ------------------------------------------------------------------------
	// find elements with Alpert basis related to volume and flux terms (no adaptive)
	void find_ptr_vol_alpt();
	void find_ptr_flx_alpt();

	// find elements with interpolation basis related to volume and flux terms (no adaptive)
	// it is different from Alpert basis since there is no orthogonal property
	void find_ptr_vol_intp();
	void find_ptr_flx_intp();	

	// set all pointers (related to vol and flx terms) to all elements
	// this function will only be used for debug, exclude bug of incorrect pointers in vol and flx
	void set_ptr_to_all_elem();

	// ------------------------------------------------------------------------
	// calculate error between DG solution and a given function
	// ------------------------------------------------------------------------
	// calculate L1, L2 and Linf error between DG solution and a given function
	// function is given in separable form in each dimension
	std::vector<double> get_error_separable_scalar(std::function<double(double, int)> func, const int gauss_points = 1) const;
	std::vector<double> get_error_separable_system(std::vector<std::function<double(double, int)>> func, const int gauss_points = 1) const;
	
	// calculate L2 error between DG solution and a given function
	// function is given in separable form in each dimension
	// split form:  (u-uh)^2 = u^2 + uh^2 - 2*u*uh
	double get_L2_error_split_separable_scalar(std::function<double(double, int)> func, const double l2_norm_exact_soln) const;

	// function is given in summation of several separable function
	std::vector<double> get_error_separable_scalar_sum(std::vector<std::function<double(double, int)>> func, const int gauss_points = 1) const;
	std::vector<double> get_error_separable_system_sum(std::vector<std::vector<std::function<double(double, int)>>> func, const int gauss_points = 1) const;
	
	// function is given in non-separable form in each dimension
	std::vector<double> get_error_no_separable_scalar(std::function<double(std::vector<double>)> func, const int gauss_points = 1) const;
	std::vector<double> get_error_no_separable_system(std::vector<std::function<double(std::vector<double>)>> func, const int gauss_points = 1) const;
	std::vector<double> get_error_no_separable_system(std::function<double(std::vector<double>)> func, const int gauss_points, int ind_var) const;
	std::vector<double> get_error_no_separable_system_omp(std::function<double(std::vector<double>)> func, const int gauss_points, int ind_var) const;
	// ------------------------------------------------------------------------
	// calculate error between value of Lagrange and Hermite basis and a given function
	// this is only used in validation of accuracy of interpolation
	// ------------------------------------------------------------------------
	std::vector<double> get_error_Lag(std::function<double(double, int)> func, const int gauss_points) const;
	std::vector<double> get_error_Her(std::function<double(double, int)> func, const int gauss_points) const;

	// compute the error for all flux functions (# of flux = Const::VEC_NUM * Const::DIM)
	std::vector<double> get_error_Lag(std::function<double(std::vector<double>, int, int)>func, const int gauss_points, std::vector< std::vector<bool> > is_intp) const;
	std::vector<double> get_error_Lag(std::vector< std::vector< std::function<double(std::vector<double>)> > >func, const int gauss_points) const;

	std::vector<double> get_error_Lag(std::function<double(std::vector<double>)> func, const int gauss_points) const;

	// compute the error for all flux functions (# of flux = Const::VEC_NUM * Const::DIM)
	std::vector<double> get_error_Her(std::function<double(std::vector<double>, int, int)> func, const int gauss_points, std::vector< std::vector<bool> > is_intp) const;
	std::vector<double> get_error_Her(std::vector< std::vector< std::function<double(std::vector<double>)> > > func, const int gauss_points) const;
	
	std::vector<double> get_error_Her(std::function<double(std::vector<double>)> func, const int gauss_points) const;
	void plot2d() const;
	
	// ------------------------------------------------------------------------
	// output information of dgsolution
	// num of elements
	// num of alpert basis
	// num of interpolation basis
	// print rhs in screen
	// ------------------------------------------------------------------------
	int size_elem() const;
	int size_basis_alpt() const; 
	int size_basis_intp() const;
	int get_dof() const;
	void print_rhs() const;


	// set all Element::rhs in all elements to be zero
	void set_rhs_zero();

	// set all Element::source in all elements to be zero
	void set_source_zero();

	// set all Element::ucoe_alpt in all elements to be zero
	void set_ucoe_alpt_zero();

	// multiply Element::ucoe_alpt in all elements by a constant
	void multiply_ucoe_alpt_by_const(const double constant);

	// copy Element::ucoe_alpt to Element::ucoe_alpt_predict
	void copy_ucoe_to_predict();

	// copy Element::ucoe_alpt_predict to Element::ucoe_alpt
	void copy_predict_to_ucoe();

	// copy Element::ucoe_ut to Element::ucoe_ut_predict
	void copy_ucoe_ut_to_predict();

	// copy Element::ucoe_ut_predict to Element::ucoe_ut
	void copy_predict_to_ucoe_ut();

	// copy Element::ucoe_alpt_t_m1 to Element::ucoe_alpt_predict_t_m1
	void copy_ucoe_to_predict_t_m1();

	// copy Element::ucoe_alpt_predict_t_m1 to Element::ucoe_alpt_t_m1
	void copy_predict_to_ucoe_t_m1();

	// copy Element::ucoe_alpt to Element::ucoe_alpt_t_m1
	void copy_ucoe_to_ucoem1();

	// copy Element::ucoe_alpt_t_m1 to Element::ucoe_alpt
	void copy_ucoem1_to_ucoe();

	// copy Element::ucoe_alpt to Element::ucoe_ut
	void copy_ucoe_to_ucoe_ut();

	// copy Element::ucoe_ut to Element::ucoe_alpt
	void copy_ucoe_ut_to_ucoe();

	std::unordered_map<int, Element> dg;

	// artificial viscosity elements
	std::unordered_set<Element*> viscosity_element;

	// return number of elements with non-zero artificial viscosity
	int num_artific_visc_elem() const { return viscosity_element.size(); };
	
	// key member function of the DGSolution class, calculate value of DG solution at point x
	// it is vector of size the number of unknown
	std::vector<double> val(const std::vector<double> & x, const std::vector<int> & derivative) const;
	
	// given a multidimensional function in seperable form
	// output coefficient of projection in each dimension (size of DIM * num_all_alpert_basis)
	VecMultiD<double> seperable_project(std::function<double(double, int)> func) const;

	/**
	 * @brief update coefficient of source in a element
	 * 
	 * @param elem 
	 * @param coefficient_1D 
	 * @param index_var 
	 */
	void source_elem_separable(Element & elem, const VecMultiD<double> & coefficient_1D, const int index_var = 0);

protected:

	const bool sparse;
	const int level_init;	
	const int NMAX;

	Hash hash;
	AllBasis<AlptBasis> all_bas;
	AllBasis<LagrBasis> all_bas_Lag;
	AllBasis<HermBasis> all_bas_Her;	

	// initialization coefficient of a given element based on coefficient of projection of separable 
	void init_elem_separable(Element & elem, const VecMultiD<double> & coefficient_1D, const int index_var = 0);

	// update order of all basis function in dg map
	// this will update Element::order_alpt_basis_in_dg and Element::order_intp_basis_in_dg
	// it will be used in assemble matrix for DG operator
	// this function will be called in initialization, refinement and coarsen in class DGAdapt
	void update_order_all_basis_in_dgmap();

private:

	// calculate value of DG solution at point x with interpolation basis
	std::vector<double> val_Lag(const std::vector<double> & x) const;

	std::vector<double> val_Her(const std::vector<double> & x) const;

	
	// elements that have intersection with artificial viscosity elements
	std::unordered_set<Element*> viscosity_intersect_element;

	// update viscosity_intersect_element
	void update_viscosity_intersect_element();

	// set all Element::fp_intp in all elements to be zero
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
		
	// calculate rhs
	void rhs_vol_intp(const double operator_coeff, const int dim, const VecMultiD<double> & mat_operator, const VecMultiD<double> & mat_mass, const bool is_only_update_new = false);

	void rhs_flx_intp(const double operator_coeff, const int dim, const VecMultiD<double> & mat_operator, const VecMultiD<double> & mat_mass, const bool is_only_update_new = false);			
};

