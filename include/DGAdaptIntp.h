#pragma once
#include "DGAdapt.h"
#include "Interpolation.h"

/**
 * @brief This class is mainly used in the initialization of DG solution by adaptive interpolation
 * 
 */
class DGAdaptIntp:
	public DGAdapt
{
public:
	DGAdaptIntp(const bool sparse_, const int level_init_, const int NMAX_, AllBasis<AlptBasis> & all_bas_, AllBasis<LagrBasis> & all_bas_Lag_, AllBasis<HermBasis> & all_bas_Her_, Hash & hash_, const double eps_, const double eta_, const bool is_find_ptr_alpt_, const bool is_find_ptr_intp_, const OperatorMatrix1D<LagrBasis, LagrBasis> & matrix_Lag_, const OperatorMatrix1D<HermBasis, HermBasis> & matrix_Her_);
	DGAdaptIntp(const bool sparse_, const int level_init_, const int NMAX_, AllBasis<AlptBasis> & all_bas_, AllBasis<LagrBasis> & all_bas_Lag_, AllBasis<HermBasis> & all_bas_Her_, Hash & hash_, const double eps_, const double eta_, const bool is_find_ptr_alpt_, const bool is_find_ptr_intp_, const bool is_find_ptr_general_, const OperatorMatrix1D<LagrBasis, LagrBasis> & matrix_Lag_, const OperatorMatrix1D<HermBasis, HermBasis> & matrix_Her_);	
	~DGAdaptIntp() {};

	/**
	 * @brief adaptive Lagrange interpolation for a given function
	 * this will update coefficients of interpolation basis, i.e. Element::ucoe_intp in DG solution
	 * this function can be used in the initialization of DG solution
	 * 
	 * @param func 
	 * @param interp 
	 * @return ** void 
	 */
	void init_adaptive_intp_Lag(std::function<double(std::vector<double>, int)> func, LagrInterpolation & interp);

	/**
	 * @brief initialization of DG solution given in non-separable form
	 * this function will do the following steps:
	 * step 1: call void init_adaptive_intp_Lag() to 
	 * do adaptive Lagrange interpolation and update coefficients of interpolation basis (Element::ucoe_intp in DG solution)
	 * step 2:
	 * transformation coefficient of Lagrange basis to Alpert basis
	 * this function can be used in the initialization of DG solution
	 * 
	 * @param func the first index in func: coordinate in physical domain;
	 * 			the second index in func: the number of unknown variables
	 */
	void init_adaptive_intp(std::function<double(std::vector<double>, int)> func);

	/**
	 * @brief adaptive Hermite interpolation for a given function
	 * this will update coefficients of interpolation basis, i.e. Element::ucoe_intp in DG solution
	 * 
	 * @note this function is similar to void init_adaptive_intp_Lag()
	 * the current one use Hermite interpolation but the other one use Lagrange interpolation
	 * 
	 * @param func 
	 * @param interp 
	 */
	void init_adaptive_intp_Herm(std::function<double(std::vector<double>, int, std::vector<int>)> func, HermInterpolation & interp);

	// compute L2 error between two numerical solutions represented by DG basis
	// err_L2 := \sqrt( \int (u_h - v_h)^2 ) = \sqrt( \int(u_h^2) - \int(2 * u_h * v_h) + \int(v_h^2) )
	// This is much faster than computing error using Gaussian-Legendre quadrature rule
	// since we can make use of orthonormality property of DG basis
	double get_L2_error_split_adaptive_intp_scalar(DGSolution & exact_solu) const;
	
	double get_L2_error_split_adaptive_intp_system(DGSolution & exact_solu, const int num_vec = 0) const;

	// return number of refinements in adaptive interpolation
	int refine_num() const { return refine_num_; };

private:
	const OperatorMatrix1D<LagrBasis, LagrBasis>*  matrix_Lag_ptr;

	const OperatorMatrix1D<HermBasis, HermBasis>*  matrix_Her_ptr;

	int refine_num_;

	// recursively refine with Lagrange interpolation basis
	void refine_init_intp_Lag(std::function<double(std::vector<double>, int)> func, LagrInterpolation & interp);

	// evaluate L2 norm with Lagrange interpolation basis
	double indicator_norm_intp_Lag(Element & elem) const;


	// recursively refine with Hermite interpolation basis
	void refine_init_intp_Herm(std::function<double(std::vector<double>, int, std::vector<int>)> func, HermInterpolation & interp);

	// evaluat L2 norm with Hermite interpolation basis
	double indicator_norm_intp_Herm(Element & elem) const;

	// // evaluate interpolation coefficients for new added elements in refine (for initialization)  
	// void init_new_ele_coe_intp(std::function<double(std::vector<double>, int)> func, LagrInterpolation & interp);
	
};

