#pragma once
#include "DGAdapt.h"
#include "Interpolation.h"

// DGAdaptIntp is derived from DGAdapt
class DGAdaptIntp :
	public DGAdapt
{
public:
	DGAdaptIntp(const bool sparse_, const int level_init_, const int NMAX_, AllBasis<AlptBasis> & all_bas_, AllBasis<LagrBasis> & all_bas_Lag_, AllBasis<HermBasis> & all_bas_Her_, Hash & hash_, const double eps_, const double eta_, const bool is_find_ptr_alpt_, const bool is_find_ptr_intp_, const OperatorMatrix1D<LagrBasis, LagrBasis> & matrix_Lag_, const OperatorMatrix1D<HermBasis, HermBasis> & matrix_Her_);
	~DGAdaptIntp() {};

	void init_adaptive_intp_Lag(std::function<double(std::vector<double>, int)> func, LagrInterpolation & interp);

	void init_adaptive_intp_Herm(std::function<double(std::vector<double>, int, std::vector<int>)> func, HermInterpolation & interp);

	double get_L2_error_split_adaptive_intp_scalar(DGAdaptIntp & exact_solu) const;

	// return number of refinements
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

