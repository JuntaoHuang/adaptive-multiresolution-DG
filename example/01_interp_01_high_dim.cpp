#include "AllBasis.h"
#include "AlptBasis.h"
#include "Basis.h"
#include "BilinearForm.h"
#include "DGAdapt.h"
#include "DGAdaptIntp.h"
#include "DGSolution.h"
#include "Element.h"
#include "ExactSolution.h"
#include "Hash.h"
#include "HermBasis.h"
#include "Interpolation.h"
#include "IO.h"
#include "LagrBasis.h"
#include "LinearForm.h"
#include "ODESolver.h"
#include "OperatorMatrix1D.h"
#include "Quad.h"
#include "subs.h"
#include "VecMultiD.h"
#include "FastMultiplyLU.h"
#include "Timer.h"

int main()
{
	/* 
		initialization
	*/	
	AlptBasis::PMAX = 2;

	LagrBasis::PMAX = 3;
	LagrBasis::msh_case = 2;

	HermBasis::PMAX = 3;
	HermBasis::msh_case = 1;

	Element::PMAX_alpt = AlptBasis::PMAX;	// max polynomial degree for Alpert's basis functions
	Element::PMAX_intp = LagrBasis::PMAX;	// max polynomial degree for interpolation basis functions
	Element::DIM = 4;			// dimension
	Element::VEC_NUM = 1;		// num of unknown variables in PDEs

	DGSolution::DIM = Element::DIM;
	DGSolution::VEC_NUM = Element::VEC_NUM;

	Interpolation::DIM = Element::DIM;
	Interpolation::VEC_NUM = Element::VEC_NUM;

	DGSolution::ind_var_vec = { 0 };
	DGAdapt::indicator_var_adapt = { 0 };

	Element::is_intp.resize(Element::VEC_NUM);
	for (size_t num = 0; num < Element::VEC_NUM; num++) { Element::is_intp[num] = std::vector<bool>(Element::DIM, true); }

	// constant variable
	const int DIM = Element::DIM;
	const int VEC_NUM = Element::VEC_NUM;
	int NMAX = 8;
	int N_init = 1;
	const bool sparse = false;
	const std::string boundary_type = "period";

	const bool is_adapt_find_ptr_alpt = true;	// variable control if need to adaptively find out pointers related to Alpert basis in DG operators
	const bool is_adapt_find_ptr_intp = true;	// variable control if need to adaptively find out pointers related to interpolation basis in DG operators

	// adaptive parameter
	const double refine_eps = 1e-4;
	const double coarsen_eta = -1;

	// hash key
	Hash hash;

	LagrBasis::set_interp_msh01();

	HermBasis::set_interp_msh01();

	AllBasis<LagrBasis> all_bas_lagr(NMAX);

	AllBasis<HermBasis> all_bas_herm(NMAX);

	AllBasis<AlptBasis> all_bas_alpt(NMAX);

	// operator matrix for Alpert basis
	OperatorMatrix1D<AlptBasis,AlptBasis> oper_matx_alpt(all_bas_alpt, all_bas_alpt, boundary_type);

	OperatorMatrix1D<HermBasis,AlptBasis> oper_matx_herm(all_bas_herm, all_bas_alpt, boundary_type);

	OperatorMatrix1D<LagrBasis,AlptBasis> oper_matx_lagr(all_bas_lagr, all_bas_alpt, boundary_type);

	OperatorMatrix1D<LagrBasis,LagrBasis> oper_matx_lagr_all(all_bas_lagr, all_bas_lagr, boundary_type);

	OperatorMatrix1D<HermBasis,HermBasis> oper_matx_herm_all(all_bas_herm, all_bas_herm, boundary_type);

	DGAdaptIntp dg_solu(sparse, N_init, NMAX, all_bas_alpt, all_bas_lagr, all_bas_herm, hash, refine_eps, coarsen_eta, is_adapt_find_ptr_alpt, is_adapt_find_ptr_intp, oper_matx_lagr_all, oper_matx_herm_all);

	// update pointers related to DG operators
    if (is_adapt_find_ptr_alpt)
    {
        dg_solu.find_ptr_vol_alpt();
        dg_solu.find_ptr_flx_alpt();
    }

	// example for Lagrange interpolation
	// function in seperable form
	// f(x_1, x_2, ..., x_n) = g_1(x_1) * g_2(x_2) * ... * g_n(x_n)
	auto init_func_in_each_dim = [](double x, int d) 
	{ 
		if (d == 0) { return sin(2 * Const::PI * x); }
		else { return cos(2 * Const::PI * x); }
	};

	// function to be interplated
	auto init_func_multi_dim = [&](std::vector<double> x, int i)->double 
	{
		double f = 1.0;
		for (size_t d = 0; d < Element::DIM; d++) { f = f * init_func_in_each_dim(x[d], d); }
		return f;
	};		

	// adaptive initialization with Lagrange interpolation basis
	LagrInterpolation interp(dg_solu);
	
	// record code running time
	auto start_evolution_time = std::chrono::high_resolution_clock::now();
	std::cout << "interpolation begin" << std::endl;

	dg_solu.init_adaptive_intp_Lag(init_func_multi_dim, interp);

	auto stop_evolution_time = std::chrono::high_resolution_clock::now(); 
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop_evolution_time - start_evolution_time);
	std::cout << "elasped time in interpolation: " << duration.count()/1e6 << " seconds"<< std::endl;

	return 0;
}

