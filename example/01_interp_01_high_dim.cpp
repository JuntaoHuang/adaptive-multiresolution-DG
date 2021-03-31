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
#include "Optparser.h"
#include "Quad.h"
#include "subs.h"
#include "VecMultiD.h"
#include "FastMultiplyLU.h"
#include "Timer.h"

// example:
// 
// ./01_interp_01_high_dim -r 1e-1 -gp 3 -rp 10000
// 
// -r:	error threshold
// -gp: number of Gauss-Legendre points in each small elements (optional)
// -rp: total number of random points

int main(int argc, char *argv[])
{
	LagrBasis::PMAX = 3;
	LagrBasis::msh_case = 1;

	HermBasis::PMAX = 3;
	HermBasis::msh_case = 1;

	AlptBasis::PMAX = LagrBasis::PMAX;

	Element::PMAX_alpt = AlptBasis::PMAX;	// max polynomial degree for Alpert's basis functions
	Element::PMAX_intp = LagrBasis::PMAX;	// max polynomial degree for interpolation basis functions
	Element::DIM = 2;			// dimension
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
	double refine_eps = 1e-2;
	const double coarsen_eta = -1;

	// number of Gauss points and random points in computing error
	int num_gauss_pt = 0;
	int num_random_pt = 10000;

	OptionsParser args(argc, argv);
	args.AddOption(&NMAX, "-NM", "--max-mesh-level", "Maximum mesh level");
	args.AddOption(&refine_eps, "-r", "--refine-epsilon", "refine parameter epsilon");
	args.AddOption(&num_gauss_pt, "-gp", "--gauss-point", "number of Gaussian points");
	args.AddOption(&num_random_pt, "-rp", "--random-point", "number of random points");

	args.Parse();
	if (!args.Good())
	{
		args.PrintUsage(std::cout);
		return 1;
	}
	args.PrintOptions(std::cout);

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

	std::cout << "DoF after refinement: " << dg_solu.size_basis_intp() << std::endl
			<< "num of refinements: " << dg_solu.refine_num() << std::endl
			<< "max mesh level: " << dg_solu.max_mesh_level() << std::endl;

	auto stop_evolution_time = std::chrono::high_resolution_clock::now(); 
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop_evolution_time - start_evolution_time);
	std::cout << "elasped time in interpolation: " << duration.count()/1e6 << " seconds"<< std::endl;

	// compute error for Lagrange interpolation	
	auto init_func_multi_dim_scalar = [&](std::vector<double> x)->double { return init_func_multi_dim(x, 0); };

	if (num_gauss_pt > 0) 
	{
	    std::vector<double> err = dg_solu.get_error_Lag_scalar(init_func_multi_dim_scalar, num_gauss_pt);	
	    std::cout << "L1, L2 and Linf error (computed by Gaussian-Legendre quadrature): " << std::endl 
		    << err[0] << ", " << err[1] << ", " << err[2] << std::endl;
	}

	for (int i = 0; i < 3; i++)
	{

	// loop to calculate errors when increasing random points by 10
	// record running time for error calculation
	auto start_error_compute_time = std::chrono::high_resolution_clock::now();

	std::vector<double> err = dg_solu.get_error_Lag_scalar_random_points(init_func_multi_dim_scalar, num_random_pt);
	std::cout << "L1, L2 and Linf error (computed by " << num_random_pt << " random points): " << std::endl
			<< err[0] << ", " << err[1] << ", " << err[2] << std::endl;
	
	auto stop_error_compute_time = std::chrono::high_resolution_clock::now(); 
	auto duration_err = std::chrono::duration_cast<std::chrono::microseconds>(stop_error_compute_time - start_error_compute_time);
	std::cout << "elasped time in error computation: " << duration_err.count()/1e6 << " seconds"<< std::endl;
	
	num_random_pt = num_random_pt * 10;
	
	}

	return 0;
}

