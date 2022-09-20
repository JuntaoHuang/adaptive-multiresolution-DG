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

int main(int argc, char *argv[])
{	
	// --------------------------------------------------------------------------------------------
	// --- Part 1: preliminary part	---
	// static variables
	const int DIM = 2;
	int auxiliary_dim = 1;
	int real_dim = DIM - auxiliary_dim;

	AlptBasis::PMAX = 2;

	LagrBasis::PMAX = 3;
	LagrBasis::msh_case = 1;

	HermBasis::PMAX = 3;
	HermBasis::msh_case = 1;

	Element::PMAX_alpt = AlptBasis::PMAX;	// max polynomial degree for Alpert's basis functions
	Element::PMAX_intp = HermBasis::PMAX;	// max polynomial degree for interpolation basis functions
	Element::DIM = DIM;			// dimension
	Element::VEC_NUM = 1;		// num of unknown variables in PDEs

	DGSolution::DIM = Element::DIM;
	DGSolution::VEC_NUM = Element::VEC_NUM;

	Interpolation::DIM = DGSolution::DIM;
	Interpolation::VEC_NUM = DGSolution::VEC_NUM;

	DGSolution::ind_var_vec = { 0 };
	DGAdapt::indicator_var_adapt = { 0 };

	Element::is_intp.resize(Element::VEC_NUM);
	for (size_t num = 0; num < Element::VEC_NUM; num++) { Element::is_intp[num] = std::vector<bool>(Element::DIM, true); }

	// constant variable
	int NMAX = 4;
	int N_init = NMAX;	
	int is_sparse = 0;
	const std::string boundary_type = "period";
	double final_time = 0.1;
	const double cfl = 0.1;
	const bool is_adapt_find_ptr_alpt = true;	// variable control if need to adaptively find out pointers related to Alpert basis in DG operators
	const bool is_adapt_find_ptr_intp = false;	// variable control if need to adaptively find out pointers related to interpolation basis in DG operators

	// adaptive parameter
	// if need to test code without adaptive, just set refine_eps large number 1e6, then no refine
	// and set coarsen_eta negative number -1, then no coarsen
	const double refine_eps = 1e10;
	// const double coarsen_eta = refine_eps/10.;
	const double coarsen_eta = -1;

	OptionsParser args(argc, argv);
	args.AddOption(&NMAX, "-N", "--max-mesh-level", "Maximum mesh level");
	args.AddOption(&is_sparse, "-s", "--sparse-grid", "sparse grid (1) or full grid (0)");
	args.AddOption(&final_time, "-tf", "--final-time", "Final time; start time is 0.");
	args.Parse();
	if (!args.Good())
	{
		args.PrintUsage(std::cout);
		return 1;
	}
	args.PrintOptions(std::cout);

	N_init = NMAX;
	bool sparse = (is_sparse == 1) ? true : false;

	// hash key
	Hash hash;

	LagrBasis::set_interp_msh01();
	HermBasis::set_interp_msh01();

	AllBasis<LagrBasis> all_bas_Lag(NMAX);
	AllBasis<HermBasis> all_bas_Her(NMAX);
	AllBasis<AlptBasis> all_bas(NMAX);

	// operator matrix for Alpert basis
	OperatorMatrix1D<AlptBasis,AlptBasis> oper_matx(all_bas, all_bas, boundary_type);
	// --- End of Part 1 ---
	// --------------------------------------------------------------------------------------------




	// --------------------------------------------------------------------------------------------
	// --- Part 2: initialization of DG solution ---
	DGAdapt dg_solu(sparse, N_init, NMAX, auxiliary_dim, all_bas, all_bas_Lag, all_bas_Her, hash, refine_eps, coarsen_eta, is_adapt_find_ptr_alpt, is_adapt_find_ptr_intp);

	// distribution function f
	DGAdapt f(sparse, N_init, NMAX, all_bas, all_bas_Lag, all_bas_Her, hash, refine_eps, coarsen_eta, is_adapt_find_ptr_alpt, is_adapt_find_ptr_intp);

	// sin(2*pi*x_1) * (2*x_2 - 1) * sqrt(3)
	auto init_func_1 = [](double x, int d)
	{
		if (d == 0) { return sin(x); }
		// else if (d == 1) { return (2*x-1)*sqrt(3.); }
		else if (d == 1) { return x; }
		// else { return 1.; }
	};
	f.init_separable_scalar(init_func_1);
	
	std::vector<int> moment_order{0, 1, 2};
	std::vector<double> moment_order_weight{1.0, 3.0, -1.};
	int num_vec = 0;
	dg_solu.compute_moment_full_grid(f, moment_order, moment_order_weight, num_vec);
	dg_solu.copy_rhs_to_ucoe();

	const int num_gauss_pt = 7;
	// exact solution at final time: u(x_1, x_2, ..., x_d, t) = cos(2 * PI * ( sum_(d=1)^DIM (x_d) - DIM * t))
	auto final_func = [=](std::vector<double> x) 
	{	
		return sin(x[0]) * 1.25;
	};
	std::vector<double> err_l1_l2_linf = dg_solu.get_error_no_separable_scalar(final_func, num_gauss_pt);
	std::cout << "L1, L2 and Linf error at final time: " << err_l1_l2_linf[0] << ", " << err_l1_l2_linf[1] << ", " << err_l1_l2_linf[2] << std::endl;

	return 0;
}