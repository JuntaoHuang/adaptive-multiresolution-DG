#include "AllBasis.h"
#include "AlptBasis.h"
#include "Basis.h"
#include "BilinearForm.h"
#include "DGAdapt.h"
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

// this shows sparse (and full) grid for scalar hyperbolic equation with periodic boundary condition
// there is no adaptivity in this example
// 
// example:
// ./02_hyperbolic_01_scalar_const_coefficient -N 7 -s 1 -tf 0.2
// 
// -N: maximum mesh level
// -s: sparse grid (1) or full grid (0)
// -tf: final time
// 
// You can also change AlptBasis::PMAX to test DG space with different polynomial degree
int main(int argc, char *argv[])
{
	// static variables
	AlptBasis::PMAX = 2;

	LagrBasis::PMAX = 3;
	LagrBasis::msh_case = 1;

	HermBasis::PMAX = 3;
	HermBasis::msh_case = 1;

	Element::PMAX_alpt = AlptBasis::PMAX;	// max polynomial degree for Alpert's basis functions
	Element::PMAX_intp = HermBasis::PMAX;	// max polynomial degree for interpolation basis functions
	Element::DIM = 2;			// dimension
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
	const int DIM = Element::DIM;
	int NMAX = 4;
	int N_init = NMAX;	
	int is_sparse = 1;
	const std::string boundary_type = "period";
	double final_time = 0.1;
	const double cfl = 0.1;
	const int rk_order = 1;	
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

	// initial condition
	// f(x,y) = exp(sin(2*x)) * exp(cos(2*y))
	//auto init_func = [=](double x, int d) { return (d==0) ? 
	//										exp(sin(2.*Const::PI*x))
	//										: exp(cos(2.*Const::PI*x)); };
	std::function<double(double, int)> init_func_1 = [](double x, int d) { return (d == 0) ? -0.5 / Const::PI*(cos(2.*Const::PI*x)) : cos(2.*Const::PI*x); };
	std::function<double(double, int)> init_func_2 = [](double x, int d) { return (d == 0) ? 0.5 / Const::PI*(sin(2.*Const::PI*x)) : sin(2.*Const::PI*x); };
	std::vector<std::function<double(double, int)>> init_func1{ { init_func_1, init_func_2} };

	std::vector<std::function<double(double, int)>> init_func{ init_func1 };
	// initialization of DG solution
	DGAdapt dg_solu(sparse, N_init, NMAX, all_bas, all_bas_Lag, all_bas_Her, hash, refine_eps, coarsen_eta, is_adapt_find_ptr_alpt, is_adapt_find_ptr_intp);

	// project initial function into numerical solution
	dg_solu.init_separable_scalar_sum(init_func);

	// ------------------------------
	// 	This part is for solving constant coefficient for linear equation	
	const std::vector<double> hyperbolicConst{1.00, 1.00};

	const int max_mesh = dg_solu.max_mesh_level();
	const double dx = 1./pow(2., max_mesh);
	double dt = dx * cfl;
	int total_time_step = ceil(final_time/dt) + 1;
	dt = final_time/total_time_step;

	HyperbolicAlpt dg_operator(dg_solu, oper_matx);
	dg_operator.assemble_matrix_scalar(hyperbolicConst);

	RK3SSP odesolver(dg_operator, dt);
	odesolver.init();

	std::cout << "--- evolution started ---" << std::endl;
	// record code running time
	auto start_evolution_time = std::chrono::high_resolution_clock::now();

	double curr_time = 0;
	for (size_t num_time_step = 0; num_time_step < total_time_step; num_time_step++)
	{	
		odesolver.step_rk();
		curr_time += dt;
		
		// record code running time
		auto stop_evolution_time = std::chrono::high_resolution_clock::now(); 
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop_evolution_time - start_evolution_time);
		std::cout << "num of time steps: " << num_time_step 
				<< "; time step: " << dt 
				<< "; curr time: " << curr_time
				<< "; running time: " << duration.count()/1e6 << " seconds"
				<< std::endl;
	}	
	odesolver.final();
	
	std::cout << "--- evolution finished ---" << std::endl;

	// calculate error between numerical solution and exact solution
	std::cout << "calculating error at final time" << std::endl;

	const int num_gauss_pt = 4;
	//auto final_func = [&](double x, int d)->double { return init_func(x-hyperbolicConst[d]*final_time, d); };
	//
	//std::vector<double> err = dg_solu.get_error_separable_scalar(final_func, num_gauss_pt);
	//std::cout << "L1, L2 and Linf error at final time: " << err[0] << ", " << err[1] << ", " << err[2] << std::endl;
	
	//Quad quad_exact(DIM);
	//auto final_exact_sol = [&](std::vector<double> x)->double { return final_func(x[0],0)*final_func(x[1],1); };
	//const double l2_norm_exact = quad_exact.norm_multiD(final_exact_sol, 2, 10)[1];
	//double err2 = dg_solu.get_L2_error_split_separable_scalar(final_func, l2_norm_exact);
	//std::cout << "L2 error at final time with split: " << err2 << std::endl;

	//auto final_func1 = [=](std::vector<double> x) {return exp(sin(2.*Const::PI*(x[0]-final_time)))*exp(cos(2.*Const::PI*(x[1]-final_time))); };
	
	auto final_func1 = [=](std::vector<double> x) {return  -0.5 / Const::PI*cos(2.*Const::PI*(x[0] + x[1] - 2 * final_time)); };
	std::vector<double> err3 = dg_solu.get_error_no_separable_scalar(final_func1, num_gauss_pt);
	std::cout << "L1, L2 and Linf error at final time: " << err3[0] << ", " << err3[1] << ", " << err3[2] << std::endl;

	IO inout(dg_solu);
	std::string file_name = "error_N" + std::to_string(NMAX) + ".txt";
	inout.write_error(NMAX, err3, file_name);
	// ------------------------------

	return 0;
}
