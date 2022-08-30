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

// this example shows sparse (and full) grid for scalar hyperbolic equation with periodic boundary condition
// there is no adaptivity in this example
// 
// example:
// ./02_hyperbolic_01_scalar_const_coefficient -N 7 -s 1 -tf 0.2
// 
// -N: maximum mesh level
// -s: sparse grid (1) or full grid (0)
// -tf: final time
// 
// You can also change DIM and AlptBasis::PMAX

// initial condition given in the form of summation of some separable functions
std::vector<std::function<double(double, int)>> init_condition(int DIM);

int main(int argc, char *argv[])
{	
	// --------------------------------------------------------------------------------------------
	// --- Part 1: preliminary part	---
	// static variables
	const int DIM = 3;
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
	dg_solu.init_separable_scalar_sum(init_condition(real_dim));

	// --- End of Part 2 ---
	// --------------------------------------------------------------------------------------------




	// --------------------------------------------------------------------------------------------
	// --- Part 3: time evolution ---
	// coefficients in the equation are all 1:
	// u_t + \sum_(d=1)^DIM u_(x_d) = 0
	const std::vector<double> hyperbolicConst(DIM, 1.);

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
	Timer record_time;
	double curr_time = 0;
	for (size_t num_time_step = 0; num_time_step < total_time_step; num_time_step++)
	{	
		odesolver.step_rk();
		curr_time += dt;
		
		// record code running time
		if (num_time_step % 10 == 0)
		{		
			std::cout << "num of time steps: " << num_time_step 
					<< "; time step size: " << dt 
					<< "; curr time: " << curr_time
					<< "; DoF: " << dg_solu.get_dof()
					<< std::endl;
			record_time.time("elasped time in evolution");
		}
	}	
	odesolver.final();
	
	std::cout << "--- evolution finished ---" << std::endl;
	// --- End of Part 3 ---
	// --------------------------------------------------------------------------------------------




	// --------------------------------------------------------------------------------------------
	// --- Part 4: calculate error between numerical solution and exact solution ---
	std::cout << "calculating error at final time" << std::endl;
	record_time.reset();

	// if (DIM <= 2)
	// for low dimension, d = 1, 2, compute the error using Gaussian quadrature in the each finest element
	// computational cost increases exponentially w.r.t. dimension
	{	
		const int num_gauss_pt = 3;
		// exact solution at final time: u(x_1, x_2, ..., x_d, t) = cos(2 * PI * ( sum_(d=1)^DIM (x_d) - DIM * t))
		auto final_func = [=](std::vector<double> x) 
		{	
			double sum_x = 0.;
			for (int d = 0; d < real_dim; d++) { sum_x += x[d]; };
			return cos(2.*Const::PI*(sum_x - real_dim * final_time)); 
		};
		std::vector<double> err_l1_l2_linf = dg_solu.get_error_no_separable_scalar(final_func, num_gauss_pt);
		std::cout << "L1, L2 and Linf error at final time: " << err_l1_l2_linf[0] << ", " << err_l1_l2_linf[1] << ", " << err_l1_l2_linf[2] << std::endl;
	}
	// --- End of Part 4 ---
	// --------------------------------------------------------------------------------------------

	return 0;
}






// initial condition in the summation of separable formulation
std::vector<std::function<double(double, int)>> init_condition(int DIM)
{
	if (DIM == 2)
	// init condition = cos(2*PI*(x0 + x1))
	{
		// func_1 = cos(2*PI*x0) * cos(2*PI*x1)
		std::function<double(double, int)> init_func_1 = [](double x, int d) 
		{
			if (d == 0) { return cos(2.*Const::PI*x); }
			else if (d == 1) { return cos(2.*Const::PI*x); }
			else { return 1.; }
		};
		// func_2 = -sin(2*PI*x0) * sin(2*PI*x1)
		std::function<double(double, int)> init_func_2 = [](double x, int d) 
		{
			if (d == 0) { return -sin(2.*Const::PI*x); }
			else if (d == 1) { return sin(2.*Const::PI*x); }
			else { return 1.; }
		};
		
		// init condition = cos(2*PI*(x0 + x1)) 
		return std::vector<std::function<double(double, int)>>{ init_func_1, init_func_2 };
	}
}