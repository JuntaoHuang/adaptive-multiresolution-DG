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

std::vector<std::function<double(double, int)>> init_condition(int DIM);

int main(int argc, char *argv[])
{	
	// --------------------------------------------------------------------------------------------
	// --- Part 1: preliminary part	---
	// static variables
	const int DIM = 3;

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
	int is_sparse = 1;
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
	DGAdapt dg_solu(sparse, N_init, NMAX, all_bas, all_bas_Lag, all_bas_Her, hash, refine_eps, coarsen_eta, is_adapt_find_ptr_alpt, is_adapt_find_ptr_intp);

	dg_solu.init_separable_scalar_sum(init_condition(DIM));

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

	if (DIM <= 2)
	// for low dimension, d = 1, 2, compute the error using Gaussian quadrature in the each finest element
	// computational cost increases exponentially w.r.t. dimension
	{	
		const int num_gauss_pt = 3;
		// exact solution at final time: u(x_1, x_2, ..., x_d, t) = cos(2 * PI * ( sum_(d=1)^DIM (x_d) - DIM * t))
		auto final_func = [=](std::vector<double> x) 
		{	
			double sum_x = 0.;
			for (int d = 0; d < DIM; d++) { sum_x += x[d]; };
			return cos(2.*Const::PI*(sum_x - DIM * final_time)); 
		};
		std::vector<double> err_l1_l2_linf = dg_solu.get_error_no_separable_scalar(final_func, num_gauss_pt);
		std::cout << "L1, L2 and Linf error at final time: " << err_l1_l2_linf[0] << ", " << err_l1_l2_linf[1] << ", " << err_l1_l2_linf[2] << std::endl;
	}
	else
	// for higher dimension, d >= 3, compute the error using adaptive interpolation
	{
		// --- step 1: construct anther DGsolution v_h and use adaptive Lagrange interpolation to approximate the exact solution
		const double refine_eps_ext = 1e-6;
		const double coarsen_eta_ext = -1; 
		OperatorMatrix1D<HermBasis, HermBasis> oper_matx_herm_herm(all_bas_Her, all_bas_Her, boundary_type);
		OperatorMatrix1D<LagrBasis, LagrBasis> oper_matx_lagr_lagr(all_bas_Lag, all_bas_Lag, boundary_type);
		
		DGAdaptIntp dg_solu_ext(sparse, N_init, NMAX, all_bas, all_bas_Lag, all_bas_Her, hash, refine_eps_ext, coarsen_eta_ext, is_adapt_find_ptr_alpt, is_adapt_find_ptr_intp, oper_matx_lagr_lagr, oper_matx_herm_herm);
		LagrInterpolation interp_ext(dg_solu_ext);

		auto final_func = [=](std::vector<double> x, int i) 
		{	
			double sum_x = 0.;
			for (int d = 0; d < DIM; d++) { sum_x += x[d]; };
			return cos(2.*Const::PI*(sum_x - DIM * final_time)); 
		};
		dg_solu_ext.init_adaptive_intp_Lag(final_func, interp_ext);

		// --- step 2: transformation coefficient of Lagrange basis to Alpert basis
		OperatorMatrix1D<LagrBasis, AlptBasis> oper_matx_lagr_alpt(all_bas_Lag, all_bas, boundary_type);;
		FastLagrInit fastLagr_init_ext(dg_solu_ext, oper_matx_lagr_alpt);
		fastLagr_init_ext.eval_ucoe_Alpt_Lagr();

		// --- step 3: compute L2 error between u_h (numerical solution) and v_h (interpolation to exact solution)
		double err_l2 = dg_solu_ext.get_L2_error_split_adaptive_intp_scalar(dg_solu);		
		std::cout << "L2 error at final time: " << err_l2 << std::endl;	
	}
	record_time.time("elasped time in computing error");

	// --- End of Part 4 ---
	// --------------------------------------------------------------------------------------------

	return 0;
}






// initial condition in the summation of separable formulation
std::vector<std::function<double(double, int)>> init_condition(int DIM)
{
	assert(DIM >=2 && DIM <=4);

	if (DIM == 2)
	// dim = 2
	// init condition = cos(2*PI*(x0 + x1))
	{
		// func_1 = cos(2*PI*x0) * cos(2*PI*x1)
		std::function<double(double, int)> init_func_1 = [](double x, int d) 
		{
			if (d == 0) { return cos(2.*Const::PI*x); }
			else { return cos(2.*Const::PI*x); }
		};
		// func_2 = -sin(2*PI*x0) * sin(2*PI*x1)
		std::function<double(double, int)> init_func_2 = [](double x, int d) 
		{
			if (d == 0) { return -sin(2.*Const::PI*x); }
			else { return sin(2.*Const::PI*x); }
		};
		
		// init condition = cos(2*PI*(x0 + x1)) 
		return std::vector<std::function<double(double, int)>>{ init_func_1, init_func_2 };
	}
	else if (DIM == 3)
	// dim = 3
	// init condition = cos(2*PI*(x0 + x1 + x2))
	{
		// func_1 = cos(2*PI*x0) * cos(2*PI*x1) * cos(2*PI*x2)
		std::function<double(double, int)> init_func_1 = [](double x, int d) 
		{
			if (d == 0) { return cos(2.*Const::PI*x); }
			else if (d == 1) { return cos(2.*Const::PI*x); }
			else { return cos(2.*Const::PI*x); }
		};
		// func_2 = -sin(2*PI*x0) * sin(2*PI*x1) * cos(2*PI*x2)
		std::function<double(double, int)> init_func_2 = [](double x, int d) 
		{
			if (d == 0) { return -sin(2.*Const::PI*x); }
			else if (d == 1) { return sin(2.*Const::PI*x); }
			else { return cos(2.*Const::PI*x); }
		};
		// func_3 = -sin(2*PI*x0) * cos(2*PI*x1) * sin(2*PI*x2)
		std::function<double(double, int)> init_func_3 = [](double x, int d) 
		{
			if (d == 0) { return -sin(2.*Const::PI*x); }
			else if (d == 1) { return cos(2.*Const::PI*x); }
			else { return sin(2.*Const::PI*x); }
		};
		// func_4 = -cos(2*PI*x0) * sin(2*PI*x1) * sin(2*PI*x2)
		std::function<double(double, int)> init_func_4 = [](double x, int d) 
		{
			if (d == 0) { return -cos(2.*Const::PI*x); }
			else if (d == 1) { return sin(2.*Const::PI*x); }
			else { return sin(2.*Const::PI*x); }
		};

		// init condition = cos(2*PI*(x0 + x1 + x2)) 
		return std::vector<std::function<double(double, int)>>{ init_func_1, init_func_2, init_func_3, init_func_4};
	}
	else
	// dim = 4
	// init condition = cos(2*PI*(x0 + x1 + x2 + x3)) 
	{
		// func_1 = cos(2*PI*x0) * cos(2*PI*x1) * cos(2*PI*x2) * cos(2*PI*x3)
		std::function<double(double, int)> init_func_1 = [](double x, int d) 
		{
			if (d == 0) { return cos(2.*Const::PI*x); }
			else if (d == 1) { return cos(2.*Const::PI*x); }
			else if (d == 2) { return cos(2.*Const::PI*x); }
			else { return cos(2.*Const::PI*x); }
		};
		// func_2 = - sin(2*PI*x0) * sin(2*PI*x1) * cos(2*PI*x2) * cos(2*PI*x3)
		std::function<double(double, int)> init_func_2 = [](double x, int d) 
		{
			if (d == 0) { return -sin(2.*Const::PI*x); }
			else if (d == 1) { return sin(2.*Const::PI*x); }
			else if (d == 2) { return cos(2.*Const::PI*x); }
			else { return cos(2.*Const::PI*x); }
		};
		// func_3 = - sin(2*PI*x0) * cos(2*PI*x1) * sin(2*PI*x2) * cos(2*PI*x2)
		std::function<double(double, int)> init_func_3 = [](double x, int d) 
		{
			if (d == 0) { return -sin(2.*Const::PI*x); }
			else if (d == 1) { return cos(2.*Const::PI*x); }
			else if (d == 2) { return sin(2.*Const::PI*x); }
			else { return cos(2.*Const::PI*x); }
		};
		// func_4 = - cos(2*PI*x0) * sin(2*PI*x1) * sin(2*PI*x2) * cos(2*PI*x3)
		std::function<double(double, int)> init_func_4 = [](double x, int d) 
		{
			if (d == 0) { return -cos(2.*Const::PI*x); }
			else if (d == 1) { return sin(2.*Const::PI*x); }
			else if (d == 2) { return sin(2.*Const::PI*x); }
			else { return cos(2.*Const::PI*x); }
		};
		// func_5 = - sin(2*PI*x0) * cos(2*PI*x1) * cos(2*PI*x2) * sin(2*PI*x3)
		std::function<double(double, int)> init_func_5 = [](double x, int d) 
		{
			if (d == 0) { return -sin(2.*Const::PI*x); }
			else if (d == 1) { return cos(2.*Const::PI*x); }
			else if (d == 2) { return cos(2.*Const::PI*x); }
			else { return sin(2.*Const::PI*x); }
		};
		// func_6 = - cos(2*PI*x0) * sin(2*PI*x1) * cos(2*PI*x2) * sin(2*PI*x3)
		std::function<double(double, int)> init_func_6 = [](double x, int d) 
		{
			if (d == 0) { return -cos(2.*Const::PI*x); }
			else if (d == 1) { return sin(2.*Const::PI*x); }
			else if (d == 2) { return cos(2.*Const::PI*x); }
			else { return sin(2.*Const::PI*x); }
		};
		// func_7 = - cos(2*PI*x0) * cos(2*PI*x1) * sin(2*PI*x2) * sin(2*PI*x2)
		std::function<double(double, int)> init_func_7 = [](double x, int d) 
		{
			if (d == 0) { return -cos(2.*Const::PI*x); }
			else if (d == 1) { return cos(2.*Const::PI*x); }
			else if (d == 2) { return sin(2.*Const::PI*x); }
			else { return sin(2.*Const::PI*x); }
		};
		// func_8 = sin(2*PI*x0) * sin(2*PI*x1) * sin(2*PI*x2) * sin(2*PI*x3)
		std::function<double(double, int)> init_func_8 = [](double x, int d) 
		{
			if (d == 0) { return sin(2.*Const::PI*x); }
			else if (d == 1) { return sin(2.*Const::PI*x); }
			else if (d == 2) { return sin(2.*Const::PI*x); }
			else { return sin(2.*Const::PI*x); }
		};

		// init condition = cos(2*PI*(x0 + x1 + x2 + x3)) 
		return std::vector<std::function<double(double, int)>>{ init_func_1, init_func_2, init_func_3, init_func_4, init_func_5, init_func_6, init_func_7, init_func_8 };
	}

}