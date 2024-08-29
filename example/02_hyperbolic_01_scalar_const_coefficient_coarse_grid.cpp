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

// this example shows sparse (and full) grid (no adaptivity) for scalar hyperbolic equation with periodic boundary condition
// 
// different mesh levels in different stages
// 
// example:
// ./02_hyperbolic_01_scalar_const_coefficient_coarse_grid -N 9 -N1 8 -N2 9 -tf 1.0 -filter 1.0 -cfl 0.2
// 
// -N: maximum mesh level
// -N1: maximum mesh level in stage 1
// -N2: maximum mesh level in stage 2
// -tf: final time
// -filter: filter coefficient
// -cfl: CFL number
// 
// You can also change DIM and AlptBasis::PMAX

int main(int argc, char *argv[])
{	
	// --------------------------------------------------------------------------------------------
	// --- Part 1: preliminary part	---
	// static variables
	const int DIM = 2;

	AlptBasis::PMAX = 1;

	LagrBasis::PMAX = 5;
	LagrBasis::msh_case = 1;

	HermBasis::PMAX = 3;
	HermBasis::msh_case = 1;

	Element::PMAX_alpt = AlptBasis::PMAX;	// max polynomial degree for Alpert's basis functions
	Element::PMAX_intp = LagrBasis::PMAX;	// max polynomial degree for interpolation basis functions
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
	double final_time = 10.0;
	double cfl = 0.1;
	const bool is_adapt_find_ptr_alpt = true;	// variable control if need to adaptively find out pointers related to Alpert basis in DG operators
	const bool is_adapt_find_ptr_intp = false;	// variable control if need to adaptively find out pointers related to interpolation basis in DG operators

	// adaptive parameter
	// if need to test code without adaptive, just set refine_eps large number 1e6, then no refine
	// and set coarsen_eta negative number -1, then no coarsen
	const double refine_eps = 1e10;
	// const double coarsen_eta = refine_eps/10.;
	const double coarsen_eta = -1;

	int NMAX_coarse_grid_stage_1 = NMAX;
	int NMAX_coarse_grid_stage_2 = NMAX;

	// filter coefficient
	double filter_coef = 1.0;

	OptionsParser args(argc, argv);
	args.AddOption(&NMAX, "-N", "--max-mesh-level", "Maximum mesh level");
	args.AddOption(&is_sparse, "-s", "--sparse-grid", "sparse grid (1) or full grid (0)");
	args.AddOption(&final_time, "-tf", "--final-time", "Final time; start time is 0.");
	args.AddOption(&cfl, "-cfl", "--cfl-number", "CFL number");
	args.AddOption(&NMAX_coarse_grid_stage_1, "-N1", "--max-mesh-stage-1", "Maximum mesh level in stage 1");
	args.AddOption(&NMAX_coarse_grid_stage_2, "-N2", "--max-mesh-stage-2", "Maximum mesh level in stage 2");
	args.AddOption(&filter_coef, "-filter", "--filter-coefficient", "Filter coefficient");
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

	AllBasis<AlptBasis> all_bas_alpt(NMAX);
	AllBasis<LagrBasis> all_bas_lagr(NMAX);
	AllBasis<HermBasis> all_bas_herm(NMAX);	

	// operator matrix
	OperatorMatrix1D<AlptBasis,AlptBasis> oper_matx(all_bas_alpt, all_bas_alpt, boundary_type);
	OperatorMatrix1D<HermBasis, HermBasis> oper_matx_herm_herm(all_bas_herm, all_bas_herm, boundary_type);
	OperatorMatrix1D<LagrBasis, LagrBasis> oper_matx_lagr_lagr(all_bas_lagr, all_bas_lagr, boundary_type);

	// --- End of Part 1 ---
	// --------------------------------------------------------------------------------------------




	// --------------------------------------------------------------------------------------------
	// --- Part 2: initialization of DG solution ---
	DGAdaptIntp dg_solu(sparse, N_init, NMAX, all_bas_alpt, all_bas_lagr, all_bas_herm, hash, refine_eps, coarsen_eta, is_adapt_find_ptr_alpt, is_adapt_find_ptr_intp, oper_matx_lagr_lagr, oper_matx_herm_herm);

	// initial condition
	// u(x,0) = cos(2 * pi * (sum_(d=1)^DIM x_d)
	auto init_non_separable_func = [=](std::vector<double> x, int i)
	{	
		double sum_x = 0.;
		for (int d = 0; d < DIM; d++) { sum_x += x[d]; };

		return cos(2*Const::PI*sum_x);
	};

	dg_solu.init_adaptive_intp(init_non_separable_func);

	// // project initial function into numerical solution
	// // f(x,y) = cos(2*pi*(x+y)) = cos(2*pi*x) * cos(2*pi*y) - sin(2*pi*x) * sin(2*pi*y)
	// auto init_func_1 = [](double x, int d) { return (d==0) ? (cos(2.*Const::PI*x)) : (cos(2.*Const::PI*x)); };
	// auto init_func_2 = [](double x, int d) { return (d==0) ? (-sin(2.*Const::PI*x)) : (sin(2.*Const::PI*x)); };
	// std::vector<std::function<double(double, int)>> init_func{init_func_1, init_func_2};	
	// dg_solu.init_separable_scalar_sum(init_func);

	// --- End of Part 2 ---
	// --------------------------------------------------------------------------------------------




	// --------------------------------------------------------------------------------------------
	// --- Part 3: time evolution ---
	// coefficients in the equation are all 1:
	// u_t + \sum_(d=1)^DIM u_(x_d) = 0
	const std::vector<double> hyperbolicConst(DIM, 1.);

	const int max_mesh = dg_solu.max_mesh_level();
	const double dx = 1./pow(2., max_mesh);
	double dt = dx * cfl / DIM;
	int total_time_step = ceil(final_time/dt) + 1;
	dt = final_time/total_time_step;

	HyperbolicAlpt dg_operator(dg_solu, oper_matx);
	dg_operator.assemble_matrix_scalar(hyperbolicConst);
	
	// std::cout << Eigen::MatrixXd(dg_operator.mat) << std::endl;
	// exit(1);

	// DG operator for coarser grid in stage 1
	HyperbolicAlpt dg_operator_coarse_grid_stage_1(dg_solu, oper_matx);	
	dg_operator_coarse_grid_stage_1.assemble_matrix_scalar_coarse_grid(hyperbolicConst, NMAX_coarse_grid_stage_1);

	// DG operator for coarser grid in stage 2
	HyperbolicAlpt dg_operator_coarse_grid_stage_2(dg_solu, oper_matx);
	dg_operator_coarse_grid_stage_2.assemble_matrix_scalar_coarse_grid(hyperbolicConst, NMAX_coarse_grid_stage_2);

	// std::cout << "NMAX stage 1: " << NMAX_coarse_grid_stage_1 << std::endl;
	// std::cout << "NMAX stage 2: " << NMAX_coarse_grid_stage_2 << std::endl;

	RK2Midpoint odesolver(dg_operator, dt);
	// RK3HeunLinear odesolver(dg_operator, dt);
	odesolver.init();

	// compute L2 norm of numerical solution at initial time
	std::vector<double> solu_l2_norm_init = dg_solu.get_L2_norm();

	std::cout << "--- evolution started ---" << std::endl;
	// record code running time
	Timer record_time;
	double curr_time = 0;
	int is_stable = 1;
	for (size_t num_time_step = 0; num_time_step < total_time_step; num_time_step++)
	{	
		odesolver.init();
		
		// stage 1
		odesolver.set_rhs_zero();
		odesolver.add_rhs_matrix(dg_operator_coarse_grid_stage_1);
		odesolver.step_stage(0);

		// stage 2
		odesolver.set_rhs_zero();
		odesolver.add_rhs_matrix(dg_operator_coarse_grid_stage_2);
		odesolver.step_stage(1);

		// // stage 3
		// odesolver.set_rhs_zero();
		// odesolver.add_rhs_matrix(dg_operator);
		// odesolver.step_stage(2);

		odesolver.final();

		dg_solu.filter(filter_coef, hyperbolicConst, dt, NMAX_coarse_grid_stage_1 + 1);

		curr_time += dt;
		
		// record code running time
		if (num_time_step % 10000 == 0)
		{		
			// compute L2 norm of numerical solution
			std::vector<double> solu_l2_norm = dg_solu.get_L2_norm();

			if ((solu_l2_norm[0] > 100.0 * solu_l2_norm_init[0]) || std::isnan(solu_l2_norm[0]) || std::isinf(solu_l2_norm[0]))
			{ 
				std::cout << "L2 norm is too large: " << solu_l2_norm[0] << std::endl;
				is_stable = 0;
				break;
			}

			std::cout << "num of time steps: " << num_time_step 
					<< "; time step size: " << dt 
					<< "; curr time: " << curr_time
					<< "; L2 norm: " << solu_l2_norm[0]
					<< "; DoF: " << dg_solu.get_dof()
					<< std::endl;
			record_time.time("elasped time in evolution");
		}
	}	
	// odesolver.final();
	
	std::cout << "--- evolution finished ---" << std::endl;
	// --- End of Part 3 ---
	// --------------------------------------------------------------------------------------------




	// --------------------------------------------------------------------------------------------
	// --- Part 4: calculate error between numerical solution and exact solution ---
	std::cout << "calculating error at final time" << std::endl;
	record_time.reset();

	// compute the error using adaptive interpolation
	// construct anther DGsolution v_h and use adaptive Lagrange interpolation to approximate the exact solution
	const double refine_eps_ext = 1e-6;
	const double coarsen_eta_ext = -1; 
	
	DGAdaptIntp dg_solu_ext(sparse, N_init, NMAX, all_bas_alpt, all_bas_lagr, all_bas_herm, hash, refine_eps_ext, coarsen_eta_ext, is_adapt_find_ptr_alpt, is_adapt_find_ptr_intp, oper_matx_lagr_lagr, oper_matx_herm_herm);

	auto final_func = [=](std::vector<double> x, int i) 
	{	
		double sum_x = 0.;
		for (int d = 0; d < DIM; d++) { sum_x += x[d]; };
		return cos(2.*Const::PI*(sum_x - DIM * final_time)); 
	};
	dg_solu_ext.init_adaptive_intp(final_func);

	// compute L2 error between u_h (numerical solution) and v_h (interpolation to exact solution)
	double err_l2 = dg_solu_ext.get_L2_error_split_adaptive_intp_scalar(dg_solu);		
	std::cout << "L2 error at final time: " << err_l2 << std::endl;	
	record_time.time("elasped time in computing error");

	// auto final_function = [&](std::vector<double> x) -> double 
	// {	
	// 	double sum_x = 0.;
	// 	for (int d = 0; d < DIM; d++) { sum_x += x[d]; };
	// 	return cos(2*Const::PI*(sum_x - DIM*final_time));
	// };
	// std::vector<double> err = dg_solu.get_error_no_separable_scalar(final_function, 4);
	// std::cout << std::scientific << std::setprecision(10);
	// std::cout << "L1, L2 and Linf error at final time: " << err[0] << ", " << err[1] << ", " << err[2] << std::endl;
	
	// --- End of Part 4 ---
	// --------------------------------------------------------------------------------------------

	// --------------------------------------------------------------------------------------------
	// --- Part 5: output results into file ---
	
	// output error
	if (std::isnan(err_l2) || std::isinf(err_l2))
	{
		is_stable = 0; err_l2 = -1;
	}
	std::string output_file_name = 
		"result_dim_" + std::to_string(DIM) 
		+ "_N1_" + std::to_string(NMAX_coarse_grid_stage_1)
		+ "_N2_" + std::to_string(NMAX_coarse_grid_stage_2)
		+ "_cfl_" + std::to_string(cfl)		
		+ "_filter_" + std::to_string(filter_coef)
		+ "_tf_" + std::to_string(final_time)
		+ ".txt";
	std::ofstream output_file(output_file_name);
	output_file << "NMAX in stage 1: " << std::endl << NMAX_coarse_grid_stage_1 << std::endl
				<< "NMAX in stage 2: " << std::endl << NMAX_coarse_grid_stage_2 << std::endl
				<< "CFL: " << std::endl << cfl << std::endl
				<< "Final time: " << std::endl << final_time << std::endl
				<< "Filter coefficient: " << std::endl << filter_coef << std::endl
				<< "Stable (1) or not (0): " << std::endl << is_stable << std::endl
				<< "L2 error at final time: " << std::endl
				<< err_l2 << std::endl;
	output_file.close();

	// --- End of Part 5 ---
	// --------------------------------------------------------------------------------------------
	return 0;
}