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

// ------------------------------------------------------------------------------------------------
// solve 2D wave equation
// u_tt = u_xx + u_yy
// with exact solution: u(x,y,t) = sin(2*\sqrt(2)*pi*t) * cos(2*pi*x) * cos(2*pi*y)
// and periodic boundary condition
// 
// solve 3D wave equation
// u_tt = u_xx + u_yy + u_zz
// with exact solution: u(x,y,z,t) = sin(2*\sqrt(3)*pi*t) * cos(2*pi*x) * cos(2*pi*y) * cos(2*pi*z)
// and periodic boundary condition
// ------------------------------------------------------------------------------------------------

int main(int argc, char *argv[])
{
	// static variables
	AlptBasis::PMAX = 3;

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
	int NMAX = 3;	
	const bool sparse = true;
	const std::string boundary_type = "period";
	double final_time = 0.1;
	double cfl = 0.1;	
	const bool is_adapt_find_ptr_alpt = true;	// variable control if need to adaptively find out pointers related to Alpert basis in DG operators
	const bool is_adapt_find_ptr_intp = false;	// variable control if need to adaptively find out pointers related to interpolation basis in DG operators

	// adaptive parameter
	// if need to test code without adaptive, just set refine_eps large number 1e6, then no refine
	// and set coarsen_eta negative number -1, then no coarsen
	double refine_eps = 1e10;	
	double coarsen_eta = -1;

	OptionsParser args(argc, argv);
	args.AddOption(&NMAX, "-N", "--max-mesh-level", "Maximum mesh level");
	args.AddOption(&refine_eps, "-r", "--refine-epsilon", "refine parameter epsilon");
	args.AddOption(&coarsen_eta, "-c", "--coarsen-eta", "coarsen parameter eta");
	args.AddOption(&cfl, "-cfl", "--cfl-number", "CFL number");	
	args.AddOption(&final_time, "-tf", "--final-time", "Final time; start time is 0.");	

	args.Parse();
	if (!args.Good())
	{
		args.PrintUsage(std::cout);
		return 1;
	}
	args.PrintOptions(std::cout);		
	
	int N_init = NMAX;

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
	auto init_func = [&](double x, int d)->double											
		{
			if (d==0) { return 2.*sqrt(DIM)*Const::PI * cos(2. * Const::PI * x); }
			else { return cos(2. * Const::PI * x); }
		};
					
	// initialization of DG solution
	DGAdapt dg_solu(sparse, N_init, NMAX, all_bas, all_bas_Lag, all_bas_Her, hash, refine_eps, coarsen_eta, is_adapt_find_ptr_alpt, is_adapt_find_ptr_intp);

	// project initial function into numerical solution
	dg_solu.init_separable_scalar(init_func);
	dg_solu.copy_ucoe_to_ucoe_ut();
	dg_solu.set_ucoe_alpt_zero();

	// ------------------------------
	// 	This part is for solving constant coefficient for linear equation	
	const std::vector<double> waveConst(DIM, 1.);
	double sigma_ipdg = 0.;
	if (DIM==2) { sigma_ipdg = 10.; }
	else if (DIM==3) { sigma_ipdg = 20.; }

	const int max_mesh = dg_solu.max_mesh_level();
	const double dx = 1./pow(2., max_mesh);
	double dt = dx * cfl;
	int total_time_step = ceil(final_time/dt) + 1;
	dt = final_time/total_time_step;

	DiffusionAlpt dg_operator_evolution(dg_solu, oper_matx, sigma_ipdg);
	dg_operator_evolution.assemble_matrix_scalar(waveConst);

	RK4ODE2nd ode_solver(dg_operator_evolution, dt);
	ode_solver.init();

	std::cout << "--- evolution started ---" << std::endl;
	// record code running time
	auto start_evolution_time = std::chrono::high_resolution_clock::now();	

	double curr_time = 0;
	for (size_t num_time_step = 0; num_time_step < total_time_step; num_time_step++)
	{
		ode_solver.step_rk();
			
		curr_time += dt;		
	}
	ode_solver.final();

	// record code running time
	auto stop_evolution_time = std::chrono::high_resolution_clock::now(); 
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop_evolution_time - start_evolution_time);
	std::cout << "running time: " << duration.count()/1e6 << " seconds" << std::endl;

	std::cout << "--- evolution finished ---" << std::endl;

	// calculate error between numerical solution and exact solution
	std::cout << "calculating error at final time" << std::endl;
	auto final_func = [&](double x, int d)->double 
		{
			if (d==0) { return sin(2.*sqrt(DIM)*Const::PI*final_time) * cos(2. * Const::PI * x); }
			else { return cos(2. * Const::PI * x); }						
		};
	
	// const int num_gauss_pt = 5;
	// std::vector<double> err = dg_solu.get_error_separable_scalar(final_func, num_gauss_pt);
	// std::cout << "L1, L2 and Linf error at final time: " << err[0] << ", " << err[1] << ", " << err[2] << std::endl;
	
	Quad quad_exact(DIM);
	auto final_exact_sol = [&](std::vector<double> x)->double 
		{
			double func = 1.;
			for (size_t d = 0; d < DIM; d++) { func *= final_func(x[d], d); }
			return func; 
		};
	const double l2_norm_exact = quad_exact.norm_multiD(final_exact_sol, 4, 10)[1];
	double err2 = dg_solu.get_L2_error_split_separable_scalar(final_func, l2_norm_exact);
	std::cout << "L2 error at final time with split: " << err2 << std::endl;

	std::vector<double> err(3, err2);
	IO inout(dg_solu);
	std::string file_name = "error_N" + std::to_string(NMAX) + ".txt";
	inout.write_error(NMAX, err, file_name);	
	// ------------------------------

	return 0;
}