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
#include "Timer.h"
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

	DGSolution::prob = "wave";
	DGSolution::ind_var_vec = { 0 };
	DGAdapt::indicator_var_adapt = { 0 };

	Element::is_intp.resize(Element::VEC_NUM);
	for (size_t num = 0; num < Element::VEC_NUM; num++) { Element::is_intp[num] = std::vector<bool>(Element::DIM, true); }

	// constant variable
	const int DIM = Element::DIM;
	int NMAX = 3;	
	int N_init = NMAX;
	const bool sparse = false;
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
	args.AddOption(&NMAX, "-NM", "--max-mesh-level", "Maximum mesh level");
	args.AddOption(&N_init, "-N0", "--initial-mesh-level", "Mesh level in initialization");	
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

	DiffusionAlpt dg_operator_evolution(dg_solu, oper_matx, sigma_ipdg);
	dg_operator_evolution.assemble_matrix_scalar(waveConst);

	std::cout << "--- evolution started ---" << std::endl;
	
	// record code running time
	Timer record_time;
	double curr_time = 0.;
	int num_time_step = 0;
	while ( curr_time < final_time )
	{			
		// --- part 1: calculate time step dt ---	
		const std::vector<int> & max_mesh = dg_solu.max_mesh_level_vec();
		
		// dt = cfl/(c1/dx1 + c2/dx2 + ... + c_dim/dx_dim)
		double sum_c_dx = 0.;	// this variable stores (c1/dx1 + c2/dx2 + ... + c_dim/dx_dim)
		for (size_t d = 0; d < DIM; d++)
		{
			sum_c_dx += std::abs(waveConst[d]) * std::pow(2., max_mesh[d]);
		}
		double dt = cfl/sum_c_dx;
		dt = std::min(dt, final_time - curr_time);

		// --- part 2: predict by Euler forward
		{	
			// before Euler forward, copy Element::ucoe_alpt to Element::ucoe_alpt_predict
			dg_solu.copy_ucoe_to_predict();
			dg_solu.copy_ucoe_ut_to_predict();

			DiffusionAlpt dg_operator_evolution(dg_solu, oper_matx, sigma_ipdg);
			dg_operator_evolution.assemble_matrix_scalar(waveConst);

			EulerODE2nd odeSolver(dg_operator_evolution, dt);
			odeSolver.init();

			// Euler forward time stepping
			odeSolver.step_rk();

			// copy eigen vector ucoe to Element::ucoe_alpt
			// now prediction of numerical solution at t^(n+1) is stored in Element::ucoe_alpt
			odeSolver.final();			
		}

		// --- part 3: refine base on Element::ucoe_alpt
		dg_solu.refine();
		const int num_basis_refine = dg_solu.size_basis_alpt();

		// after refine, copy Element::ucoe_alpt_predict back to Element::ucoe_alpt
		dg_solu.copy_predict_to_ucoe();
		dg_solu.copy_predict_to_ucoe_ut();

		// --- part 4: time evolution
		DiffusionAlpt dg_operator_evolution(dg_solu, oper_matx, sigma_ipdg);
		dg_operator_evolution.assemble_matrix_scalar(waveConst);

		RK4ODE2nd odeSolver(dg_operator_evolution, dt);
		odeSolver.init();

		odeSolver.step_rk();

		odeSolver.final();

		// --- part 5: coarsen
		dg_solu.coarsen();
		const int num_basis_coarsen = dg_solu.size_basis_alpt();		
		
		// record code running time
		if (num_time_step % 10 == 0)
		{
			record_time.time("running time");
			std::cout << "num of time steps: " << num_time_step << "; time step: " << dt << "; curr time: " << curr_time << std::endl
					<< "curr max mesh: " << dg_solu.max_mesh_level() << std::endl
					<< "num of basis after refine: " << num_basis_refine << "; num of basis after coarsen: " << num_basis_coarsen << std::endl << std::endl;
		}	

		// add current time and increase time steps
		curr_time += dt;
		num_time_step ++;
	}

	std::cout << "--- evolution finished ---" << std::endl;
	
	record_time.time("total running time");
	std::cout << "total num of time steps: " << num_time_step << std::endl
			  << "total num of basis (last time step): " << dg_solu.size_basis_alpt() << std::endl;

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