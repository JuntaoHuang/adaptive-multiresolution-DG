#include "AllBasis.h"
#include "AlptBasis.h"
#include "Basis.h"
#include "BilinearForm.h"
#include "DGAdapt.h"
#include "DGSolution.h"
#include "DGAdaptIntp.h"
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
// 1D wave equation
// u_tt = u_xx
// u(x,0) = f(x)
// u_t(x,0) = g(x)
// has the solution
// u(x,t) = 1/2 * (f(x+t) + f(x-t)) + 1/2 * \int^(x+t)_(x-t) g(s)ds
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
	Element::DIM = 3;			// dimension
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
	int NMAX = 7;
	int N_init = 2;
	const bool sparse = false;
	double final_time = 1.0;
	double cfl = 0.1;
	const bool is_adapt_find_ptr_alpt = true;	// variable control if need to adaptively find out pointers related to Alpert basis in DG operators
	const bool is_adapt_find_ptr_intp = false;	// variable control if need to adaptively find out pointers related to interpolation basis in DG operators
	const int output_step = 10;
	std::vector<double> output_time = linspace(0.01, 0.50, 50);

	// adaptive parameter
	// if need to test code without adaptive, just set refine_eps large number 1e6, then no refine
	// and set coarsen_eta negative number -1, then no coarsen
	double refine_eps = 1e-4;	
	double coarsen_eta = 1e-5;

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

	AllBasis<AlptBasis> all_bas_alpt(NMAX);

	// operator matrix for Alpert basis
	OperatorMatrix1D<AlptBasis,AlptBasis> oper_matx(all_bas_alpt, all_bas_alpt, "inside");
	OperatorMatrix1D<LagrBasis,AlptBasis> oper_matx_lagr(all_bas_Lag, all_bas_alpt, "period");
	OperatorMatrix1D<LagrBasis, LagrBasis> oper_matx_lagr_all(all_bas_Lag, all_bas_Lag, "period");
	OperatorMatrix1D<HermBasis, HermBasis> oper_matx_herm_all(all_bas_Her, all_bas_Her, "period");

	// initialization of DG solution
	DGAdaptIntp dg_solu(sparse, N_init, NMAX, all_bas_alpt, all_bas_Lag, all_bas_Her, hash, refine_eps, coarsen_eta, is_adapt_find_ptr_alpt, is_adapt_find_ptr_intp, oper_matx_lagr_all, oper_matx_herm_all);
	
	// ------------------------------
	// 	This part is for solving constant coefficient for linear equation	
	const double wave_speed = 1.;
	const std::vector<double> waveConst(DIM, std::pow(wave_speed, 2.));
	const double sigma_ipdg = 20.;

	// initial condition
	auto init_func = [&](std::vector<double> x, int i)->double 
		{
			double r2 = 0;
			for (size_t d = 0; d < DIM; d++) { r2 += std::pow(x[d], 2.); }
			return 100 * exp(-500*r2);
		};
	LagrInterpolation interp(dg_solu);
	dg_solu.init_adaptive_intp_Lag(init_func, interp);
	FastLagrInit fast_init(dg_solu, oper_matx_lagr);
	fast_init.eval_ucoe_Alpt_Lagr();

	dg_solu.copy_ucoe_to_ucoe_ut();
	dg_solu.set_ucoe_alpt_zero();
	
	std::cout << "--- evolution started ---" << std::endl;
	// record code running time
	Timer record_time;
	IO inout(dg_solu);
	double curr_time = 0.;
	int num_time_step = 0;
	auto start_evolution_time = std::chrono::high_resolution_clock::now();	

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

		curr_time += dt;

		// --- part 5: coarsen
		dg_solu.coarsen();
		const int num_basis_coarsen = dg_solu.size_basis_alpt();	

		// record code running time
		if (num_time_step % output_step == 0)
		{
			record_time.time("running time");
			std::cout << "num of time steps: " << num_time_step << "; time step: " << dt << "; curr time: " << curr_time << std::endl
					<< "curr max mesh: " << dg_solu.max_mesh_level() << std::endl
					<< "num of basis after refine: " << num_basis_refine << "; num of basis after coarsen: " << num_basis_coarsen << std::endl << std::endl;		
		}

		auto min_time_iter = std::min_element(output_time.begin(), output_time.end());
		if ((curr_time <= *min_time_iter) && (curr_time+dt >= *min_time_iter))
		{
			auto exact_1D = [&](std::vector<double> x)->double 
				{
					double r = std::abs(x[0]);
					return 1./2.*std::sqrt(5.*Const::PI)*(std::erf(10*std::sqrt(5)*(curr_time-r)) + std::erf(10*std::sqrt(5)*(curr_time+r)));
				};

			std::string file_name = "profile2D_" + std::to_string(curr_time) + ".txt";
			if (DIM==1) { inout.output_num_exa(file_name, exact_1D); }
			else if (DIM==2) { inout.output_num(file_name); }
			else if (DIM==3) { inout.output_num_cut_2D(file_name, 0.0, 2); }

			if (DIM==1)
			{
				std::string file_name = "suppt1D_" + std::to_string(curr_time) + ".txt";
				inout.output_element_level_support(file_name);
			}

			std::string file_name_center = "center2D_" + std::to_string(curr_time) + ".txt";
			inout.output_element_center(file_name_center);

			output_time.erase(min_time_iter);
		}		

		// add current time and increase time steps		
		num_time_step ++;
	}

	std::cout << "--- evolution finished ---" << std::endl;

	// ------------------------------

	return 0;
}