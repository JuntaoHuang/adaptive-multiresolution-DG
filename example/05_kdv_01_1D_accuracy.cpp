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
// solve 1D KdV equation, accuracy test
// u_t + (u^2/2)_x + u_xxx = s(x,t)
// 
// with exact solution: u(x,t) = sin(2*pi*(x-t)) and periodic boundary condition
// 
// Note: use ultra-weak DG to discretize u_xxx, so take k>=2 (k=0 and k=1 inconsistent)
// ------------------------------------------------------------------------------------------------

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
	Element::DIM = 1;			// dimension
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
	int NMAX = 8;
	int N_init = 2;
	const bool sparse = false;
	const std::string boundary_type = "period";
	double final_time = 0.1;
	double cfl = 0.02;
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
	args.AddOption(&cfl, "-cfl", "--cfl-number", "CFL number");	
	args.AddOption(&final_time, "-tf", "--final-time", "Final time; start time is 0.");
	args.AddOption(&refine_eps, "-r", "--refine-epsilon", "refine parameter epsilon");
	args.AddOption(&coarsen_eta, "-c", "--coarsen-eta", "coarsen parameter eta");

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

	AllBasis<AlptBasis> all_bas_alpt(NMAX);
	AllBasis<LagrBasis> all_bas_lagr(NMAX);
	AllBasis<HermBasis> all_bas_herm(NMAX);

	// operator matrix for Alpert basis
	OperatorMatrix1D<AlptBasis,AlptBasis> oper_matx_alpt(all_bas_alpt, all_bas_alpt, boundary_type);
	OperatorMatrix1D<HermBasis,AlptBasis> oper_matx_herm(all_bas_herm, all_bas_alpt, boundary_type);
	OperatorMatrix1D<LagrBasis,AlptBasis> oper_matx_lagr(all_bas_lagr, all_bas_alpt, boundary_type);

	// physical flux
	auto func_flux = [&](std::vector<double> u, int i, int d)->double { return FluxFunction::burgers_flux_scalar(u[0]); };
	
	// first derivative of physical flux
	auto func_flux_d1 = [&](std::vector<double> u, int i, int d, int i1)->double { return FluxFunction::burgers_flux_1st_derivative_scalar(u[0]); };

	const std::vector<double> lxf_alpha{1.1};

	// initial condition
	auto init_func = [](double x, int d)->double { return sin(2*Const::PI*x); };
					
	// initialization of DG solution
	DGAdapt dg_solu(sparse, N_init, NMAX, all_bas_alpt, all_bas_lagr, all_bas_herm, hash, refine_eps, coarsen_eta, is_adapt_find_ptr_alpt, is_adapt_find_ptr_intp);

	// project initial function into numerical solution
	dg_solu.init_separable_scalar(init_func);

	dg_solu.coarsen();
	std::cout << "total num of basis (initial): " << dg_solu.size_basis_alpt() << std::endl;

	// ------------------------------
	const int max_mesh = dg_solu.max_mesh_level();
	const double dx = 1./pow(2., max_mesh);
	double dt = dx * cfl;
	int total_time_step = ceil(final_time/dt) + 1;
	dt = final_time/total_time_step;

	HyperbolicSameFluxHermRHS fast_rhs_herm(dg_solu, oper_matx_herm);
	HermInterpolation interp_herm(dg_solu);

	// Burgers' equation have the same flux in x and y direction, thus interpolation is only needed in x direction
	std::vector< std::vector<bool> > is_intp_herm;
	is_intp_herm.push_back(std::vector<bool>{true});

	// fast hermite interpolation
	FastHermIntp fast_herm_intp(dg_solu, interp_herm.Her_pt_Alpt_1D);

	// source term
	auto source_func_time = [&](double x, int i, double t)->double { return 2 * Const::PI * cos(2 * Const::PI * (-t + x)) * (-1 - 4 * Const::PI * Const::PI + sin(2 * Const::PI * (-t + x))); };

	std::cout << "--- evolution started ---" << std::endl;
	// record code running time
	auto start_evolution_time = std::chrono::high_resolution_clock::now();	
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
			sum_c_dx += std::pow(2., max_mesh[d]);
		}		
		double dt = cfl/sum_c_dx;
		dt = std::min( dt, final_time - curr_time );

		// --- part 2: predict by Euler forward
		{
			// before Euler forward, copy Element::ucoe_alpt to Element::ucoe_alpt_predict
			dg_solu.copy_ucoe_to_predict();

			HyperbolicAlpt linear(dg_solu, oper_matx_alpt);
			linear.assemble_matrix_flx_scalar(0, -1, lxf_alpha[0]/2);
			linear.assemble_matrix_flx_scalar(0, 1, -lxf_alpha[0]/2);

			KdvAlpt dg_operator(dg_solu, oper_matx_alpt);
			dg_operator.assemble_matrix_scalar();

			IMEXEuler odesolver(dg_operator, dt, "sparselu");
			odesolver.init();

			// Hermite interpolation
			interp_herm.nonlinear_Herm_1D(func_flux, func_flux_d1, is_intp_herm);
			
			// calculate rhs and update Element::ucoe_alpt
			dg_solu.set_rhs_zero();

			fast_rhs_herm.rhs_vol_scalar();
			fast_rhs_herm.rhs_flx_intp_scalar();

			// add to rhs in ode solver
			odesolver.set_rhs_zero();
			odesolver.add_rhs_to_eigenvec();
			odesolver.add_rhs_matrix(linear);				

			// source term
			double source_time = curr_time;
			auto source_func = [&](double x, int i)->double { return source_func_time(x,i,source_time); };
			Domain1DIntegral source(dg_solu, all_bas_alpt, source_func);
			source.assemble_vector();
			odesolver.add_rhs_vector(source.vec_b);

			odesolver.step_stage(0);

			odesolver.final();
		}

		// --- part 3: refine base on Element::ucoe_alpt
		dg_solu.refine();
		const int num_basis_refine = dg_solu.size_basis_alpt();

		// after refine, copy Element::ucoe_alpt_predict back to Element::ucoe_alpt
		dg_solu.copy_predict_to_ucoe();
	
		// --- part 4: time evolution
		HyperbolicAlpt linear(dg_solu, oper_matx_alpt);
		linear.assemble_matrix_flx_scalar(0, -1, lxf_alpha[0]/2);
		linear.assemble_matrix_flx_scalar(0, 1, -lxf_alpha[0]/2);

		KdvAlpt dg_operator(dg_solu, oper_matx_alpt);
		dg_operator.assemble_matrix_scalar();

		IMEX43 odesolver(dg_operator, dt, "sparselu");
		odesolver.init();

		for (size_t num_stage = 0; num_stage < odesolver.num_stage; num_stage++)
		{
			// calculate rhs for explicit part only for stage = 2, 3, 4
			// no need for stage = 0, 1
			if (num_stage >= 2)
			{
				// Hermite interpolation
				interp_herm.nonlinear_Herm_1D(func_flux, func_flux_d1, is_intp_herm);
				
				// calculate rhs and update Element::ucoe_alpt
				dg_solu.set_rhs_zero();

				fast_rhs_herm.rhs_vol_scalar();
				fast_rhs_herm.rhs_flx_intp_scalar();

				// add to rhs in ode solver
				odesolver.set_rhs_zero();
				odesolver.add_rhs_to_eigenvec();
				odesolver.add_rhs_matrix(linear);				

				// source term
				double source_time = curr_time;
				if (num_stage == 3) { source_time += dt; }
				else if (num_stage == 4) { source_time += dt/2.; }
				auto source_func = [&](double x, int i)->double { return source_func_time(x,i,source_time); };
				Domain1DIntegral source(dg_solu, all_bas_alpt, source_func);
				source.assemble_vector();
				odesolver.add_rhs_vector(source.vec_b);
			}
			odesolver.step_stage(num_stage);

			odesolver.final();
		}

		// --- part 5: coarsen
		dg_solu.coarsen();
		const int num_basis_coarsen = dg_solu.size_basis_alpt();		

		// add current time and increase time steps
		curr_time += dt;
		num_time_step ++;

		if (num_time_step % 10 == 0)
		{
			// record code running time
			auto stop_evolution_time = std::chrono::high_resolution_clock::now(); 
			auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop_evolution_time - start_evolution_time);
			std::cout << "num of time steps: " << num_time_step 
					<< "; time step: " << dt 
					<< "; curr time: " << curr_time
					<< "; running time: " << duration.count()/1e6 << " seconds" << std::endl
					<< "num of basis after refine: " << num_basis_refine
					<< "; num of basis after coarsen: " << num_basis_coarsen				
					<< std::endl << std::endl;
		}
	}
	
	std::cout << "--- evolution finished ---" << std::endl;

	std::cout << "total num of time steps: " << num_time_step << std::endl
			  << "total num of basis (last time step): " << dg_solu.size_basis_alpt() << std::endl;

	// calculate error between numerical solution and exact solution
	std::cout << "calculating error at final time" << std::endl;
	auto final_func = [&](double x, int d)->double { return sin(2 * Const::PI * (x-final_time)); };
	
	const int num_gauss_pt = 10;
	std::vector<double> err = dg_solu.get_error_separable_scalar(final_func, num_gauss_pt);
	std::cout << "L1, L2 and Linf error at final time: " << err[0] << ", " << err[1] << ", " << err[2] << std::endl;

	IO inout(dg_solu);
	std::string file_name = "error_N" + std::to_string(NMAX) + ".txt";
	inout.write_error(NMAX, err, file_name);	
	// ------------------------------

	return 0;
}
