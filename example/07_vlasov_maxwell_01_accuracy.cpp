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

int main(int argc, char *argv[])
{
	// constant variable
	const int DIM = 3;
	const int VEC_NUM = 3;

	// static variables
	AlptBasis::PMAX = 3;

	LagrBasis::PMAX = 4;
	LagrBasis::msh_case = 2;

	HermBasis::PMAX = 3;
	HermBasis::msh_case = 1;

	Element::PMAX_alpt = AlptBasis::PMAX;	// max polynomial degree for Alpert's basis functions
	Element::PMAX_intp = LagrBasis::PMAX;	// max polynomial degree for interpolation basis functions
	Element::DIM = DIM;						// dimension
	Element::VEC_NUM = VEC_NUM;				// num of unknown variables in PDEs

	DGSolution::DIM = Element::DIM;
	DGSolution::VEC_NUM = Element::VEC_NUM;

	Interpolation::DIM = DGSolution::DIM;
	Interpolation::VEC_NUM = DGSolution::VEC_NUM;

	DGSolution::ind_var_vec = { 0, 1, 2 };
	DGAdapt::indicator_var_adapt = { 0 };

	Element::is_intp.resize(Element::VEC_NUM);
	for (size_t num = 0; num < Element::VEC_NUM; num++) { Element::is_intp[num] = std::vector<bool>(Element::DIM, true); }

	// computation parameter
	int NMAX = 3;
	int N_init = NMAX;
	int is_init_sparse = 0;			// use full grid (0) or sparse grid (1) when initialization
	double final_time = 0.1;
	double cfl = 0.1;
	std::string boundary_type = "period";	// this variable will be used in constructors of class OperatorMatrix1D

	// adaptive parameter
	// if you need to test code without adaptive
	// just set refine_eps to be a large number (for example 1e10), then no refine
	// and set coarsen_eta to be a negative number -1, then no coarsen
	double refine_eps = 1e10;
	double coarsen_eta = -1;

	// variable control if need to adaptively find out pointers related to Alpert and interpolation basis in DG operators
	// if using artificial viscosity, we should set is_adapt_find_ptr_intp to be true
	bool is_adapt_find_ptr_alpt = true;
	bool is_adapt_find_ptr_intp = false;

	// output info in screen every time steps
	int output_time_interval = 100;

	// num of gauss points in computing error
	int num_gauss_pt_compute_error = 3;

	OptionsParser args(argc, argv);
	args.AddOption(&NMAX, "-NM", "--max-mesh-level", "Maximum mesh level");
	args.AddOption(&N_init, "-N0", "--initial-mesh-level", "Mesh level in initialization");
	args.AddOption(&is_init_sparse, "-s", "--sparse-grid-initial", "Use full grid (0) or sparse grid (1 by default) in initialization");
	args.AddOption(&final_time, "-tf", "--final-time", "Final time; start time is 0.");
	args.AddOption(&cfl, "-cfl", "--cfl-number", "CFL number");	
	args.AddOption(&refine_eps, "-r", "--refine-epsilon", "refine parameter epsilon");
	args.AddOption(&coarsen_eta, "-c", "--coarsen-eta", "coarsen parameter eta");
	args.AddOption(&output_time_interval, "-p", "--print-every-time-step", "every time steps output info in screen");
	args.AddOption(&num_gauss_pt_compute_error, "-g", "--num-gauss-pt-compute-error", "number of gauss points in computing error");

	args.Parse();
	if (!args.Good())
	{
		args.PrintUsage(std::cout);
		return 1;
	}
	args.PrintOptions(std::cout);

	// check mesh level in initialization should be less or equal to maximum mesh level
	if (N_init>NMAX) { std::cout << "Mesh level in initialization should not be larger than Maximum mesh level" << std::endl; return 1; }
	bool sparse = ((is_init_sparse==1) ? true : false);

	// initialize hash key
	Hash hash;

	// initialize interpolation points for Lagrange and Hermite interpolation basis
	LagrBasis::set_interp_msh01();
	HermBasis::set_interp_msh01();

	// initialize all basis functions in 1D for Alpert, Lagrange and Hermite basis
	AllBasis<LagrBasis> all_bas_lagr(NMAX);
	AllBasis<HermBasis> all_bas_herm(NMAX);
	AllBasis<AlptBasis> all_bas_alpt(NMAX);

	// operator matrix in 1D
	OperatorMatrix1D<AlptBasis,AlptBasis> oper_matx_alpt(all_bas_alpt, all_bas_alpt, boundary_type);
	OperatorMatrix1D<HermBasis,AlptBasis> oper_matx_herm(all_bas_herm, all_bas_alpt, boundary_type);
	OperatorMatrix1D<LagrBasis,AlptBasis> oper_matx_lagr(all_bas_lagr, all_bas_alpt, boundary_type);

	OperatorMatrix1D<HermBasis, HermBasis> oper_matx_herm_herm(all_bas_herm, all_bas_herm, boundary_type);
	OperatorMatrix1D<LagrBasis, LagrBasis> oper_matx_lagr_lagr(all_bas_lagr, all_bas_lagr, boundary_type);

	// initialization of DG solution (distribution function f)
	DGAdaptIntp dg_f(sparse, N_init, NMAX, all_bas_alpt, all_bas_lagr, all_bas_herm, hash, refine_eps, coarsen_eta, is_adapt_find_ptr_alpt, is_adapt_find_ptr_intp, oper_matx_lagr_lagr, oper_matx_herm_herm);

	// initial condition for f
	auto init_func_1 = [](double x, int d) -> double
	{
		if (d == 0) { return sin(2.*Const::PI*x); }
		else if (d == 1) { return pow(sin(2.*Const::PI*x), 2.); }
		else if (d == 2) { return pow(cos(2.*Const::PI*x), 2.); }
	};
	auto init_func_zero = [](double x, int d) { return 0.; };
	std::vector<std::function<double(double, int)>> init_func_f{init_func_1, init_func_zero, init_func_zero};
	dg_f.init_separable_system(init_func_f);

	// initialization of magnetic and electric field with v in the auxiliary dimension
	// dg_BE = (B3(x2, v1, v2), E1(x2, v1, v2), E2(x2, v1, v2))
	const int auxiliary_dim = 2;
	const bool sparse_E = false;	// use full grid in x
	DGAdapt dg_BE(sparse_E, N_init, NMAX, auxiliary_dim, all_bas_alpt, all_bas_lagr, all_bas_herm, hash, refine_eps, coarsen_eta, is_adapt_find_ptr_alpt, is_adapt_find_ptr_intp);
	
	// initial condition for (B3, E1, E2)
	auto init_func_B3 = [](double x, int d) -> double
	{
		if (d==0) { return cos(2.*Const::PI*x); }
		else { return 1.0; }
	};
	auto init_func_E1 = [](double x, int d) -> double
	{
		if (d==0) { return sin(2.*Const::PI*x); }
		else { return 1.0; }
	};
	auto init_func_E2 = [](double x, int d) -> double
	{
		if (d==0) { return cos(4.*Const::PI*x); }
		else { return 1.0; }
	};		
	std::vector<std::function<double(double, int)>> init_func_BE{init_func_B3, init_func_E1, init_func_E2};
	dg_BE.init_separable_system(init_func_BE);
	
	// ------------------------------
	HyperbolicLagrRHS fast_rhs_lagr(dg_f, oper_matx_lagr);

	// fast Lagrange interpolation
    LagrInterpolation interp_lagr(dg_f);
	FastLagrIntp fast_lagr_intp_f(dg_f, interp_lagr.Lag_pt_Alpt_1D, interp_lagr.Lag_pt_Alpt_1D_d1);
	FastLagrIntp fast_lagr_intp_BE(dg_BE, interp_lagr.Lag_pt_Alpt_1D, interp_lagr.Lag_pt_Alpt_1D_d1);

	// constant in global Lax-Friedrich flux	
	const double lxf_alpha = 2.;
	// wave speed in x and y direction
	const std::vector<double> wave_speed{1., 1., 1.};

	// begin time evolution
	std::cout << "--- evolution started ---" << std::endl;
	Timer record_time;
	double curr_time = 0.;
	int num_time_step = 0;

	while ( curr_time < final_time )
	{			
		// --- part 1: calculate time step dt ---	
		const std::vector<int> & max_mesh = dg_f.max_mesh_level_vec();
		
		// dt = cfl/(c1/dx1 + c2/dx2 + ... + c_dim/dx_dim)
		double sum_c_dx = 0.;	// this variable stores (c1/dx1 + c2/dx2 + ... + c_dim/dx_dim)
		for (size_t d = 0; d < DIM; d++)
		{
			sum_c_dx += std::abs(wave_speed[d]) * std::pow(2., max_mesh[d]);
		}		
		double dt = cfl/sum_c_dx;
		dt = std::min( dt, final_time - curr_time );
		
        // --- part 4: time evolution
		// linear operator for f
		HyperbolicAlpt linear_f(dg_f, oper_matx_alpt);
		// std::vector<std::vector<double>> coefficient_d0 = {{1, 0, 0}, {0, 0, 0}, {0, 0, 0}};
		// std::vector<std::vector<double>> coefficient_d1 = {{2, 0, 0}, {0, 0, 0}, {0, 0, 0}};
		// std::vector<std::vector<double>> coefficient_d2 = {{2, 0, 0}, {0, 0, 0}, {0, 0, 0}};
		// std::vector<std::vector<std::vector<double>>> coefficient = {coefficient_d0, coefficient_d1, coefficient_d2};
		for (int d = 0; d < DIM; d++)
		{
			// flux integral
			linear_f.assemble_matrix_flx_system(d, -1, {1, 0, 0}, lxf_alpha/2);
			linear_f.assemble_matrix_flx_system(d, 1, {1, 0, 0}, -lxf_alpha/2);
		}

		// linear operator for (B3, E1, E2)
		HyperbolicAlpt linear_BE(dg_BE, oper_matx_alpt);
		
		int d = 0;	// Maxwell equation only in x2 dimension
		// volume integral
		linear_BE.assemble_matrix_vol_system(d, {{0, -1, 0}, {-1, 0, 0}, {0, 0, 0}});

		// flux integral, upwind flux
		linear_BE.assemble_matrix_flx_system(d, -1, {{0.5, -0.5, 0}, {-0.5, 0.5, 0}, {0, 0, 0}});
		linear_BE.assemble_matrix_flx_system(d, 1, {{-0.5, -0.5, 0}, {-0.5, -0.5, 0}, {0, 0, 0}});

        RK3SSP odeSolver_f(linear_f, dt);
        odeSolver_f.init();

		RK3SSP odeSolver_BE(linear_BE, dt);
		odeSolver_BE.init();

        for ( int stage = 0; stage < odeSolver_f.num_stage; ++stage )
        {			
			// --- step 1: update RHS for f ---
            // Lagrange interpolation
            LagrInterpolation interp_f(dg_f);
			interp_f.interp_Vlasov_1D2V(dg_BE, fast_lagr_intp_f, fast_lagr_intp_BE);
            
			// start computation of rhs
            dg_f.set_rhs_zero();

			// compute source for f
			FastLagrInit fastLagr_source_f(dg_f, oper_matx_lagr);
			double source_time = curr_time;
            if (stage == 1) { source_time += dt; }
            else if (stage == 2) { source_time += dt/2.; }
			auto source_func_f = [&](std::vector<double> x, int i)->double
			{
				double x2 = x[0]; double v1 = x[1]; double v2 = x[2];
				double pi = Const::PI; double t = source_time;				
				if (i==0)
				{
					// return exp(t)*cos(2*pi*v2)*sin(2*pi*v1)*sin(2*pi*x2) - 2*pi*exp(t)*sin(2*pi*v1)*sin(2*pi*v2)*sin(2*pi*x2)*(exp(-t)*(2*pow(cos(2*pi*x2),2) - 1) - v1*cos(2*pi*x2)*(t + 1)) + 2*v2*pi*exp(t)*cos(2*pi*v2)*cos(2*pi*x2)*sin(2*pi*v1) + 2*pi*exp(t)*cos(2*pi*v1)*cos(2*pi*v2)*sin(2*pi*x2)*(exp(2*t)*sin(2*pi*x2) + v2*cos(2*pi*x2)*(t + 1));
					return exp(t)*pow(cos(2*pi*v2),2)*pow(sin(2*pi*v1),2)*sin(2*pi*x2) + 2*v2*pi*exp(t)*pow(cos(2*pi*v2),2)*cos(2*pi*x2)*pow(sin(2*pi*v1),2) - 4*pi*exp(t)*cos(2*pi*v2)*pow(sin(2*pi*v1),2)*sin(2*pi*v2)*sin(2*pi*x2)*(exp(-t)*(2*pow(cos(2*pi*x2),2) - 1) - v1*cos(2*pi*x2)*(t + 1)) + 4*pi*exp(t)*cos(2*pi*v1)*pow(cos(2*pi*v2),2)*sin(2*pi*v1)*sin(2*pi*x2)*(exp(2*t)*sin(2*pi*x2) + v2*cos(2*pi*x2)*(t + 1));
				}
				else { return 0.; }
			};
			interp_f.source_from_lagr_to_rhs(source_func_f, fastLagr_source_f);

			fast_rhs_lagr.rhs_vol_scalar();
			fast_rhs_lagr.rhs_flx_intp_scalar();

            // add to rhs in odeSolver_f
            odeSolver_f.set_rhs_zero();
            odeSolver_f.add_rhs_to_eigenvec();
            odeSolver_f.add_rhs_matrix(linear_f);

            odeSolver_f.step_stage(stage);
			
			// // --- step 2: update RHS for E ---
			dg_BE.set_rhs_zero();

			// compute moments of f
			// compute - \iint f(x2, v1, v2) v1 dv1 dv2
			dg_BE.compute_moment_1D2V(dg_f, {1, 0}, -1.0, 1, 0);
			// compute - \iint f(x2, v1, v2) v2 dv1 dv2
			dg_BE.compute_moment_1D2V(dg_f, {0, 1}, -1.0, 2, 0);

			// compute source for (B3, E1, E2)
			LagrInterpolation interp_BE(dg_BE);
			FastLagrInit fastLagr_source_BE(dg_BE, oper_matx_lagr);
			auto source_func_BE = [&](std::vector<double> x, int i)->double
			{
				double x2 = x[0]; double v1 = x[1]; double v2 = x[2];
				double pi = Const::PI; double t = source_time;

				// source for B3
				if (i==0)
				{
					return -cos(2*pi*x2)*(2*pi*exp(2*t) - 1);
				}
				// source for E1
				else if (i==1)
				{
					// return 2*sin(2*pi*x2)*(pi + exp(2*t) + pi*t);
					return (sin(2*pi*x2)*(16*pi + 16*exp(2*t) + exp(t) + 16*pi*t))/8;

				}
				// source for E2
				else
				{
					// return -exp(-t)*cos(4*pi*x2);
					return (exp(-t)*(exp(2*t)*sin(2*pi*x2) + 16*pow(sin(2*pi*x2),2) - 8))/8;
				}
			};
			interp_BE.source_from_lagr_to_rhs(source_func_BE, fastLagr_source_BE);

            odeSolver_BE.set_rhs_zero();
			odeSolver_BE.add_rhs_to_eigenvec();

			// compute Maxwell
			odeSolver_BE.add_rhs_matrix(linear_BE);
			
			odeSolver_BE.step_stage(stage);

			// --- step 3: copy ODESolver::ucoe to Element::ucoe_alpt for f and set ODESolver::rhs to be zero ---
            odeSolver_f.final();

			// --- step 4: copy ODESolver::ucoe to Element::ucoe_alpt for E and set ODESolver::rhs to be zero ---
			odeSolver_BE.final();
        }

		// add current time and increase time steps
		curr_time += dt;
		num_time_step ++;
		
		// record code running time
		if (num_time_step % output_time_interval == 0)
		{
			record_time.time("running time");
			std::cout << "num of time steps: " << num_time_step 
					<< "; time step: " << dt 
					<< "; elapsed time: " << curr_time << std::endl
					<< std::endl << std::endl;
		}		
	}

	std::cout << "--- evolution finished ---" << std::endl;
	
	record_time.time("running time");
	std::cout << "num of time steps: " << num_time_step
			<< "; num of basis: " << dg_f.size_basis_alpt() << std::endl;

	// compute error for f = f(x, v, t) = sin(2*pi*(x+v+t))
	auto final_func_f = [&](std::vector<double> x) -> double 
	{	
		double x2 = x[0]; double v1 = x[1]; double v2 = x[2];
		double pi = Const::PI; double t = final_time;

		// return sin(2*pi*(x2)) * sin(2*pi*(v1)) * cos(2*pi*(v2)) * exp(t);
		return sin(2*pi*(x2)) * pow(sin(2*pi*(v1)), 2.) * pow(cos(2*pi*(v2)), 2.) * exp(t);
	};
	std::vector<double> err_f = dg_f.get_error_no_separable_system(final_func_f, num_gauss_pt_compute_error, 0);
	std::cout << "L1, L2 and Linf error (for f) at final time: " << err_f[0] << ", " << err_f[1] << ", " << err_f[2] << std::endl;

	// compute error for (B3, E1, E2)
	auto final_func_B3 = [&](std::vector<double> x) -> double 
	{
		double x2 = x[0]; double pi = Const::PI; double t = final_time;		
		return cos(2*pi*x2) * (t + 1);
		// // exact solution to only Maxwell equation
		// return 0.5 * (init_func_B3(x2+t, 0) + init_func_B3(x2-t, 0) + init_func_E1(x2+t, 0) - init_func_E1(x2-t, 0));
	};
	auto final_func_E1 = [&](std::vector<double> x) -> double 
	{
		double x2 = x[0]; double pi = Const::PI; double t = final_time;		
		return sin(2*pi*x2) * exp(2*t);
		// // exact solution to only Maxwell equation
		// return 0.5 * (init_func_B3(x2+t, 0) - init_func_B3(x2-t, 0) + init_func_E1(x2+t, 0) + init_func_E1(x2-t, 0));
	};
	auto final_func_E2 = [&](std::vector<double> x) -> double 
	{
		double x2 = x[0]; double pi = Const::PI; double t = final_time;		
		return cos(4*pi*x2) * exp(-t);
	};
	std::vector<double> err_B3 = dg_BE.get_error_no_separable_system(final_func_B3, num_gauss_pt_compute_error, 0);
	std::vector<double> err_E1 = dg_BE.get_error_no_separable_system(final_func_E1, num_gauss_pt_compute_error, 1);
	std::vector<double> err_E2 = dg_BE.get_error_no_separable_system(final_func_E2, num_gauss_pt_compute_error, 2);
	std::cout << "L1, L2 and Linf error (for B3) at final time: " << err_B3[0] << ", " << err_B3[1] << ", " << err_B3[2] << std::endl;
	std::cout << "L1, L2 and Linf error (for E1) at final time: " << err_E1[0] << ", " << err_E1[1] << ", " << err_E1[2] << std::endl;
	std::cout << "L1, L2 and Linf error (for E2) at final time: " << err_E2[0] << ", " << err_E2[1] << ", " << err_E2[2] << std::endl;

	return 0;
}
