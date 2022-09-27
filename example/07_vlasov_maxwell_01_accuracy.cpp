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

	// initial condition for f = f(x, v) = sin(2*pi*(x+v))
	auto init_func_1 = [](double x, int d) -> double
	{
		if (d == 0) { return sin(2.*Const::PI*x); }
		else if (d == 1) { return cos(2.*Const::PI*x); }
		else if (d == 2) { return (cos(2.*Const::PI*x)); }
	};
	auto init_func_zero = [](double x, int d) { return 0.; };
	std::vector<std::function<double(double, int)>> init_func_f{init_func_1, init_func_zero, init_func_zero};
	dg_f.init_separable_system(init_func_f);

	// initialization of magnetic and electric field with v in the auxiliary dimension
	// dg_BE = (B3(x2, v1, v2), E1(x2, v1, v2), E2(x2, v1, v2))
	const int auxiliary_dim = 2;
	const bool sparse_E = false;	// use full grid in x
	DGAdapt dg_BE(sparse_E, N_init, NMAX, auxiliary_dim, all_bas_alpt, all_bas_lagr, all_bas_herm, hash, refine_eps, coarsen_eta, is_adapt_find_ptr_alpt, is_adapt_find_ptr_intp);
	
	// initial condition for E = cos(2*pi*x)
	auto init_func_3 = [](double x, int d) { return (d==0) ? (cos(2.*Const::PI*x)) : (1.); };
	std::vector<std::function<double(double, int)>> init_func_E{init_func_3, init_func_zero, init_func_zero};
	dg_BE.init_separable_system(init_func_E);
	
	// ------------------------------
	HyperbolicLagrRHS fast_rhs_lagr(dg_f, oper_matx_lagr);

	// fast Lagrange interpolation
    LagrInterpolation interp_lagr(dg_f);
	FastLagrIntp fast_lagr_intp_f(dg_f, interp_lagr.Lag_pt_Alpt_1D, interp_lagr.Lag_pt_Alpt_1D_d1);
	FastLagrIntp fast_lagr_intp_BE(dg_BE, interp_lagr.Lag_pt_Alpt_1D, interp_lagr.Lag_pt_Alpt_1D_d1);

	// constant in global Lax-Friedrich flux	
	const double lxf_alpha = 1.;
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
		HyperbolicAlpt linear(dg_f, oper_matx_alpt);
		std::vector<std::vector<double>> coefficient_d0 = {{1, 0, 0}, {0, 0, 0}, {0, 0, 0}};
		std::vector<std::vector<double>> coefficient_d1 = {{0.9, 0, 0}, {0, 0, 0}, {0, 0, 0}};
		std::vector<std::vector<double>> coefficient_d2 = {{0.88, 0, 0}, {0, 0, 0}, {0, 0, 0}};
		std::vector<std::vector<std::vector<double>>> coefficient = {coefficient_d0, coefficient_d1, coefficient_d2};

		for (int d = 0; d < DIM; d++)
		{
			// volume integral
			linear.assemble_matrix_vol_system(d, coefficient[d]);

			// flux integral
			linear.assemble_matrix_flx_system(d, -1, coefficient[d], 0.5);
			linear.assemble_matrix_flx_system(d, 1, coefficient[d], 0.5);

			// flux integral
			linear.assemble_matrix_flx_system(d, -1, {1, 0, 0}, lxf_alpha/2);
			linear.assemble_matrix_flx_system(d, 1, {1, 0, 0}, -lxf_alpha/2);
		}

        RK3SSP odeSolver_f(linear, dt);
        odeSolver_f.init();

		RK3SSP odeSolver_BE(dg_BE, dt);
		odeSolver_BE.init();

        for ( int stage = 0; stage < odeSolver_f.num_stage; ++stage )
        {			
			// --- step 1: update RHS for f ---
            // Lagrange interpolation
            LagrInterpolation interp(dg_f);
            // // variable to control which flux need interpolation
            // // the first index is # of unknown variable, the second one is # of dimension
            // std::vector< std::vector<bool> > is_intp;
            // is_intp.push_back(std::vector<bool>(DIM, true));
            
            // // f = f(x, v, t) in 1D1V
            // // interpolation for v * f and E * f
			// // f_t + v * f_x + E * f_v = 0
			// const std::vector<int> zero_derivative(DIM, 0);
            // auto coe_func = [&](std::vector<double> x, int d) -> double 
            // {
            //     if (d==0) { return x[1]; }
            //     else { return dg_E.val(x, zero_derivative)[0]; }
            // };
			// interp.var_coeff_u_Lagr_fast(coe_func, is_intp, fast_lagr_intp_f);
			
			// interp.interp_Vlasov_1D1V(dg_E, fast_lagr_intp_f, fast_lagr_intp_E);
			// auto coe_v = [&](double v) -> double { return v; };
			// auto coe_E = [&](double E) -> double { return E; };
			// interp.interp_Vlasov_1D1V(dg_E, coe_v, coe_E, fast_lagr_intp_f, fast_lagr_intp_E);
			
			// interp.interp_Vlasov_1D2V(dg_BE, fast_lagr_intp_f, fast_lagr_intp_BE);
            
			// calculate rhs and update Element::ucoe_alpt
            dg_f.set_rhs_zero();

			// fast_rhs_lagr.rhs_vol_scalar();
			// fast_rhs_lagr.rhs_flx_intp_scalar();

            // add to rhs in odeSolver_f
            odeSolver_f.set_rhs_zero();
            odeSolver_f.add_rhs_to_eigenvec();
            odeSolver_f.add_rhs_matrix(linear);

            // // source term for f
			// {			
            // double source_time = curr_time;
            // if (stage == 1) { source_time += dt; }
            // else if (stage == 2) { source_time += dt/2.; }
			// auto source_func = [&](std::vector<double> x, int i)->double
			// {
			// 	return (cos(2*(Const::PI)*(source_time + x[1] + x[0]))*(sin(2*(Const::PI)*(source_time + x[0])) - sin(2*(Const::PI)*x[0]) + 4*x[1]*(Const::PI)*(Const::PI) + 4*(Const::PI)*(Const::PI)*cos(2*(Const::PI)*x[0]) + 4*(Const::PI)*(Const::PI)))/(2*(Const::PI));
			// };
            // Domain2DIntegral source_operator(dg_f, all_bas_alpt, source_func);
            // source_operator.assemble_vector();
            // odeSolver_f.add_rhs_vector(source_operator.vec_b);
            // }

            odeSolver_f.step_stage(stage);
			
			// // --- step 2: update RHS for E ---
			// dg_E.set_rhs_zero();

			// const std::vector<int> moment_order{0, 1};
			// const std::vector<double> moment_order_weight{0., -1.};
			// const int num_vec = 0;
			// dg_E.compute_moment_full_grid(dg_f, moment_order, moment_order_weight, num_vec);

            // odeSolver_E.set_rhs_zero();
			// odeSolver_E.add_rhs_to_eigenvec();

			// odeSolver_E.step_stage(stage);

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
					<< "; curr time: " << curr_time << std::endl
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
		// return sin(2*Const::PI*(x[0] + x[1] + final_time));
		return sin(2*Const::PI*(x[0] - final_time)) * cos(2*Const::PI*(x[1] - 0.9*final_time)) * (cos(2.*Const::PI*(x[2] - 0.88*final_time)));
	};
	std::vector<double> err_f = dg_f.get_error_no_separable_scalar(final_func_f, num_gauss_pt_compute_error);
	std::cout << "L1, L2 and Linf error (for f) at final time: " << err_f[0] << ", " << err_f[1] << ", " << err_f[2] << std::endl;

	// // compute error for E
	// auto final_func_E = [&](std::vector<double> x) -> double 
	// {
	// 	return (sin(2*(Const::PI)*(final_time + x[0])) - sin(2*(Const::PI)*x[0]) + 4*(Const::PI)*(Const::PI)*cos(2*(Const::PI)*x[0]))/(4*(Const::PI)*(Const::PI));
	// };
	// std::vector<double> err_E = dg_BE.get_error_no_separable_scalar(final_func_E, num_gauss_pt_compute_error);
	// std::cout << "L1, L2 and Linf error (for E) at final time: " << err_E[0] << ", " << err_E[1] << ", " << err_E[2] << std::endl;

	return 0;
}
