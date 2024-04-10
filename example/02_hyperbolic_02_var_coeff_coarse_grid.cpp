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
	const int DIM = 2;

	// static variables
	AlptBasis::PMAX = 1;

	LagrBasis::PMAX = 2;
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

	// computation parameter
	int NMAX = 3;
	int N_init = NMAX;
	int is_init_sparse = 0;			// use full grid (0) or sparse grid (1) when initialization
	double final_time = 2 * Const::PI;
	double cfl = 0.2;
	std::string boundary_type = "period";	// this variable will be used in constructors of class OperatorMatrix1D

	// adaptive parameter
	// if test code without adaptive
	// just set refine_eps to be a large number (for example 1e10), then no refine
	// and set coarsen_eta to be a negative number -1, then no coarsen
	double refine_eps = 1e10;
	double coarsen_eta = -1;

	// variable control if need to adaptively find out pointers related to Alpert and interpolation basis in DG operators
	// if using artificial viscosity, we should set is_adapt_find_ptr_intp to be true
	bool is_adapt_find_ptr_alpt = true;
	bool is_adapt_find_ptr_intp = false;

	// output info in screen every time steps
	int output_time_interval = 1000;

	// num of gauss points in computing error
	int num_gauss_pt_compute_error = 1;

	int NMAX_coarse_grid_stage_1 = NMAX;
	int NMAX_coarse_grid_stage_2 = NMAX;

	// filter coefficient
	double filter_coef = 1.0;

	// numerical test number
	// 1: example 4.2 (solid body rotation in 2D) in Guo and Cheng, SISC (2016)
	// 4: example 4.2 (solid body rotation in 2D) in Guo and Cheng, SISC (2016), with initial condition Gaussian function
	// 2: example 4.2 (solid body rotation in 3D) in Guo and Cheng, SISC (2016)
	// 3: example 4.3 (deformational flow in 2D) in Guo and Cheng, SISC (2016)
	int test_num = 4;

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
	args.AddOption(&NMAX_coarse_grid_stage_1, "-N1", "--max-mesh-stage-1", "Maximum mesh level in stage 1");
	args.AddOption(&NMAX_coarse_grid_stage_2, "-N2", "--max-mesh-stage-2", "Maximum mesh level in stage 2");
	args.AddOption(&filter_coef, "-filter", "--filter-coefficient", "Filter coefficient");
	args.AddOption(&test_num, "-test", "--test-number", "Numerical test number");

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
	
	// check if the test number is valid
	if (test_num < 1 || test_num > 4) { std::cout << "Invalid test number" << std::endl; return 1; }
	if (((test_num == 1) || (test_num == 3) || (test_num == 4)) && DIM != 2) { std::cout << "Test number 1, 3, 4 is only valid for 2D" << std::endl; return 1; }
	if (test_num == 2 && DIM != 3) { std::cout << "Test number 2 is only valid for 3D" << std::endl; return 1; }
	if (test_num == 3) { std::cout << "Test number 3 has variable characteristic speed, not implemented yet" << std::endl; return 1; }

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
	// --- End of Part 1 ---
	// --------------------------------------------------------------------------------------------


	// --------------------------------------------------------------------------------------------
	// --- Part 2: initialization of DG solution ---
	DGAdaptIntp dg_f(sparse, N_init, NMAX, all_bas_alpt, all_bas_lagr, all_bas_herm, hash, refine_eps, coarsen_eta, is_adapt_find_ptr_alpt, is_adapt_find_ptr_intp, oper_matx_lagr_lagr, oper_matx_herm_herm);
	
	// adaptive interpolation for initial function
	auto init_func = [=](std::vector<double> x, int i)
	{
		std::vector<double> xc;
		double b = 0.;
		
		if (test_num == 1 || test_num == 2 || test_num == 3)
		{
			// example 4.2 (deformational flow in 2D)
			if (test_num == 1)
			{
				xc = {0.75, 0.5};
				b = 0.23;
			}
			// example 4.2 (deformational flow in 3D)
			else if (test_num == 2)
			{
				xc = {0.5, 0.55, 0.5};
				b = 0.45;
			}
			// example 4.3 (deformational flow in 2D)
			else if (test_num == 3)
			{
				xc = {0.65, 0.5};
				b = 0.35;
			}

			double r_sqr = 0.;
			for (int d = 0; d < DIM; d++) { r_sqr += pow(x[d] - xc[d], 2.); };
			double r = pow(r_sqr, 0.5);
			if (r <= b) { return pow(b, DIM-1) * pow(cos(Const::PI*r/(2.*b)), 6.); }
			else { return 0.; }
		}

		if (test_num == 4)
		{
			xc = {0.55, 0.5};
			double sigma = 0.075;
			return 1.0/(sigma * sqrt(2.0*Const::PI)) * exp(-1.0/(2.0*sigma*sigma) * (pow(x[0] - xc[0], 2.0) + pow(x[1] - xc[1], 2.0)));
		}
	};
	dg_f.init_adaptive_intp(init_func);

	// // project initial function into numerical solution
	// // f(x,y) = cos(2*pi*(x+y)) = cos(2*pi*x) * cos(2*pi*y) - sin(2*pi*x) * sin(2*pi*y)
	// auto init_func_1 = [](double x, int d) { return (d==0) ? (cos(2.*Const::PI*x)) : (cos(2.*Const::PI*x)); };
	// auto init_func_2 = [](double x, int d) { return (d==0) ? (-sin(2.*Const::PI*x)) : (sin(2.*Const::PI*x)); };
	// std::vector<std::function<double(double, int)>> init_func{init_func_1, init_func_2};	
	// dg_f.init_separable_scalar_sum(init_func);

    // output
	IO inout(dg_f);

	// ------------------------------	
	HyperbolicLagrRHS fast_rhs_lagr(dg_f, oper_matx_lagr);

	// fast Lagrange interpolation
    LagrInterpolation interp_lagr(dg_f);
	FastLagrIntp fast_lagr_intp(dg_f, interp_lagr.Lag_pt_Alpt_1D, interp_lagr.Lag_pt_Alpt_1D_d1);

	// wave speed
	std::vector<double> wave_speed;
	if ((test_num == 1) || (test_num == 4))
	{
		wave_speed = {0.5, 0.5};
	}
	else if (test_num == 2)
	{
		wave_speed = {sqrt(2.)/4., sqrt(2.)/2., sqrt(2.)/4.};
	}
	const std::vector<double> lxf_alpha = wave_speed;

	const int max_mesh = dg_f.max_mesh_level();
	const double dx = 1./pow(2., max_mesh);

	double sum_wave_speed = 0.;
	for (int d = 0; d < DIM; d++) { sum_wave_speed += wave_speed[d]; };
	double dt = dx * cfl / sum_wave_speed;

	int total_time_step = ceil(final_time/dt) + 1;
	dt = final_time/total_time_step;
	// --- End of Part 2 ---
	// --------------------------------------------------------------------------------------------


	// --------------------------------------------------------------------------------------------
	// --- Part 3: time evolution ---
	// // linear operator
	HyperbolicAlpt linear_stage_1(dg_f, oper_matx_alpt);
	for (int d = 0; d < DIM; d++)
	{
		// u^- * [v] * alpha / 2
		linear_stage_1.assemble_matrix_flx_scalar_coarse_grid(d, -1, NMAX_coarse_grid_stage_1, lxf_alpha[d]/2);

		// - u^+ * [v] * alpha / 2
		linear_stage_1.assemble_matrix_flx_scalar_coarse_grid(d, 1, NMAX_coarse_grid_stage_1, -lxf_alpha[d]/2);
	}

	HyperbolicAlpt linear_stage_2(dg_f, oper_matx_alpt);
	for (int d = 0; d < DIM; d++)
	{
		// u^- * [v] * alpha / 2
		linear_stage_2.assemble_matrix_flx_scalar_coarse_grid(d, -1, NMAX_coarse_grid_stage_2, lxf_alpha[d]/2);

		// - u^+ * [v] * alpha / 2
		linear_stage_2.assemble_matrix_flx_scalar_coarse_grid(d, 1, NMAX_coarse_grid_stage_2, -lxf_alpha[d]/2);
	}	

	std::cout << "--- evolution started ---" << std::endl;
	Timer record_time;
	double curr_time = 0.;
	int is_stable = 1;

	// Lagrange interpolation
	LagrInterpolation interp(dg_f);
	// variable to control which flux need interpolation
	// the first index is # of unknown variable, the second one is # of dimension
	std::vector< std::vector<bool> > is_intp;
	is_intp.push_back(std::vector<bool>(DIM, true));

	RK2Midpoint odeSolver(linear_stage_1, dt);

	for (size_t num_time_step = 0; num_time_step < total_time_step; num_time_step++)
	{			        
        odeSolver.init();		

        for ( int stage = 0; stage < odeSolver.num_stage; ++stage )
        {
            auto coe_func = [&](std::vector<double> x, int d) -> double 
            {
				// a = (-x2 + 0.5, x1 - 0.5)
				if (test_num == 1)
				{
					if (d==0) { return -x[1] + 0.5; }
					else { return x[0] - 0.5; }
				}
				
				// a = (-sqrt(2)/2 * (x2 - 0.5),
				// 		sqrt(2)/2 * (x1 - 0.5) + sqrt(2)/2 * (x3 - 0.5)
				// 		-sqrt(2)/2 * (x2 - 0.5))
				else if (test_num == 2)
				{
					if (d==0) { return -sqrt(2.)/2. * (x[1] - 0.5); }
					else if (d==1) { return sqrt(2.)/2. * (x[0] - 0.5) + sqrt(2.)/2. * (x[2] - 0.5); }
					else { return -sqrt(2.)/2. * (x[1] - 0.5); }
				}
            };

			if (stage == 0)
			{
				interp.var_coeff_u_Lagr_fast_coarse_grid(coe_func, is_intp, fast_lagr_intp, NMAX_coarse_grid_stage_1);
			}
			else if (stage == 1)
			{
				interp.var_coeff_u_Lagr_fast_coarse_grid(coe_func, is_intp, fast_lagr_intp, NMAX_coarse_grid_stage_2);
			}
            
            // calculate rhs and update Element::ucoe_alpt
            dg_f.set_rhs_zero();

			if (stage == 0)
			{
				fast_rhs_lagr.rhs_vol_scalar_coarse_grid(NMAX_coarse_grid_stage_1);
				fast_rhs_lagr.rhs_flx_intp_scalar_coarse_grid(NMAX_coarse_grid_stage_1);
			}
			else if (stage == 1)
			{
				fast_rhs_lagr.rhs_vol_scalar_coarse_grid(NMAX_coarse_grid_stage_2);
				fast_rhs_lagr.rhs_flx_intp_scalar_coarse_grid(NMAX_coarse_grid_stage_2);
			}

            // add to rhs in odeSolver
            odeSolver.set_rhs_zero();
            odeSolver.add_rhs_to_eigenvec();
			if (stage == 0)
			{
				odeSolver.add_rhs_matrix(linear_stage_1);
			}
			else if (stage == 1)
			{
				odeSolver.add_rhs_matrix(linear_stage_2);
			}
            
            odeSolver.step_stage(stage);
            odeSolver.final();
        }

		// add filter to solution for stability
		dg_f.filter(filter_coef, wave_speed, dt, NMAX_coarse_grid_stage_1 + 1);

		curr_time += dt;
		
		// record code running time
		if (num_time_step % output_time_interval == 0)
		{
			// compute L2 norm of numerical solution
			std::vector<double> solu_l2_norm = dg_f.get_L2_norm();

			if ((solu_l2_norm[0] > 100.0) || std::isnan(solu_l2_norm[0]) || std::isinf(solu_l2_norm[0]))
			{ 
				std::cout << "L2 norm is too LARGE: " << solu_l2_norm[0] << std::endl;
				is_stable = 0;
				break;
			}

			record_time.time("running time");
			std::cout << "num of time steps: " << num_time_step 
					<< "; time step: " << dt 
					<< "; curr time: " << curr_time << std::endl;
		}		
	}

	std::cout << "--- evolution finished ---" << std::endl;
	
	record_time.time("running time");
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
		return init_func(x, i);
	};
	dg_solu_ext.init_adaptive_intp(init_func);

	// compute L2 error between u_h (numerical solution) and v_h (interpolation to exact solution)
	double err_l2 = dg_solu_ext.get_L2_error_split_adaptive_intp_scalar(dg_f);
	std::cout << "L2 error at final time: " << err_l2 << std::endl;	
	record_time.time("elasped time in computing error");

	// auto final_func = [&](std::vector<double> x) -> double 
	// {	
	// 	return exp(- (x[0] - 0.5) * (x[0] - 0.5) * 100. - (x[1] - 0.5) * (x[1] - 0.5) * 100.);
	// };
	// std::vector<double> err = dg_f.get_error_no_separable_scalar(final_func, num_gauss_pt_compute_error);
	// std::cout << "L1, L2 and Linf error at final time: " << err[0] << ", " << err[1] << ", " << err[2] << std::endl;

	// auto final_function = [&](std::vector<double> x) -> double 
	// {	
	// 	double sum_x = 0.;
	// 	for (int d = 0; d < DIM; d++) { sum_x += x[d]; };
	// 	return cos(2*Const::PI*(sum_x - DIM*final_time));
	// };
	// std::vector<double> err = dg_f.get_error_no_separable_scalar(final_function, 4);	
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
