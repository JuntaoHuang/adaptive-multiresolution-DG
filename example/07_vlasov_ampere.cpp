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
#include "FastMultiplyLU.h"
#include "Timer.h"


int main(int argc, char *argv[])
{
	// constant variable
	const int DIM = 2;

	// static variables
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

	DGSolution::ind_var_vec = { 0 };
	DGAdapt::indicator_var_adapt = { 0 };

	Element::is_intp.resize(Element::VEC_NUM);
	for (size_t num = 0; num < Element::VEC_NUM; num++) { Element::is_intp[num] = std::vector<bool>{true, false}; }

	Interpolation::DIM = DGSolution::DIM;
	Interpolation::VEC_NUM = DGSolution::VEC_NUM;

	DGSolution::ind_var_vec = { 0 };
	DGAdapt::indicator_var_adapt = { 0 };

	Element::is_intp.resize(Element::VEC_NUM);
	for (size_t num = 0; num < Element::VEC_NUM; num++) { Element::is_intp[num] = std::vector<bool>(Element::DIM, true); }

	// computation parameter
	int NMAX = 3;
	int N_init = NMAX;
	int is_init_sparse = 1;			// use full grid (0) or sparse grid (1) when initialization
									// is this an initial mesh used? 
	double final_time = 0.03;
	double cfl = 0.05;		// cfl number when exist shock, you should tune this parameter
	double cfl_hyper = 0.2;	// cfl number when no shock, 0.33 for P1, 0.2 for P2
	int rk_order = 3;
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
	bool is_adapt_find_ptr_intp = true;

	// output info in screen every time steps
	int output_time_interval = 10;

	// num of gauss points in computing error
	int num_gauss_pt_compute_error = 1;

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

	// initialization of DG solution
	DGAdapt dg_solu(sparse, N_init, NMAX, all_bas_alpt, all_bas_lagr, all_bas_herm, hash, refine_eps, coarsen_eta, is_adapt_find_ptr_alpt, is_adapt_find_ptr_intp);

	// project initial function into numerical solution
	// f(x,y) = sin(2*pi*(x+y)) = sin(2*pi*x) * cos(2*pi*y) + cos(2*pi*x) * sin(2*pi*y)
	auto init_func_1 = [](double x, int d) { return (d==0) ? (sin(2.*Const::PI*x)) : (cos(2.*Const::PI*x)); };
	auto init_func_2 = [](double x, int d) { return (d==0) ? (cos(2.*Const::PI*x)) : (sin(2.*Const::PI*x)); };
	std::vector<std::function<double(double, int)>> init_func{init_func_1, init_func_2};	
	dg_solu.init_separable_scalar_sum(init_func);
	
    // output
	IO inout(dg_solu);

	// ------------------------------
	HyperbolicDiffFluxLagrRHS fast_rhs_lagr(dg_solu, oper_matx_lagr);		

	// fast Lagrange interpolation
    LagrInterpolation interp_lagr(dg_solu);
	FastLagrIntp fast_lagr_intp(dg_solu, interp_lagr.Lag_pt_Alpt_1D, interp_lagr.Lag_pt_Alpt_1D_d1);

	// constant in global Lax-Friedrich flux	
	const std::vector<double> lxf_alpha{1.2, 1.2};
	// wave speed in x and y direction
	const std::vector<double> wave_speed{1., 1.};

	// begin time evolution
	std::cout << "--- evolution started ---" << std::endl;
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
			sum_c_dx += std::abs(wave_speed[d]) * std::pow(2., max_mesh[d]);
		}		
		double dt = cfl/sum_c_dx;
		dt = std::min( dt, final_time - curr_time );
		
        // --- part 4: time evolution
		// linear operator
		// x direction: - [u] * [v] * alpha_x / 2
		HyperbolicAlpt linear(dg_solu, oper_matx_alpt);
		// x direction: u^- * [v] * alpha_x / 2
		linear.assemble_matrix_flx_scalar(0, -1, lxf_alpha[0]/2);
		// x direction: u^+ * [v] * (-alpha_x) / 2
		linear.assemble_matrix_flx_scalar(0, 1, -lxf_alpha[0]/2);
				
        RK3SSP odeSolver(linear, dt);
        odeSolver.init();		

        for ( int stage = 0; stage < odeSolver.num_stage; ++stage )
        {			
            // Lagrange interpolation
            LagrInterpolation interp(dg_solu);
            // variable to control which flux need interpolation
            // the first index is # of unknown variable, the second one is # of dimension
            std::vector< std::vector<bool> > is_intp;
            is_intp.push_back(std::vector<bool>(DIM, true));
            
            // f = f(x, v, t) in 1D1V
            // interpolation for f * v
            // f_t + v * f_x = 0
            auto coe_func = [&](std::vector<double> x, int d) -> double 
            {
                if (d==0) { return x[1]; }
                else { return 0.; }
            };
            interp.var_coeff_u_Lagr_fast(coe_func, is_intp, fast_lagr_intp);

            // calculate rhs and update Element::ucoe_alpt
            dg_solu.set_rhs_zero();

            fast_rhs_lagr.rhs_vol_scalar();
            fast_rhs_lagr.rhs_flx_intp_scalar();

            // add to rhs in odeSolver
            odeSolver.set_rhs_zero();
            odeSolver.add_rhs_to_eigenvec();
            odeSolver.add_rhs_matrix(linear);

            // source term
            double source_time = curr_time;
            if (stage == 1) { source_time += dt; }
            else if (stage == 2) { source_time += dt/2.; }
            // compute projection of source term f(x, y, t) with separable formulation
            auto source_func = [&](std::vector<double> x, int i)->double { return 2*Const::PI*(x[1]+1.)*cos(2*Const::PI*(source_time+x[0]+x[1])); };
            Domain2DIntegral source_operator(dg_solu, all_bas_alpt, source_func);
            source_operator.assemble_vector();
            odeSolver.add_rhs_vector(source_operator.vec_b);
            
            odeSolver.step_stage(stage);
            odeSolver.final();
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
			<< "; num of basis: " << dg_solu.size_basis_alpt() << std::endl;

	auto final_func = [&](std::vector<double> x) -> double 
	{
		return sin(2*Const::PI*(x[0] + x[1] + final_time));
	};

	std::vector<double> err = dg_solu.get_error_no_separable_scalar(final_func, num_gauss_pt_compute_error);
	std::cout << "L1, L2 and Linf error at final time: " << err[0] << ", " << err[1] << ", " << err[2] << std::endl;
		
	std::string file_name = "error_N" + std::to_string(NMAX) + ".txt";
	inout.write_error(NMAX, err, file_name);

	return 0;
}
