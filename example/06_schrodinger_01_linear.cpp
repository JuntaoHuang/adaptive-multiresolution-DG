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

// sparse or full grid for nonlinear schrodinger equation in 2D
// i u_t + u_xx + u_yy + (|u|^2 + |u|^4) * u = 0
// with periodic boundary condition
// exact solution:
// u(x,y,t) = exp(i(2*pi*(x+y) - wt))
// with w = 8*pi^2 - 2
int main(int argc, char *argv[])
{
	// static variables
	AlptBasis::PMAX = 2;

	LagrBasis::PMAX = AlptBasis::PMAX;
	LagrBasis::msh_case = 2;
	if (LagrBasis::PMAX == 3) { LagrBasis::msh_case = 3; }

	HermBasis::PMAX = 3;
	HermBasis::msh_case = 1;

	Element::PMAX_alpt = AlptBasis::PMAX;	// max polynomial degree for Alpert's basis functions
	Element::PMAX_intp = LagrBasis::PMAX;	// max polynomial degree for interpolation basis functions
	Element::DIM = 2;			// dimension
	Element::VEC_NUM = 2;		// num of unknown variables in PDEs

	DGSolution::DIM = Element::DIM;
	DGSolution::VEC_NUM = Element::VEC_NUM;

	Interpolation::DIM = DGSolution::DIM;
	Interpolation::VEC_NUM = DGSolution::VEC_NUM;

	DGSolution::ind_var_vec = { 0, 1 };
	DGAdapt::indicator_var_adapt = { 0, 1 };

	Element::is_intp.resize(Element::VEC_NUM);
	for (size_t num = 0; num < Element::VEC_NUM; num++) { Element::is_intp[num] = std::vector<bool>(Element::DIM, true); }

	// constant variable
	int DIM = Element::DIM;
	int VEC_NUM = Element::VEC_NUM;
	int NMAX = 6;
	int N_init = NMAX;
	bool sparse = true;			// use full grid (0) or sparse grid (1) when initialization
	const std::string boundary_type = "period";
	double final_time = 0.1;
	const double cfl = 0.1;
	const bool is_adapt_find_ptr_alpt = true;	// variable control if need to adaptively find out pointers related to Alpert basis in DG operators
	const bool is_adapt_find_ptr_intp = false;	// variable control if need to adaptively find out pointers related to interpolation basis in DG operators

	// adaptive parameter
	// if need to test code without adaptive, just set refine_eps large number 1e6, then no refine
	// and set coarsen_eta negative number -1, then no coarsen
	const double refine_eps = 1e10;
	const double coarsen_eta = -1;

	OptionsParser args(argc, argv);
	args.AddOption(&NMAX, "-N", "--max-mesh-level", "Maximum mesh level");
	args.AddOption(&final_time, "-tf", "--final-time", "Final time; start time is 0.");
	args.Parse();
	if (!args.Good())
	{
		args.PrintUsage(std::cout);
		return 1;
	}
	args.PrintOptions(std::cout);

	N_init = NMAX;

	// hash key
	Hash hash;

	LagrBasis::set_interp_msh01();

	HermBasis::set_interp_msh01();

	AllBasis<LagrBasis> all_bas_lagr(NMAX);

	AllBasis<HermBasis> all_bas_herm(NMAX);

	AllBasis<AlptBasis> all_bas_alpt(NMAX);

	// operator matrix for Alpert basis
	OperatorMatrix1D<AlptBasis, AlptBasis> oper_matx_alpt(all_bas_alpt, all_bas_alpt, boundary_type);

	OperatorMatrix1D<HermBasis, AlptBasis> oper_matx_herm(all_bas_herm, all_bas_alpt, boundary_type);

	OperatorMatrix1D<LagrBasis, AlptBasis> oper_matx_lagr(all_bas_lagr, all_bas_alpt, boundary_type);

	auto func_source = [](std::vector<double> u, int i, int d)->double
		{
			// return 0.;
			double u2 = u[0]*u[0]+u[1]*u[1];
			// |u|^2 * u + |u|^4 * u
			if (i==0) { return -(u2 + u2 * u2)*u[1]; }
			else if (i==1) { return (u2 + u2 * u2)*u[0]; }
			// // |u|^2 * u
			// if (i==0) { return -(u2)*u[1]; }
			// else if (i==1) { return (u2)*u[0]; }
		};

	// initial condition
	// u(x, y) = exp(i*2*pi*(x+y)) = cos(2*pi*(x+y)) + i * sin(2*pi*(x+y))
	std::function<double(double, int)> init_func_1 = [](double x, int d) { return (d == 0) ? (cos(2.*Const::PI*x)) : (cos(2.*Const::PI*x)); };
	std::function<double(double, int)> init_func_2 = [](double x, int d) { return (d == 0) ? -(sin(2.*Const::PI*x)) : (sin(2.*Const::PI*x)); };
	std::function<double(double, int)> init_func_3 = [](double x, int d) { return 0.; };
	std::function<double(double, int)> init_func_4 = [](double x, int d) { return 0.; };
	std::vector<std::function<double(double, int)>> init_func1{ init_func_1, init_func_2, init_func_3, init_func_4 };

	init_func_1 = [](double x, int d) { return (d == 0) ? (sin(2.*Const::PI*x)) : (cos(2.*Const::PI*x)); };
	init_func_2 = [](double x, int d) { return (d == 0) ? (cos(2.*Const::PI*x)) : (sin(2.*Const::PI*x)); };
	init_func_3 = [](double x, int d) { return 0.; };
	init_func_4 = [](double x, int d) { return 0.; };
	std::vector<std::function<double(double, int)>> init_func2{ init_func_1, init_func_2, init_func_3, init_func_4 };
		
	std::vector<std::vector<std::function<double(double, int)>>> init_func{ init_func1, init_func2};

	// initialization of DG solution
	DGAdapt dg_solu(sparse, N_init, NMAX, all_bas_alpt, all_bas_lagr, all_bas_herm, hash, refine_eps, coarsen_eta, is_adapt_find_ptr_alpt, is_adapt_find_ptr_intp);

	// project initial function into numerical solution
	dg_solu.init_separable_system_sum(init_func);

	const int max_mesh = dg_solu.max_mesh_level();
	const double dx = 1. / pow(2., max_mesh);
	double dt = cfl * dx;
	if (AlptBasis::PMAX==3) { dt = cfl * pow(dx, 4./3.); }
	int total_time_step = ceil(final_time / dt) + 1;
	dt = final_time / total_time_step;

	SchrodingerAlpt linear(dg_solu, oper_matx_alpt);
	linear.assemble_matrix();

	IMEX43 odesolver(linear, dt, "sparselu");	
	odesolver.init();

	LagrInterpolation interp_lagr(dg_solu);
	FastLagrIntp fast_lagr_intp(dg_solu, interp_lagr.Lag_pt_Alpt_1D, interp_lagr.Lag_pt_Alpt_1D_d1);	

	SourceFastLagr fast_source_lagr(dg_solu, oper_matx_lagr);

	// only interpolate in dimension 0
	std::vector< std::vector<bool> > is_intp_lagr;
	for (int i = 0; i < VEC_NUM; i++)
	{
		is_intp_lagr.push_back(std::vector<bool>{true, false});
	}

	std::cout << "--- evolution started ---" << std::endl;
	// record code running time
	auto start_evolution_time = std::chrono::high_resolution_clock::now();

	double curr_time = 0;
	for (size_t num_time_step = 0; num_time_step < total_time_step; num_time_step++)
	{
		odesolver.init();
		for (size_t num_stage = 0; num_stage < odesolver.num_stage; num_stage++)
		{
			// calculate rhs for explicit part only for stage = 2, 3, 4
			// no need for stage = 0, 1
			if (num_stage >= 2)
			{
				interp_lagr.nonlinear_Lagr_fast(func_source, is_intp_lagr, fast_lagr_intp);
				
				// calculate rhs and update Element::ucoe_alpt
				dg_solu.set_rhs_zero();

				// nonlinear source terms
				fast_source_lagr.rhs_source();

				// add to rhs in ode solver
				odesolver.set_rhs_zero();
				odesolver.add_rhs_to_eigenvec();
			}
			odesolver.step_stage(num_stage);

			odesolver.final();
		}

		curr_time += dt;

		// record code running time
		auto stop_evolution_time = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop_evolution_time - start_evolution_time);
		std::cout << "num of time steps: " << num_time_step
			<< "; time step: " << dt
			<< "; curr time: " << curr_time
			<< "; running time: " << duration.count() / 1e6 << " seconds"
			<< std::endl;
	}

	std::cout << "--- evolution finished ---" << std::endl;

	// calculate error between numerical solution and exact solution
	std::cout << "calculating error at final time" << std::endl;
	
	// exact solution
	// |u|^2 * u + |u|^4 * u
	// u(x,y,t) = exp(i*(2*pi*(x+y)-(4d*pi^2-2)*t)) = cos(2*pi*(x+y)-4d*pi^2*t) + i * sin(2*pi*(x+y)-(4d*pi^2-2)*t)
	auto final_func1 = [=](std::vector<double> x) {return cos(2.*Const::PI*(x[0]+x[1]) - (DIM * pow(2.*Const::PI, 2.)-2)*final_time); };
	auto final_func2 = [=](std::vector<double> x) {return sin(2.*Const::PI*(x[0]+x[1]) - (DIM * pow(2.*Const::PI, 2.)-2)*final_time); };
	// |u|^2 * u
	// auto final_func1 = [=](std::vector<double> x) {return cos(2.*Const::PI*(x[0]+x[1]) - (DIM * pow(2.*Const::PI, 2.)-1)*final_time); };
	// auto final_func2 = [=](std::vector<double> x) {return sin(2.*Const::PI*(x[0]+x[1]) - (DIM * pow(2.*Const::PI, 2.)-1)*final_time); };
	// zero source
	// auto final_func1 = [=](std::vector<double> x) {return cos(2.*Const::PI*(x[0]+x[1]) - (DIM * pow(2.*Const::PI, 2.))*final_time); };
	// auto final_func2 = [=](std::vector<double> x) {return sin(2.*Const::PI*(x[0]+x[1]) - (DIM * pow(2.*Const::PI, 2.))*final_time); };	
	std::vector<std::function<double(std::vector<double>)>> final_func{ final_func1, final_func2 };

	const int num_gauss_pt = 5;
	std::vector<double> err = dg_solu.get_error_no_separable_system(final_func, num_gauss_pt);
	std::cout << "L1, L2 and Linf error at final time: " << err[0] << ", " << err[1] << ", " << err[2] << std::endl;

	IO inout(dg_solu);
	std::string file_name = "error_N" + std::to_string(NMAX) + ".txt";
	inout.write_error(NMAX, err, file_name);
	// ------------------------------

	return 0;
}
