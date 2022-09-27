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


// this example solve linear hyperbolic system with constant coefficients
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
	int NMAX = 6;
	int N_init = NMAX;
	int is_init_sparse = 1;			// use full grid (0) or sparse grid (1) when initialization
	const std::string boundary_type = "period";
	double final_time = 0.2;
	const double cfl = 0.1;
	const bool is_adapt_find_ptr_alpt = true;	// variable control if need to adaptively find out pointers related to Alpert basis in DG operators
	const bool is_adapt_find_ptr_intp = true;	// variable control if need to adaptively find out pointers related to interpolation basis in DG operators

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
	bool sparse = ((is_init_sparse == 1) ? true : false);

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

	// initial condition
	// project initial function into numerical solution
	// u(x,y) = 1/(2*sqrt(2)) * (sin(2*pi*x) * cos(2*pi*y) + cos(2*pi*x) * sin(2*pi*y) 
	//		  - cos(2*pi*x) * cos(2*pi*y) + sin(2*pi*x) * sin(2*pi*y))
	// v(x,y) = (sqrt(2)-1)/(2*sqrt(2)) * (sin(2*pi*x) * cos(2*pi*y) + cos(2*pi*x) * sin(2*pi*y)) 
	//		  + (sqrt(2)+1)/(2*sqrt(2)) * (cos(2*pi*x) * cos(2*pi*y) - sin(2*pi*x) * sin(2*pi*y))
	std::function<double(double, int)> init_func_1 = [](double x, int d) { return (d == 0) ? 0.5 / sqrt(2.)*(sin(2.*Const::PI*x)) : cos(2.*Const::PI*x); };
	std::function<double(double, int)> init_func_2 = [](double x, int d) { return (d == 0) ? 0.5 / sqrt(2.)*(cos(2.*Const::PI*x)) : sin(2.*Const::PI*x); };
	std::function<double(double, int)> init_func_3 = [](double x, int d) { return (d == 0) ? -0.5 / sqrt(2.)*(cos(2.*Const::PI*x)) : cos(2.*Const::PI*x); };
	std::function<double(double, int)> init_func_4 = [](double x, int d) { return (d == 0) ? 0.5 / sqrt(2.)*(sin(2.*Const::PI*x)) : sin(2.*Const::PI*x); };
	std::vector<std::function<double(double, int)>> init_func1{ init_func_1, init_func_2, init_func_3, init_func_4 };

	init_func_1 = [](double x, int d) { return (d == 0) ? 0.5*(sqrt(2.) - 1) / sqrt(2.)*(sin(2.*Const::PI*x)) : cos(2.*Const::PI*x); };
	init_func_2 = [](double x, int d) { return (d == 0) ? 0.5*(sqrt(2.) - 1) / sqrt(2.)*(cos(2.*Const::PI*x)) : sin(2.*Const::PI*x); };
	init_func_3 = [](double x, int d) { return (d == 0) ? 0.5*(sqrt(2.) + 1) / sqrt(2.)*(cos(2.*Const::PI*x)) : cos(2.*Const::PI*x); };
	init_func_4 = [](double x, int d) { return (d == 0) ? -0.5*(sqrt(2.) + 1) / sqrt(2.)*(sin(2.*Const::PI*x)) : sin(2.*Const::PI*x); };
	std::vector<std::function<double(double, int)>> init_func2{ init_func_1, init_func_2, init_func_3, init_func_4 };
		
	std::vector<std::vector<std::function<double(double, int)>>> init_func{ init_func1, init_func2};

	// initialization of DG solution
	DGAdapt dg_solu(sparse, N_init, NMAX, all_bas_alpt, all_bas_lagr, all_bas_herm, hash, refine_eps, coarsen_eta, is_adapt_find_ptr_alpt, is_adapt_find_ptr_intp);

	// project initial function into numerical solution
	dg_solu.init_separable_system_sum(init_func);

	// ------------------------------
	// 	This part is for solving constant coefficient for linear system	
	// here hyperbolicConst will a total of DIM VEC_NUM*VEC_NUM matrices
	std::vector<std::vector<std::vector<double>>> hyperbolicConst = { {{-1, 0}, {0,1}}, {{0, -1}, {-1,0}} };

	const int max_mesh = dg_solu.max_mesh_level();
	const double dx = 1. / pow(2., max_mesh);
	double dt = dx * cfl;
	int total_time_step = ceil(final_time / dt) + 1;
	dt = final_time / total_time_step;

	HyperbolicAlpt linear(dg_solu, oper_matx_alpt);

	// volume integral
	linear.assemble_matrix_vol_system(0, hyperbolicConst[0]);
	linear.assemble_matrix_vol_system(1, hyperbolicConst[1]);

	// flux integral
	linear.assemble_matrix_flx_system(0, -1, hyperbolicConst[0], 0.5);
	linear.assemble_matrix_flx_system(0, 1,  hyperbolicConst[0], 0.5);
	linear.assemble_matrix_flx_system(1, -1, hyperbolicConst[1], 0.5);
	linear.assemble_matrix_flx_system(1, 1,  hyperbolicConst[1], 0.5);

	// flux integral (from penalty in Lax-Friedrich flux)
	linear.assemble_matrix_flx_system(0, -1, {1, 1}, 0.5);
	linear.assemble_matrix_flx_system(0, 1,  {1, 1}, -0.5);
	linear.assemble_matrix_flx_system(1, -1, {1, 1}, 0.5);
	linear.assemble_matrix_flx_system(1, 1,  {1, 1}, -0.5);

	RK3SSP odesolver(linear, dt);
	odesolver.init();

	std::cout << "--- evolution started ---" << std::endl;
	// record code running time
	auto start_evolution_time = std::chrono::high_resolution_clock::now();

	double curr_time = 0;
	for (size_t num_time_step = 0; num_time_step < total_time_step; num_time_step++)
	{
		odesolver.step_rk();
		curr_time += dt;

		// record code running time
		if (num_time_step % 100 == 0)
		{		
			auto stop_evolution_time = std::chrono::high_resolution_clock::now();
			auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop_evolution_time - start_evolution_time);
			std::cout << "num of time steps: " << num_time_step
				<< "; time step: " << dt
				<< "; curr time: " << curr_time
				<< "; running time: " << duration.count() / 1e6 << " seconds"
				<< std::endl;
		}
	}
	odesolver.final();

	std::cout << "--- evolution finished ---" << std::endl;

	// calculate error between numerical solution and exact solution
	std::cout << "calculating error at final time" << std::endl;
	
	const double sqrt2 = sqrt(2.);
	auto final_func1 = [=](std::vector<double> x) {return 0.5 / sqrt2 * (sin(2.*Const::PI*(x[0] + x[1] + sqrt2 * final_time))
		- cos(2.*Const::PI*(x[0] + x[1] - sqrt2 * final_time))); };
	auto final_func2 = [=](std::vector<double> x) {return (sqrt2 - 1.) * 0.5 / sqrt2 * (sin(2.*Const::PI*(x[0] + x[1] + sqrt2 * final_time))) + 
		+ (sqrt2 + 1.) * 0.5 / sqrt2 * cos(2.*Const::PI*(x[0] + x[1] - sqrt2 * final_time)); };


	std::vector<std::function<double(std::vector<double>)>> final_func{ final_func1, final_func2 };

	const int num_gauss_pt = 3;
	std::vector<double> err = dg_solu.get_error_no_separable_system(final_func, num_gauss_pt);
	std::cout << "L1, L2 and Linf error at final time: " << err[0] << ", " << err[1] << ", " << err[2] << std::endl;

	IO inout(dg_solu);
	std::string file_name = "error_N" + std::to_string(NMAX) + ".txt";
	inout.write_error(NMAX, err, file_name);
	// ------------------------------

	return 0;
}
