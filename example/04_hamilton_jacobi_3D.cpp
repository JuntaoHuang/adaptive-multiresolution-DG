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
#include "HamiltonJacobi.h"
#include <omp.h>

// For this example, Hamilton-Jacobi equations will be considered.
// the LDG formulation by Yan and Osher will be used


double Hamiltonian(const std::vector<double> & u, const int problem, const std::vector<double> & x);




std::vector<std::vector<std::function<double(double, int)>>> set_initial_condition_seperable(int DIM, int problem);

int main(int argc, char *argv[])
{
	omp_set_dynamic(0);
	omp_set_nested(2);
	Element::DIM = 3;			// dimension

	const int problem = 2;


	// static variables
	AlptBasis::PMAX = 3;

	LagrBasis::PMAX = 5;
	LagrBasis::msh_case = (LagrBasis::PMAX == 3) ? 3 : 2;
	if (LagrBasis::PMAX == 5) LagrBasis::msh_case = 1; // interface nodes

	HermBasis::PMAX = 3;
	HermBasis::msh_case = 3;

	Element::PMAX_alpt = AlptBasis::PMAX;	// max polynomial degree for Alpert's basis functions
	Element::PMAX_intp = LagrBasis::PMAX;	// max polynomial degree for interpolation basis functions
	Element::VEC_NUM = 2 * Element::DIM + 1;		// num of unknown variables needed for HJ is 2d+1;
													// the last variable denotes the solution phi

	DGSolution::DIM = Element::DIM;
	DGSolution::VEC_NUM = Element::VEC_NUM;
	DGSolution::ind_var_vec = { 0 };
	DGAdapt::indicator_var_adapt = { 0 }; // index of the variable among VEC_NUM variables for the adaptive error indicator

	Interpolation::DIM = DGSolution::DIM;
	Interpolation::VEC_NUM = DGSolution::VEC_NUM;

	Element::is_intp.resize(Element::VEC_NUM);
	for (auto & u : Element::is_intp)
	{
		u.resize(Element::DIM);
		u.assign(u.size(), false);
	}
	Element::is_intp[0][0] = true; // for HJ

	// constant variable
	DGSolution::prob = "HJ";
	int DIM = Element::DIM;
	int VEC_NUM = Element::VEC_NUM;

	int NMAX = 3;
	int N_init = NMAX;
	int is_init_sparse = 2;			// use full grid (0) or sparse grid (1) when initialization
	const std::string boundary_type = "period";
	double final_time = 0.005;
	const int ierror = 1;
	const double cfl = 0.005;         // problem == 4, cfl = 0.02 
	const int rk_order = 3;
	const bool is_adapt_find_ptr_alpt = true;	// variable control if need to adaptively find out pointers related to Alpert basis in DG operators
	const bool is_adapt_find_ptr_intp = true;	// variable control if need to adaptively find out pointers related to interpolation basis in DG operators

	// adaptive parameter
	// if need to test code without adaptive, just set refine_eps large number 1e6, then no refine
	// and set coarsen_eta negative number -1, then no coarsen
	const double refine_eps = 1e10;
	const double refine_eps_ext = 1e-7;
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

	OperatorMatrix1D<LagrBasis, LagrBasis> oper_matx_lagr_all(all_bas_lagr, all_bas_lagr, boundary_type);

	OperatorMatrix1D<HermBasis, HermBasis> oper_matx_herm_all(all_bas_herm, all_bas_herm, boundary_type);
	// initial condition
	// project initial function into numerical solution
	//std::function<double(double, int)> init_func_1 = [](double x, int d) { return (d == 0) ? -0.5/Const::PI*(cos(2.*Const::PI*x)) : cos(2.*Const::PI*x); };
	//std::function<double(double, int)> init_func_2 = [](double x, int d) { return (d == 0) ? 0.5 /Const::PI*(sin(2.*Const::PI*x)) : sin(2.*Const::PI*x); };
	//std::vector<std::function<double(double, int)>> init_func1{ { init_func_1, init_func_2}};

	//std::vector<std::vector<std::function<double(double, int)>>> init_func{ { init_func1} };


	// initialization of DG solution
	DGAdaptIntp dg_solu(sparse, N_init, NMAX, all_bas_alpt, all_bas_lagr, all_bas_herm, hash, refine_eps, coarsen_eta, is_adapt_find_ptr_alpt, is_adapt_find_ptr_intp, oper_matx_lagr_all, oper_matx_herm_all);
	// project initial function into numerical solution
	// for HJ equations, the first vari stores phi
	dg_solu.init_separable_system_sum(set_initial_condition_seperable(DIM, problem));

	// ------------------------------
	// 	This part is for solving constant coefficient for linear system	
	// here hyperbolicConst will a total of DIM VEC_NUM*VEC_NUM matrices

	//VecMultiD<double> hyperbolicConst(3, {DIM, DGSolution::VEC_NUM, DGSolution::VEC_NUM });
	//std::vector<std::vector<std::vector<double>>> hyperbolicConst = { {{-1, 0}, {0,1}}, {{0, -1}, {-1,0}} };
	//hyperbolicConst.resize(DIM);
	//for (auto i = 0; i < DIM; ++i)
	//{
	//	hyperbolicConst[i].resize(DGSolution::VEC_NUM);
	//	for(auto j=0; j< DGSolution::VEC_NUM; ++j)
	//		hyperbolicConst[i][j].resize(DGSolution::VEC_NUM);
	//}

	const int max_mesh = dg_solu.max_mesh_level();
	const double dx = 1. / pow(2., max_mesh);
	double dt = dx * cfl;
	int total_time_step = ceil(final_time / dt) + 1;
	dt = final_time / total_time_step;

	std::vector<HyperbolicAlpt> grad_linear(2 * DIM, HyperbolicAlpt(dg_solu, oper_matx_alpt, 1));

	omp_set_num_threads(2 * DIM);
#pragma omp parallel for
	for (int d = 0; d < 2 * DIM; ++d) // 2 * DIM matrices for computing the gradient of phi via LDG
	{
		int sign = 2 * (d % 2) - 1;
		int dd = d / 2;
		grad_linear[d].assemble_matrix_flx_scalar(dd, sign, -1);
		grad_linear[d].assemble_matrix_vol_scalar(dd, -1);
		//grad_linear[2 * d].assemble_matrix_flx_scalar(d, -1, -1);
		//grad_linear[2 * d + 1].assemble_matrix_flx_scalar(d, 1, -1);
		//grad_linear[2 * d].assemble_matrix_vol_scalar(d, -1);
		//grad_linear[2 * d + 1].assemble_matrix_vol_scalar(d, -1);
	}

	HamiltonJacobiLDG<HyperbolicAlpt>  HJ(grad_linear, DIM);


	std::vector< std::vector<bool> > is_intp(VEC_NUM, std::vector<bool>(DIM, false));

	is_intp[0][0] = true;


	std::vector<double> alpha(DIM, 1.); // how about general case

	auto LFHamiltonian = [=](const std::vector<double> & u, int i, int d, const std::vector<double> & x = std::vector<double>()) 
	{
		double jump = 0;
		static std::vector<double> ave(DIM);
		for (auto i = 0; i < DIM; i++) {
			jump += alpha[i] * (u[2 * i + 2] - u[2 * i + 1]);
			ave[i] = 0.5 * (u[2 * i + 1] + u[2 * i + 2]);
		}

		// negative LF Hamiltonian 
		auto sign = [](double x) -> double {return ((x > 0.) - (x < 0.)); };
		if (problem == 3) {
			double _2pi = 2.*Const::PI;
			return -(-sin(_2pi*x[1]) * ave[0] + (-sin(_2pi*x[0]) + sign(u[3]))*ave[1] - 0.5 * pow(sin(_2pi*x[1]), 2) - cos(_2pi*x[0]) - 1.) + jump;
		}
		return -Hamiltonian(ave, problem, x) + jump;
	
	};



	LagrInterpolation interp(dg_solu);

	FastLagrIntp fastintp(dg_solu, interp.Lag_pt_Alpt_1D, interp.Lag_pt_Alpt_1D_d1);

	FastRHSHamiltonJacobi fastRHShj(dg_solu, oper_matx_lagr);

	double curr_time = 0;
	for (size_t num_time_step = 0; num_time_step < total_time_step; num_time_step++)
	{
		auto start_evolution_time = std::chrono::high_resolution_clock::now();

		RK3SSP odeSolver(dg_solu, dt);
		odeSolver.init_HJ(HJ);

		for (size_t num_stage = 0; num_stage < odeSolver.num_stage; num_stage++)
		{
			dg_solu.set_rhs_zero();

			odeSolver.compute_gradient_HJ(HJ); // compute the gradient of phi

			// interp.nonlinear_Lagr(LFHamiltonian, is_intp); // interpolate the LFHamiltonian
			if (problem != 3)
				interp.nonlinear_Lagr_fast(LFHamiltonian, is_intp, fastintp); // interpolate the LFHamiltonian
			else
				interp.nonlinear_Lagr_fast_full(LFHamiltonian, is_intp, fastintp); // interpolate the LFHamiltonian



			fastRHShj.rhs_nonlinear();

			odeSolver.rhs_to_eigenvec("HJ");

			odeSolver.step_stage(num_stage);

			odeSolver.final_HJ(HJ);
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

	// calculate error between numerical solution and exact solution
	std::cout << "calculating error at final time" << std::endl;


	HJexact* HJExact;
	if (problem == 1)
		HJExact = new HJBurgersExact;
	else if (problem == 2)
		HJExact = new HJCosExact;
	else if (problem == 4)
		HJExact = new HJNonlinearExact;
	//HJCosExact HJExact;
	auto final_func1 = [&](std::vector<double> x, int i) {return  HJExact->exact_3d(x, curr_time); };
	auto final_func2 = [&](std::vector<double> x) {return  HJExact->exact_3d(x, curr_time); };


	IO inout(dg_solu);

	if (ierror == 1)
	{


		DGAdaptIntp dg_solu_ext(sparse, N_init, NMAX, all_bas_alpt, all_bas_lagr, all_bas_herm, hash, refine_eps_ext, coarsen_eta, is_adapt_find_ptr_alpt, is_adapt_find_ptr_intp, oper_matx_lagr_all, oper_matx_herm_all);
		LagrInterpolation interp_ext(dg_solu_ext);

		dg_solu_ext.init_adaptive_intp_Lag(final_func1, interp_ext);

		FastLagrInit  fastLagr_init_ext(dg_solu_ext, oper_matx_lagr);

		fastLagr_init_ext.eval_ucoe_Alpt_Lagr();

		//const int num_gauss_pt = 3;
		std::vector<double> err(3, 0.);
		std::cout << "compute L2 error via projection " << std::endl;
		//std::vector<double> err2 = dg_solu.get_error_no_separable_system_omp(final_func2, num_gauss_pt, 0);
		err[1] = dg_solu.get_L2_error_split_adaptive_intp_scalar(dg_solu_ext);
		std::string file_name = "error_N" + std::to_string(NMAX) + ".txt";
		inout.write_error(NMAX, err, file_name);
		std::cout << "L1, L2 and Linf error at final time: " << err[0] << ", " << err[1] << ", " << err[2] << std::endl;
		//std::cout << "L1, L2 and Linf error at final time: " << err2[0] << ", " << err2[1] << ", " << err2[2] << std::endl;
		//inout.output_num_exa("exact.txt", final_func1);
	}

	inout.output_num_cut_2D("numerical.txt", 1e-10, 2);
	inout.output_element_center("element.txt");
	// ------------------------------

	return 0;
}




std::vector<std::vector<std::function<double(double, int)>>> set_initial_condition_seperable(int DIM, int problem)
{
	if (problem == 3 && DIM == 2)
	{
		std::function<double(double, int)> init_func_1 = [](double x, int d) { return 0.; };

		std::vector<std::function<double(double, int)>> init_func1{ { init_func_1} };

		return std::vector<std::vector<std::function<double(double, int)>>>{ { init_func1} };

	}
	if (problem == 4 && DIM == 2)
	{
		std::function<double(double, int)> init_func_1 = [](double x, int d) { return (d == 0) ? -0.5 / Const::PI*sin(2.*Const::PI*x) : 1.; };;
		std::function<double(double, int)> init_func_2 = [](double x, int d) { return (d == 0) ? 1. : -0.5 / Const::PI*cos(2.*Const::PI*x); };

		std::vector<std::function<double(double, int)>> init_func1{ { init_func_1, init_func_2} };

		return std::vector<std::vector<std::function<double(double, int)>>>{ { init_func1} };

	}
	if (DIM == 2) {
		std::function<double(double, int)> init_func_1 = [](double x, int d) { return (d == 0) ? -0.5 / Const::PI*(cos(2.*Const::PI*x)) : cos(2.*Const::PI*x); };
		std::function<double(double, int)> init_func_2 = [](double x, int d) { return (d == 0) ? 0.5 / Const::PI*(sin(2.*Const::PI*x)) : sin(2.*Const::PI*x); };

		std::vector<std::function<double(double, int)>> init_func1{ { init_func_1, init_func_2} };

		return std::vector<std::vector<std::function<double(double, int)>>>{ { init_func1} };
	}

	else if (DIM == 3) {
		std::function<double(double, int)> init_func_1 = [](double x, int d) { return (d == 0) ? -0.5 / Const::PI*(cos(2.*Const::PI*x)) : cos(2.*Const::PI*x); };
		std::function<double(double, int)> init_func_2 = [](double x, int d) { return (d == 2) ? 0.5 / Const::PI*(cos(2.*Const::PI*x)) : sin(2.*Const::PI*x); };
		std::function<double(double, int)> init_func_3 = [](double x, int d) { return (d == 1) ? 0.5 / Const::PI*(cos(2.*Const::PI*x)) : sin(2.*Const::PI*x); };
		std::function<double(double, int)> init_func_4 = [](double x, int d) { return (d == 0) ? 0.5 / Const::PI*(cos(2.*Const::PI*x)) : sin(2.*Const::PI*x); };

		std::vector<std::function<double(double, int)>> init_func1{ { init_func_1, init_func_2,  init_func_3, init_func_4} };

		return std::vector<std::vector<std::function<double(double, int)>>>{ { init_func1} };
	}

	assert("error, initialization setup");

}


double Hamiltonian(const std::vector<double> & u, const int problem, const std::vector<double> & x)
{
	double sum = std::accumulate(u.begin(), u.end(), 0.);

	// problem == 1: burgers equation
	if (problem == 1)
	{
		return 0.5*sum*sum;

	}
	else if (problem == 2)
	{

		return -cos(sum + 1);
	}
	else if (problem == 4)
	{
		return u[0] * u[1];
	}
	auto sign = [](double x) -> double {return ((x > 0.) - (x < 0.)); };
	static double inv2pi = 1. / (2. * Const::PI);
	static double _2pi = (2. * Const::PI);

	if (problem == 3)
		return -sin(_2pi*x[1]) * u[0] + (-sin(_2pi*x[0]) + sign(u[1]))*u[1] - 0.5 * pow(sin(_2pi*x[1]), 2) - cos(_2pi*x[0]) - 1.;
}



