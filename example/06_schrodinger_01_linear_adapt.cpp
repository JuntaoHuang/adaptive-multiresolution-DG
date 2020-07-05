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

double sech(const double x) { return 1./cosh(x); }

namespace prob3
{
	const double M = 30.;
}

namespace prob4
{
	const double x1 = -10.;
	const double x2 = 10.;
	const double c1 = 4.;
	const double c2 = -4.;
	const double M = 50.;
}

namespace prob5
{
	const double M = 90.;
	const double A = 1.78;
}

int main(int argc, char *argv[])
{
	// problem
	// 0: 1D accuracy test without source: 
	// 		iu_t + u_xx = 0
	// 1: 1D accuracy test with nonlinear source: 
	// 		iu_t + u_xx + (|u|^2 + |u|^4) * u = 0
	// 2: 2D accuracy test
	// 		iu_t + u_xx + u_yy + (|u|^2 + |u|^4) * u = 0
	// 3: 1D single soliton, example 3.2 in Xu and Shu, 2005
	// 4: 1D double soliton, example 3.3 in Xu and Shu, 2005
	// 5: birth of standing soliton, example 3.4 in Xu and Shu, 2005
	// 6: birth of moving soliton, example 3.4 in Xu and Shu, 2005
	int problem = 3;

	// static variables
	AlptBasis::PMAX = 3;

	LagrBasis::PMAX = AlptBasis::PMAX;
	LagrBasis::msh_case = 1;

	HermBasis::PMAX = 3;
	HermBasis::msh_case = 1;

	Element::PMAX_alpt = AlptBasis::PMAX;	// max polynomial degree for Alpert's basis functions
	Element::PMAX_intp = LagrBasis::PMAX;	// max polynomial degree for interpolation basis functions
	Element::DIM = 1;			// dimension
	if (problem == 2) { Element::DIM = 2; };
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
	bool sparse = false;			// use full grid (0) or sparse grid (1) when initialization
	const std::string boundary_type = "period";
	double final_time = 0.03;
	double cfl = 0.1;
	const bool is_adapt_find_ptr_alpt = true;	// variable control if need to adaptively find out pointers related to Alpert basis in DG operators
	const bool is_adapt_find_ptr_intp = false;	// variable control if need to adaptively find out pointers related to interpolation basis in DG operators
	std::vector<double> output_time = {0, final_time};
	if (problem == 3) { final_time = 2.0; output_time = linspace(0.1, 3.0, 30); }
	if (problem == 4) { final_time = 5.0; output_time = linspace(0.1, 5.0, 50); }
	if (problem == 5 || problem == 6) { final_time = 4.0; output_time = linspace(0.1, 4.0, 40); }

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

	AllBasis<LagrBasis> all_bas_lagr(NMAX);
	AllBasis<HermBasis> all_bas_herm(NMAX);
	AllBasis<AlptBasis> all_bas_alpt(NMAX);

	// operator matrix for Alpert basis
	OperatorMatrix1D<AlptBasis, AlptBasis> oper_matx_alpt(all_bas_alpt, all_bas_alpt, boundary_type);
	OperatorMatrix1D<HermBasis, AlptBasis> oper_matx_herm(all_bas_herm, all_bas_alpt, boundary_type);
	OperatorMatrix1D<LagrBasis, AlptBasis> oper_matx_lagr(all_bas_lagr, all_bas_alpt, boundary_type);

	auto func_source = [&](std::vector<double> u, int i, int d)->double
		{
			if (problem == 0) { return 0.; }

			double u2 = u[0]*u[0]+u[1]*u[1];
			if (problem == 1 || problem == 2)
			{
				// |u|^2 * u + |u|^4 * u
				if (i==0) { return -(u2 + u2 * u2)*u[1]; }
				else if (i==1) { return (u2 + u2 * u2)*u[0]; }
			}
			else if (problem == 3 || problem == 4)
			{			
				// 2 * |u|^2 * u
				if (i==0) { return -2*(u2)*u[1]; }
				else if (i==1) { return 2*(u2)*u[0]; }
			}
		};

	// initial condition
	// problem 1: u(x, y) = exp(i*2*pi*x) = cos(2*pi*x) + i * sin(2*pi*x)
	// problem 2: u(x, y) = exp(i*2*pi*(x+y)) = cos(2*pi*(x+y)) + i * sin(2*pi*(x+y))
	std::function<double(double, int)> init_func_1 = [](double x, int d) { return 0.; };
	std::function<double(double, int)> init_func_2 = [](double x, int d) { return 0.; };
	std::function<double(double, int)> init_func_3 = [](double x, int d) { return 0.; };
	std::function<double(double, int)> init_func_4 = [](double x, int d) { return 0.; };
	// initial condition for real part of u	
	if (problem == 0 || problem == 1)
	{
		init_func_1 = [](double x, int d) { return cos(2.*Const::PI*x); };
	}
	else if (problem == 2)
	{
		init_func_1 = [](double x, int d) { return (d == 0) ? (cos(2.*Const::PI*x)) : (cos(2.*Const::PI*x)); };
		init_func_2 = [](double x, int d) { return (d == 0) ? -(sin(2.*Const::PI*x)) : (sin(2.*Const::PI*x)); };
	}	
	else if (problem == 3)
	{		
		init_func_1 = [&](double x, int d) { return sech(prob3::M*(x - 0.5)) * cos(2*prob3::M*(x - 0.5)); };
	}
	else if (problem == 4)
	{
		init_func_1 = [&](double x, int d)->double
			{
				double x_shift = prob4::M*(x-0.5);
				return sech(x_shift-prob4::x1) * cos(0.5*prob4::c1*(x_shift-prob4::x1)) + sech(x_shift-prob4::x2) * cos(0.5*prob4::c2*(x_shift-prob4::x2));
			};
	}
	else if (problem == 5)
	{
		init_func_1 = [&](double x, int d)->double
			{
				double x_shift = prob5::M*(x-0.5);
				return prob5::A * exp(-std::pow(x_shift, 2.));
			};
	}
	else if (problem == 6)
	{
		init_func_1 = [&](double x, int d)->double
			{
				double x_shift = prob5::M*(x-0.5);
				return prob5::A * exp(-std::pow(x_shift, 2.)) * cos(2*x_shift);
			};
	}	
	std::vector<std::function<double(double, int)>> init_func1{ init_func_1, init_func_2, init_func_3, init_func_4 };

	// initial condition for imaginary part of u
	if (problem == 0 || problem == 1)
	{
		init_func_1 = [](double x, int d) { return sin(2.*Const::PI*x); };
	}
	else if (problem == 2)
	{
		init_func_1 = [](double x, int d) { return (d == 0) ? (sin(2.*Const::PI*x)) : (cos(2.*Const::PI*x)); };
		init_func_2 = [](double x, int d) { return (d == 0) ? (cos(2.*Const::PI*x)) : (sin(2.*Const::PI*x)); };
	}	
	else if (problem == 3)
	{
		init_func_1 = [&](double x, int d) { return sech(prob3::M*(x - 0.5)) * sin(2*prob3::M*(x - 0.5)); };
	}
	else if (problem == 4)
	{
		init_func_1 = [&](double x, int d)->double
			{
				double x_shift = prob4::M*(x-0.5);
				return sech(x_shift-prob4::x1) * sin(0.5*prob4::c1*(x_shift-prob4::x1)) + sech(x_shift-prob4::x2) * sin(0.5*prob4::c2*(x_shift-prob4::x2));
			};
	}	
	else if (problem == 5)
	{
		init_func_1 = [&](double x, int d) { return 0.; };
	}	
	else if (problem == 6)
	{
		init_func_1 = [&](double x, int d)->double
			{
				double x_shift = prob5::M*(x-0.5);
				return prob5::A * exp(-std::pow(x_shift, 2.)) * sin(2*x_shift);
			};
	}		
	std::vector<std::function<double(double, int)>> init_func2{ init_func_1, init_func_2, init_func_3, init_func_4 };
		
	std::vector<std::vector<std::function<double(double, int)>>> init_func{ init_func1, init_func2};

	// initialization of DG solution
	DGAdapt dg_solu(sparse, N_init, NMAX, all_bas_alpt, all_bas_lagr, all_bas_herm, hash, refine_eps, coarsen_eta, is_adapt_find_ptr_alpt, is_adapt_find_ptr_intp);

	// project initial function into numerical solution
	dg_solu.init_separable_system_sum(init_func);

	dg_solu.coarsen();
	std::cout << "total num of basis (initial): " << dg_solu.size_basis_alpt() << std::endl;

	IO inout(dg_solu);
	inout.output_num("profile1D_init.txt");
	inout.output_element_level_support("suppt1D_init.txt");

	LagrInterpolation interp_lagr(dg_solu);
	FastLagrIntp fast_lagr_intp(dg_solu, interp_lagr.Lag_pt_Alpt_1D, interp_lagr.Lag_pt_Alpt_1D_d1);	

	SourceFastLagr fast_source_lagr(dg_solu, oper_matx_lagr);

	// only interpolate in dimension 0
	std::vector< std::vector<bool> > is_intp_lagr;
	for (int i = 0; i < VEC_NUM; i++) 
	{
		std::vector<bool> true_1st(DIM, false);
		true_1st[0] = true;
		is_intp_lagr.push_back(true_1st);
	}

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

			SchrodingerAlpt dg_operator(dg_solu, oper_matx_alpt);
			if (problem == 3) { dg_operator.assemble_matrix(1./pow(prob3::M, 2.)); }
			else if (problem == 4) { dg_operator.assemble_matrix(1./pow(prob4::M, 2.)); }
			else if (problem == 5 || problem == 6) { dg_operator.assemble_matrix(1./pow(prob5::M, 2.)); }
			else { dg_operator.assemble_matrix(); }

			IMEXEuler odesolver(dg_operator, dt, "sparselu");
			odesolver.init();

			interp_lagr.nonlinear_Lagr_fast(func_source, is_intp_lagr, fast_lagr_intp);
			
			// calculate rhs and update Element::ucoe_alpt
			dg_solu.set_rhs_zero();

			// nonlinear source terms
			fast_source_lagr.rhs_source();

			// add to rhs in ode solver
			odesolver.set_rhs_zero();
			odesolver.add_rhs_to_eigenvec();

			odesolver.step_stage(0);

			odesolver.final();
		}

		// --- part 3: refine base on Element::ucoe_alpt
		dg_solu.refine();
		const int num_basis_refine = dg_solu.size_basis_alpt();

		// after refine, copy Element::ucoe_alpt_predict back to Element::ucoe_alpt
		dg_solu.copy_predict_to_ucoe();
	
		// --- part 4: time evolution
		SchrodingerAlpt dg_operator(dg_solu, oper_matx_alpt);
		if (problem == 3) { dg_operator.assemble_matrix(1./pow(prob3::M, 2.)); }
		else if (problem == 4) { dg_operator.assemble_matrix(1./pow(prob4::M, 2.)); }
		else if (problem == 5 || problem == 6) { dg_operator.assemble_matrix(1./pow(prob5::M, 2.)); }
		else { dg_operator.assemble_matrix(); }

		IMEX43 odesolver(dg_operator, dt, "sparselu");
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

		// --- part 5: coarsen
		dg_solu.coarsen();
		const int num_basis_coarsen = dg_solu.size_basis_alpt();

		// add current time and increase time steps
		curr_time += dt;		
		num_time_step ++;

		if (num_time_step % 100 == 0)
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
					<< "; curr max mesh level: " << dg_solu.max_mesh_level()
					<< std::endl << std::endl;

			// output error in evolution
			std::function<double(std::vector<double> x)> final_func1 = [](std::vector<double> x) { return 0.; };
			std::function<double(std::vector<double> x)> final_func2 = [](std::vector<double> x) { return 0.; };
			if (problem == 0)
			{		
				final_func1 = [&](std::vector<double> x) {return cos(2.*Const::PI*x[0] - (pow(2.*Const::PI, 2.))*curr_time); };
				final_func2 = [&](std::vector<double> x) {return sin(2.*Const::PI*x[0] - (pow(2.*Const::PI, 2.))*curr_time); };
			}
			else if (problem == 1)
			{		
				final_func1 = [&](std::vector<double> x) {return cos(2.*Const::PI*x[0] - (pow(2.*Const::PI, 2.)-2)*curr_time); };
				final_func2 = [&](std::vector<double> x) {return sin(2.*Const::PI*x[0] - (pow(2.*Const::PI, 2.)-2)*curr_time); };
			}
			else if (problem == 2)
			{		
				final_func1 = [&](std::vector<double> x) {return cos(2.*Const::PI*(x[0]+x[1]) - (DIM * pow(2.*Const::PI, 2.)-2)*curr_time); };
				final_func2 = [&](std::vector<double> x) {return sin(2.*Const::PI*(x[0]+x[1]) - (DIM * pow(2.*Const::PI, 2.)-2)*curr_time); };
			}
			else if (problem == 3)
			{
				// since solution is periodic, we make it periodic in the range of 0 < x < 1
				final_func1 = [&](std::vector<double> x) 
					{
						double x_period = x[0];
						x_period = std::fmod(x_period, 1.);
						if (x_period<0) x_period = std::fmod(x_period+1, 1.);
						return sech(prob3::M*(x_period-0.5)-4*curr_time)*cos(2*(prob3::M*(x_period-0.5)-1.5*curr_time)); 
					};
				final_func2 = [&](std::vector<double> x) 
					{
						double x_period = x[0];
						x_period = std::fmod(x_period, 1.);
						if (x_period<0) x_period = std::fmod(x_period+1, 1.);
						return sech(prob3::M*(x_period-0.5)-4*curr_time)*sin(2*(prob3::M*(x_period-0.5)-1.5*curr_time)); 
					};
			}	
			std::vector<std::function<double(std::vector<double>)>> final_func{ final_func1, final_func2 };

			if (problem == 0 || problem == 1 || problem == 2 || problem == 3)
			{
				const int num_gauss_pt = 5;
				std::vector<double> err = dg_solu.get_error_no_separable_system(final_func, num_gauss_pt);
				std::cout << "L1, L2 and Linf error time: " << curr_time << std::endl << err[0] << " " << err[1] << " " << err[2] << std::endl;
			}
		}

		auto min_time_iter = std::min_element(output_time.begin(), output_time.end());
		if ((curr_time <= *min_time_iter) && (curr_time+dt >= *min_time_iter))
		{
			inout.output_num("profile1D_" + std::to_string(curr_time) + ".txt");
			
			inout.output_element_level_support("suppt1D_" + std::to_string(curr_time) + ".txt");

			output_time.erase(min_time_iter);
		}		
	}

	std::cout << "--- evolution finished ---" << std::endl;
	std::cout << "total num of basis (last time step): " << dg_solu.size_basis_alpt() << std::endl;

	// calculate error between numerical solution and exact solution
	std::cout << "calculating error at final time" << std::endl;
	
	// exact solution
	// problem 1: u(x,y,t) = exp(i*(2*pi*x-(4*pi^2-2)*t)) = cos(2*pi*x-(4*pi^2-2)*t) + i * sin(2*pi*x-(4*pi^2-2)*t)
	// problem 2: u(x,y,t) = exp(i*(2*pi*(x+y)-(4d*pi^2-2)*t)) = cos(2*pi*(x+y)-4d*pi^2*t) + i * sin(2*pi*(x+y)-(4d*pi^2-2)*t)
	std::function<double(std::vector<double> x)> final_func1 = [](std::vector<double> x) { return 0.; };
	std::function<double(std::vector<double> x)> final_func2 = [](std::vector<double> x) { return 0.; };
	if (problem == 0)
	{		
		final_func1 = [&](std::vector<double> x) {return cos(2.*Const::PI*x[0] - (pow(2.*Const::PI, 2.))*final_time); };
		final_func2 = [&](std::vector<double> x) {return sin(2.*Const::PI*x[0] - (pow(2.*Const::PI, 2.))*final_time); };
	}
	else if (problem == 1)
	{		
		final_func1 = [&](std::vector<double> x) {return cos(2.*Const::PI*x[0] - (pow(2.*Const::PI, 2.)-2)*final_time); };
		final_func2 = [&](std::vector<double> x) {return sin(2.*Const::PI*x[0] - (pow(2.*Const::PI, 2.)-2)*final_time); };
	}
	else if (problem == 2)
	{		
		final_func1 = [&](std::vector<double> x) {return cos(2.*Const::PI*(x[0]+x[1]) - (DIM * pow(2.*Const::PI, 2.)-2)*final_time); };
		final_func2 = [&](std::vector<double> x) {return sin(2.*Const::PI*(x[0]+x[1]) - (DIM * pow(2.*Const::PI, 2.)-2)*final_time); };
	}
	else if (problem == 3)
	{
		// since solution is periodic, we make it periodic in the range of 0 < x < 1	
		final_func1 = [&](std::vector<double> x) 
			{
				double x_period = x[0];
				x_period = std::fmod(x_period, 1.);
				if (x_period<0) x_period = std::fmod(x_period+1, 1.);
				return sech(prob3::M*(x_period-0.5)-4*final_time)*cos(2*(prob3::M*(x_period-0.5)-1.5*final_time)); 
			};
		final_func2 = [&](std::vector<double> x) 
			{
				double x_period = x[0];
				x_period = std::fmod(x_period, 1.);
				if (x_period<0) x_period = std::fmod(x_period+1, 1.);
				return sech(prob3::M*(x_period-0.5)-4*final_time)*sin(2*(prob3::M*(x_period-0.5)-1.5*final_time)); 
			};
	}
	std::vector<std::function<double(std::vector<double>)>> final_func{ final_func1, final_func2 };

	if (problem == 0 || problem == 1 || problem == 2 || problem == 3)
	{
		const int num_gauss_pt = 5;
		std::vector<double> err = dg_solu.get_error_no_separable_system(final_func, num_gauss_pt);
		std::cout << "L1, L2 and Linf error at final time: " << err[0] << " " << err[1] << " " << err[2] << std::endl;

		std::vector<std::vector<double>> err_each;
		for (int num_var = 0; num_var < VEC_NUM; num_var++)
		{
			std::vector<double> err = dg_solu.get_error_no_separable_system_each(final_func, num_gauss_pt, num_var);
			std::cout << "L1, L2 and Linf error of " << num_var << " variable at final time: " << err[0] << " " << err[1] << " " << err[2] << std::endl;
			err_each.push_back(err);
		}

		inout.write_error(NMAX, err, "error_N" + to_str(NMAX) + ".txt");
		// inout.write_error_eps_dof(refine_eps, dg_solu.size_basis_alpt(), err_each, "error_eps" + to_str(refine_eps) + ".txt");
	
		double refine_eps1 = 1/refine_eps;

		inout.write_error_eps_dof(refine_eps, dg_solu.size_basis_alpt(), err_each, "error_eps" + to_str(refine_eps1) + ".txt");
	}

	inout.output_num("profile1D_final.txt");
	inout.output_element_level_support("suppt1D_final.txt");
	// ------------------------------

	return 0;
}
