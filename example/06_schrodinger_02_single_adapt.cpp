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

//parameter for schrodinger equation
namespace schrodinger
{
	const double const_alpha = 0.0;
	const double const_beta = 18.0;

	const double const_M = 0.0;
	const double const_x = 0.0;

	// const double const_c = 0.0;
	// const double const_w1 = 0.0;
	// const double const_M1 = 0.0;

	// accuracy in 1D iu_t + i * alpha * u_x + u_xx + |u|^2 u + |u|^4 u = 0
	const double const_c = 2.0 * Const::PI;
	const double const_w1 = const_alpha * const_c + const_c * const_c - 2.0;
	const double const_M1 = 1.0;

	// bound state solution
	// const double const_M = 30.0;
	// const double const_x = 0.5;

	// birth of soliton
	// const double const_M = 90.0;
	// const double const_x = 0.5;

	// accuracy in 2D: iu_t + u_xx + u_yy + |u|^2 u + |u|^4 u = 0
	// const double const_c = 2.0 * Const::PI;
	// const double const_w1 = const_c * const_c + const_c * const_c - 2.0;
	// const double const_M1 = 1.0;

	// // singular solution in 2D: iu_t + u_xx/M^2 + u_yy/M^2 + |u|^2 u = 0
	// const double const_M = 2.0 * Const::PI;
	// const double const_x = 0.0;

	// // single blow up in 2D: iu_t + u_xx/M^2 + u_yy/M^2 + |u|^2 u = 0
	// const double const_M = 10.0;
	// const double const_x = 0.5;

	// 2nd blow up example in 2D: iu_t + u_xx/M^2 + u_yy/M^2 + |u|^2 u = 0
	// const double const_M = 2.0 * Const::PI;
	// const double const_x = 0.5;
}

double sech(const double x) { return 1./cosh(x); }

int main(int argc, char *argv[])
{
	// 0: 1D accuracy, 1: bound state solution, 2: birth of soliton
	// 3: 2D accuracy, 4: 2D singular, 5: 2D single blow up
	// 6: 2nd blow up example in 2D
	// when change problem, should also check the parameters in namespace::schrodinger
	int problem = 0; 

	int accuracy = 1; // 0: no accuracy; 1: compute error and order

	// static variables
	AlptBasis::PMAX = 3;

	LagrBasis::PMAX = AlptBasis::PMAX;
	LagrBasis::msh_case = 3;

	HermBasis::PMAX = 3;
	HermBasis::msh_case = 1;

	Element::PMAX_alpt = AlptBasis::PMAX;	// max polynomial degree for Alpert's basis functions
	Element::PMAX_intp = LagrBasis::PMAX;	// max polynomial degree for interpolation basis functions
	Element::DIM = 1;			// dimension
	// if (problem == 2) { Element::DIM = 2; };
	Element::VEC_NUM = 2;		// num of unknown variables in PDEs

	DGSolution::DIM = Element::DIM;
	DGSolution::VEC_NUM = Element::VEC_NUM;

	Interpolation::DIM = DGSolution::DIM;
	Interpolation::VEC_NUM = DGSolution::VEC_NUM;

	DGSolution::ind_var_vec = { 0, 1};
	DGAdapt::indicator_var_adapt = { 0, 1};

	Element::is_intp.resize(Element::VEC_NUM);
	for (size_t num = 0; num < Element::VEC_NUM; num++) { Element::is_intp[num] = std::vector<bool>(Element::DIM, true); }

	// constant variable
	int DIM = Element::DIM;
	int VEC_NUM = Element::VEC_NUM;
	int NMAX = 8;
	// int N_init = NMAX;
	int N_init = 2;
	bool sparse = false;			// use full grid (0) or sparse grid (1) when initialization
	const std::string boundary_type = "period";
	double final_time = 10.0;
	double cfl = 0.1;
	const bool is_adapt_find_ptr_alpt = true;	// variable control if need to adaptively find out pointers related to Alpert basis in DG operators
	const bool is_adapt_find_ptr_intp = false;	// variable control if need to adaptively find out pointers related to interpolation basis in DG operators
	std::vector<double> output_time;
	if (problem == 5) { output_time = {0.03, 0.04}; }
	else { output_time = {20.0}; }
	// if (problem == 3) { output_time = linspace(1e-3, 5e-3, 5); }

	// adaptive parameter
	// if need to test code without adaptive, just set refine_eps large number 1e6, then no refine
	// and set coarsen_eta negative number -1, then no coarsen
	// double refine_eps = 1e10;
	// double coarsen_eta = -1;

	double refine_eps = 1e-2;
	double coarsen_eta = refine_eps/10.0;

	OptionsParser args(argc, argv);
	args.AddOption(&NMAX, "-NM", "--max-mesh-level", "Maximum mesh level");
	args.AddOption(&N_init, "-N0", "--initial-mesh-level", "Mesh level in initialization");
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

	// const double prob3_M = 50;
	auto func_source = [=](std::vector<double> u, int i, int d)->double
		{
			// return 0.;
			double u20 = u[0]*u[0]+u[1]*u[1];
			double u21 = u20 * u20;
			
			if (problem == 0)
			{
				if (i==0){ return -1.0 * (u20 + u21) * u[1]; }
				else if (i==1){ return (u20 + u21) * u[0]; }
			}
			else if (problem == 3)
			{
				if (i==0){ return -1.0 * (u20 + u21) * u[1]; }
				else if (i==1){ return (u20 + u21) * u[0]; }
			}
			else if (problem >= 4)
			{
				if (i==0){ return -1.0 * u20 * u[1]; }
				else if (i==1){ return u20 * u[0]; }
			}
			else
			{
				if (i==0){ return -1.0 * schrodinger::const_beta * u20 * u[1]; }
				else if (i==1){ return schrodinger::const_beta * u20 * u[0]; }
			}	
		};

	// initial condition setup
	std::vector<std::vector<std::function<double(double, int)>>> init_func;
	// std::cout << "size of init_func before: " << init_func.size() << std::endl;
	// problem: 0: 1D accuracy, 1: bound state solution, 2: birth of soliton, 
	// 3: 2D accuracy, 4: 2D singular, 5: 2D single blow up, 6: 2nd single blow example
	if (problem == 0) // 1D accuracy
	{
		std::vector<std::function<double(double, int)>> init_func1;
		std::function<double(double, int)> init_func_1 = [=](double x, int d) 
		{ 	
			return cos(schrodinger::const_c * x); 
		};
		init_func1.push_back(init_func_1);

		std::vector<std::function<double(double, int)>> init_func2;
		init_func_1 = [=](double x, int d) 
		{ 	
			return sin(schrodinger::const_c * x); 
		};
		init_func2.push_back(init_func_1);

		init_func.push_back(init_func1);
		init_func.push_back(init_func2);
	}
	else if (problem == 1) // 1D bound state
	{
		std::vector<std::function<double(double, int)>> init_func1;
		std::function<double(double, int)> init_func_1 = [=](double x, int d) 
		{ 	
			return sech(schrodinger::const_M * (x - schrodinger::const_x) ); 
		};
		init_func1.push_back(init_func_1);

		std::vector<std::function<double(double, int)>> init_func2;
		init_func_1 = [=](double x, int d) 
		{ 	
			return 0.0; 
		};
		init_func2.push_back(init_func_1);

		init_func.push_back(init_func1);
		init_func.push_back(init_func2);
	}
	else if (problem == 2)
	{
		std::vector<std::function<double(double, int)>> init_func1;
		std::function<double(double, int)> init_func_1 = [=](double x, int d) 
		{ 	
			double val = schrodinger::const_M * (x - schrodinger::const_x);
			double a = 1.78;
			// return a * exp(-1.0 * val * val); // for birth of standing soliton
			return a * exp(-1.0 * val * val) * cos(2.0 * val); // for birth of mobile soliton 

		};
		init_func1.push_back(init_func_1);

		std::vector<std::function<double(double, int)>> init_func2;
		init_func_1 = [=](double x, int d) 
		{ 	
			double val = schrodinger::const_M * (x - schrodinger::const_x);
			double a = 1.78;
			// return 0.0; // for birth of standing soliton
			return a * exp(-1.0 * val * val) * sin(2.0 * val); // for birth of mobile soliton
		};
		init_func2.push_back(init_func_1);

		init_func.push_back(init_func1);
		init_func.push_back(init_func2);
	}
	else if (problem == 3) // 2D accuracy
	{
		// initial condition
		// u(x, y) = exp(i*2*pi*(x+y)) = cos(2*pi*(x+y)) + i * sin(2*pi*(x+y))
		std::function<double(double, int)> init_func_1 = [](double x, int d) { return (d == 0) ? (cos(2.*Const::PI*x)) : (cos(2.*Const::PI*x)); };
		std::function<double(double, int)> init_func_2 = [](double x, int d) { return (d == 0) ? -(sin(2.*Const::PI*x)) : (sin(2.*Const::PI*x)); };
		std::vector<std::function<double(double, int)>> init_func1{ init_func_1, init_func_2};

		init_func_1 = [](double x, int d) { return (d == 0) ? (sin(2.*Const::PI*x)) : (cos(2.*Const::PI*x)); };
		init_func_2 = [](double x, int d) { return (d == 0) ? (cos(2.*Const::PI*x)) : (sin(2.*Const::PI*x)); };
		std::vector<std::function<double(double, int)>> init_func2{ init_func_1, init_func_2};

		init_func.push_back(init_func1);
		init_func.push_back(init_func2);
	}
	else if (problem == 4) // 2D singular
	{
		// initial condition
		// u(x, y) = (1 + sin(2*pi*x))(2 + sin(2*pi*y))
		std::function<double(double, int)> init_func_1 = [](double x, int d) { return (d == 0) ? (1.0 + sin(2.*Const::PI*x)) : (2.0 + sin(2.*Const::PI*x)); };
		std::vector<std::function<double(double, int)>> init_func1{ init_func_1};

		init_func_1 = [](double x, int d) { return 0.0; };
		std::vector<std::function<double(double, int)>> init_func2{ init_func_1};

		init_func.push_back(init_func1);
		init_func.push_back(init_func2);
	}
	else if (problem == 5) // 2D single blow up
	{
		// initial condition
		// u(x, y) = 6*sqrt(2)*e^(-(M*(x-0.5))^2)*e^(-(M*(y-0.5))^2)
		std::function<double(double, int)> init_func_1 = [](double x, int d) 
		{ 
			double val = schrodinger::const_M * (x - schrodinger::const_x);
			double a = 6.0 * sqrt(2.0);
			if (d==0) {return a * exp(-1.0 * val * val);}
			else {return exp(-1.0 * val * val);}
		};
		std::vector<std::function<double(double, int)>> init_func1{ init_func_1};

		init_func_1 = [](double x, int d) { return 0.0; };
		std::vector<std::function<double(double, int)>> init_func2{ init_func_1};

		init_func.push_back(init_func1);
		init_func.push_back(init_func2);
	}
	else if (problem == 6) // 2D 2nd blow up example
	{
		// initial condition
		// u(x, y) = 2.0 + 0.01sin(M*(x-0.5) + pi/4)*sin(M*(y-0.5) + pi/4)
		std::function<double(double, int)> init_func_1 = [](double x, int d) 
		{ 
			double val = schrodinger::const_M * (x - schrodinger::const_x);
			double a = 0.01;
			if (d==0) {return a * sin(val + Const::PI/4.0); }
			else {return sin(val + Const::PI/4.0); }
		};
		std::function<double(double, int)> init_func_2 = [](double x, int d) { return (d == 0) ? (2.0) : (1.0); };
		std::vector<std::function<double(double, int)>> init_func1{ init_func_1, init_func_2};

		init_func_1 = [](double x, int d) { return 0.0; };
		std::vector<std::function<double(double, int)>> init_func2{ init_func_1};

		init_func.push_back(init_func1);
		init_func.push_back(init_func2);
	}

	// initialization of DG solution
	DGAdapt dg_solu(sparse, N_init, NMAX, all_bas_alpt, all_bas_lagr, all_bas_herm, hash, refine_eps, coarsen_eta, is_adapt_find_ptr_alpt, is_adapt_find_ptr_intp);

	// project initial function into numerical solution
	dg_solu.init_separable_system_sum(init_func);

	dg_solu.coarsen();
	std::cout << "total num of basis (initial): " << dg_solu.size_basis_alpt() << std::endl;

	IO inout(dg_solu);
	if (problem < 3) // 1D
	{
		inout.output_num_schrodinger("profile1D_init.txt");
		inout.output_element_level_support("suppt1D_init.txt");
	}
	else  // 2D
	{
		inout.output_num_schrodinger("profile2D_init.txt");
		inout.output_element_center("center2D_init.txt");
	}

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
	int istart = 0;
	while ( curr_time < final_time )
	{			
		// --- part 1: calculate time step dt ---
		const std::vector<int> & max_mesh = dg_solu.max_mesh_level_vec();
		
		// dt = cfl/(c1/dx1 + c2/dx2 + ... + c_dim/dx_dim)
		double sum_c_dx = 0.;	// this variable stores (c1/dx1 + c2/dx2 + ... + c_dim/dx_dim)
		for (size_t d = 0; d < DIM; d++)
		{
			// sum_c_dx += std::pow(2., max_mesh[d]) * schrodinger::const_alpha;
			if (problem == 2) {sum_c_dx += std::pow(2., max_mesh[d]) / schrodinger::const_M;}
			else 
			{
				if(AlptBasis::PMAX == 3 && accuracy == 1){sum_c_dx += std::pow(std::pow(2., max_mesh[d]), 4.0/3.0);}
				else{sum_c_dx += std::pow(2., max_mesh[d]);}
			}
		}		
		double dt = cfl/sum_c_dx;
		dt = std::min( dt, final_time - curr_time );
		
		// --- part 2: predict by Euler forward
		{
			// before Euler forward, copy Element::ucoe_alpt to Element::ucoe_alpt_predict
			dg_solu.copy_ucoe_to_predict();

			SchrodingerAlpt dg_operator(dg_solu, oper_matx_alpt);
			// dg_operator.assemble_matrix();
			if (problem == 0 || problem == 3){dg_operator.assemble_matrix_couple(1.0);  }
			else { dg_operator.assemble_matrix_couple(1.0/(schrodinger::const_M * schrodinger::const_M));  }
			
			// HyperbolicAlpt linear_hyper(dg_solu, oper_matx_alpt);
			// linear_hyper.assemble_matrix_schrodinger(schrodinger::const_alpha);

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
			// odesolver.add_rhs_matrix(linear_hyper);

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
		if (problem == 0 || problem == 3){dg_operator.assemble_matrix_couple(1.0);  }
		else { dg_operator.assemble_matrix_couple(1.0/(schrodinger::const_M * schrodinger::const_M));  }

		// HyperbolicAlpt linear_hyper(dg_solu, oper_matx_alpt);
		// linear_hyper.assemble_matrix_schrodinger(schrodinger::const_alpha);

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
				// odesolver.add_rhs_matrix(linear_hyper);
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

		std::string file_name = "error_time_N" + std::to_string(NMAX) + "_P" + std::to_string(AlptBasis::PMAX)
										+ "_eps" + to_str(refine_eps) + ".txt";

		if (num_time_step % 20 == 0)
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

			if (accuracy == 1)
			{
				// calculate error between numerical solution and exact solution
				std::cout << "calculating error at: " << curr_time << std::endl;
				std::vector<std::function<double(std::vector<double>)>> final_func;
				// exact solution
				if (problem == 0) // 1D accuracy
				{
					auto final_func1 = [=](std::vector<double> x) {return cos(2.*Const::PI*x[0] - schrodinger::const_w1 * curr_time); };
					auto final_func2 = [=](std::vector<double> x) {return sin(2.*Const::PI*x[0] - schrodinger::const_w1 * curr_time ); };
					
					final_func.push_back(final_func1);
					final_func.push_back(final_func2);
				}
				else if (problem == 3) // 2D accuracy
				{
					auto final_func1 = [=](std::vector<double> x) {return cos(2.*Const::PI*(x[0]+x[1]) - schrodinger::const_w1 * curr_time); };
					auto final_func2 = [=](std::vector<double> x) {return sin(2.*Const::PI*(x[0]+x[1]) - schrodinger::const_w1 * curr_time); };

					final_func.push_back(final_func1);
					final_func.push_back(final_func2);
				}

				const int num_gauss_pt = 5;
				std::vector<double> err = dg_solu.get_error_no_separable_system(final_func, num_gauss_pt);
				std::cout << "L1, L2 and Linf error at time: " << curr_time << std::endl;
				std::cout << err[0] << ", " << err[1] << ", " << err[2] << std::endl;

				inout.write_error_time(curr_time, err, istart, file_name);
			}
			istart = 1;
		}

		if (accuracy == 0)
		{
			if (problem < 3) // 1D
			{
				auto min_time_iter = std::min_element(output_time.begin(), output_time.end());
				if ((curr_time <= *min_time_iter) && (curr_time+dt >= *min_time_iter))
				{
					std::cout << "current time is: " << curr_time << std::endl;

					std::cout << "total num of basis (last time step): " << dg_solu.size_basis_alpt() << std::endl;

					inout.output_num_schrodinger("profile1D_" + std::to_string(curr_time) + ".txt");
					
					inout.output_element_level_support("suppt1D_" + std::to_string(curr_time) + ".txt");

					output_time.erase(min_time_iter);
				}
			}
			else // 2D
			{
				auto min_time_iter = std::min_element(output_time.begin(), output_time.end());
				if ((curr_time <= *min_time_iter) && (curr_time+dt >= *min_time_iter))
				{
					std::cout << "current time is: " << curr_time << std::endl;

					std::cout << "total num of basis (last time step): " << dg_solu.size_basis_alpt() << std::endl;

					inout.output_num_schrodinger("profile2D_" + std::to_string(curr_time) + ".txt");
					
					inout.output_element_center("center2D_" + std::to_string(curr_time) + ".txt");

					output_time.erase(min_time_iter);
				}
			}
		}	
	}

	std::cout << "--- evolution finished ---" << std::endl;
	std::cout << "total num of basis (last time step): " << dg_solu.size_basis_alpt() << std::endl;

	if (accuracy == 1)
	{
		// calculate error between numerical solution and exact solution
		std::cout << "calculating error at final time" << std::endl;
		std::vector<std::function<double(std::vector<double>)>> final_func;
		// exact solution
		if (problem == 0)
		{
			auto final_func1 = [=](std::vector<double> x) {return cos(2.*Const::PI*x[0] - schrodinger::const_w1 * final_time ); };
			auto final_func2 = [=](std::vector<double> x) {return sin(2.*Const::PI*x[0] - schrodinger::const_w1 * final_time ); };
			
			final_func.push_back(final_func1);
			final_func.push_back(final_func2);
		}
		else if (problem == 3)
		{
			auto final_func1 = [=](std::vector<double> x) {return cos(2.*Const::PI*(x[0]+x[1]) - schrodinger::const_w1 * final_time); };
			auto final_func2 = [=](std::vector<double> x) {return sin(2.*Const::PI*(x[0]+x[1]) - schrodinger::const_w1 * final_time); };

			final_func.push_back(final_func1);
			final_func.push_back(final_func2);
		}
		

		const int num_gauss_pt = 5;
		std::vector<double> err = dg_solu.get_error_no_separable_system(final_func, num_gauss_pt);
		std::cout << "L1, L2 and Linf error at final time: " << err[0] << ", " << err[1] << ", " << err[2] << std::endl;

		std::vector<std::vector<double>> err_each;
		for (int num_var = 0; num_var < VEC_NUM; num_var++)
		{
			std::vector<double> err = dg_solu.get_error_no_separable_system_each(final_func, num_gauss_pt, num_var);
			std::cout << "L1, L2 and Linf error of " << num_var << " variable at final time: " << err[0] << " " << err[1] << " " << err[2] << std::endl;
			err_each.push_back(err);
		}

		IO inout(dg_solu);
		// std::string file_name = "error_N" + std::to_string(NMAX) + ".txt";
		// inout.write_error(NMAX, err, file_name);
		double refine_eps1 = 1/refine_eps;
		inout.write_error_eps_dof(refine_eps, dg_solu.size_basis_alpt(), err_each, "error_eps" + to_str(refine_eps1) + ".txt");
	}	

	if (problem < 3) // 1D
	{
		inout.output_num_schrodinger("profile1D_final.txt");
		inout.output_element_level_support("suppt1D_final.txt");
	}
	else  // 2D
	{
		inout.output_num_schrodinger("profile2D_final.txt");
		inout.output_element_center("center2D_final.txt");
	}

	// ------------------------------

	return 0;
}
