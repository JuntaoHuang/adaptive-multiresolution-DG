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
#include "Timer.h"
#include "VecMultiD.h"

// ------------------------------------------------------------------------------------------------
// solve 2D and 3D wave equation with variable coefficient and periodic boundary condition
// 
// u_tt = grad \cdot (k(x) grad u) + f(x, t)
// 
// The initial condition u, source function f, coefficient function k can be found in the following three functions.
// 
// Problem 1 (2D case):
// 	u[x, y, t] = Sin[ pi t] Sin[2 pi x] Cos[2 pi y]
// 	k[x, y] = (Cos[2 pi x] Cos[2 pi y] + 2)/3
// 
// Problem 2 (3D case):
// 	u[x, y, t] = Sin[ pi t] Sin[2 pi x] Cos[2 pi y] Cos[2 pi z]
// 	k[x, y] = (Cos[2 pi x] Cos[2 pi y] Cos[2 pi z] + 2)/3
// ------------------------------------------------------------------------------------------------
double init_function_setup(double x, int d, int problem);

double source_function_separabla_setup(double x, int dim, int num_separable, int var_index, int problem);

std::vector<int> source_num_separable(int problem);

double time_function_setup(double t, int problem);

double coeffcient_function_setup(std::vector<double> x, int d, int problem);

int main(int argc, char *argv[])
{
	// static variables
	AlptBasis::PMAX = 3;

	LagrBasis::PMAX = 4;
	LagrBasis::msh_case = 2;

	HermBasis::PMAX = 3;
	HermBasis::msh_case = 1;

	Element::PMAX_alpt = AlptBasis::PMAX;	// max polynomial degree for Alpert's basis functions
	Element::PMAX_intp = LagrBasis::PMAX;	// max polynomial degree for interpolation basis functions
	Element::DIM = 2;			// dimension
	Element::VEC_NUM = 1;		// num of unknown variables in PDEs

	DGSolution::DIM = Element::DIM;
	DGSolution::VEC_NUM = Element::VEC_NUM;

	Interpolation::DIM = DGSolution::DIM;
	Interpolation::VEC_NUM = DGSolution::VEC_NUM;

	DGSolution::prob = "wave";
	DGSolution::ind_var_vec = { 0 };
	DGAdapt::indicator_var_adapt = { 0 };

	Element::is_intp.resize(Element::VEC_NUM);
	for (size_t num = 0; num < Element::VEC_NUM; num++) { Element::is_intp[num] = std::vector<bool>(Element::DIM, true); }

	// constant variable
	const int DIM = Element::DIM;
	int NMAX = 8;
	const bool sparse = true;
	const std::string boundary_type = "period";
	double final_time = 0.1;
	const double cfl = (DIM==2) ? 0.1 : 0.05;
	const bool is_adapt_find_ptr_alpt = true;	// variable control if need to adaptively find out pointers related to Alpert basis in DG operators
	const bool is_adapt_find_ptr_intp = false;	// variable control if need to adaptively find out pointers related to interpolation basis in DG operators
	int problem = (DIM==2) ? 1 : 2;

	int N_init = 2;

	// adaptive parameter
	// if need to test code without adaptive, just set refine_eps large number 1e6, then no refine
	// and set coarsen_eta negative number -1, then no coarsen
	double refine_eps = 1e10;
	double coarsen_eta = -1;

	OptionsParser args(argc, argv);
	args.AddOption(&NMAX, "-NM", "--max-mesh-level", "Maximum mesh level");
	args.AddOption(&N_init, "-N0", "--initial-mesh-level", "Mesh level in initialization");
	args.AddOption(&refine_eps, "-r", "--refine-epsilon", "refine parameter epsilon");
	args.AddOption(&coarsen_eta, "-c", "--coarsen-eta", "coarsen parameter eta");
	args.AddOption(&final_time, "-tf", "--final-time", "Final time; start time is 0.");
	args.AddOption(&problem, "-p", "--problem", "Problem setup to use.");

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
	OperatorMatrix1D<AlptBasis,AlptBasis> oper_matx_alpt(all_bas_alpt, all_bas_alpt, boundary_type);

	OperatorMatrix1D<HermBasis,AlptBasis> oper_matx_herm(all_bas_herm, all_bas_alpt, boundary_type);

	OperatorMatrix1D<LagrBasis,AlptBasis> oper_matx_lagr(all_bas_lagr, all_bas_alpt, boundary_type);

	// initialization of DG solution
	DGAdapt dg_solu(sparse, N_init, NMAX, all_bas_alpt, all_bas_lagr, all_bas_herm, hash, refine_eps, coarsen_eta, is_adapt_find_ptr_alpt, is_adapt_find_ptr_intp);

	// initial condition
	auto init_func = [&](double x, int d)->double 
		{ 
			if (d==0) { return Const::PI * sin(2 * Const::PI * x); }
			else { return cos(2 * Const::PI * x); }
		};

	// project initial function into numerical solution
	dg_solu.init_separable_scalar(init_func);
	dg_solu.copy_ucoe_to_ucoe_ut();
	dg_solu.set_ucoe_alpt_zero();

	// ------------------------------
	// 	This part is for solving constant coefficient for linear equation	
	std::vector<double> waveConst(DIM, 1.);
	double sigma_ipdg = 0.;
	if (DIM == 2) { sigma_ipdg = 10; }
	else if (DIM == 3) { sigma_ipdg = 20; }

	DiffusionRHS diffuseRHS(dg_solu, oper_matx_lagr);	

	// compute projection of source term f(x, y, t) with separable formulation
	auto source_func = [&](double x, int dim, int separable, int var_index)->double { return source_function_separabla_setup(x, dim, separable, var_index, problem); };
	DomainIntegral source_operator(dg_solu, all_bas_alpt, source_func, source_num_separable(problem));

	// coefficient function k
	auto coe_func = [&](std::vector<double> x, int d) ->double { return coeffcient_function_setup(x,d,problem); };

	LagrInterpolation interp(dg_solu);
	//	variable to control which flux need interpolation
	std::vector< std::vector<bool> > is_intp;
	is_intp.push_back(std::vector<bool>(DIM, true));

	std::vector< std::vector<bool> > is_intp_d0;
	std::vector<bool> a(DIM, false);
	a[0] = true;
	is_intp_d0.push_back(a);

	FastLagrIntp fastLagr(dg_solu, interp.Lag_pt_Alpt_1D, interp.Lag_pt_Alpt_1D_d1);	

	std::cout << "--- evolution started ---" << std::endl;

	// record code running time
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
			sum_c_dx += std::abs(waveConst[d]) * std::pow(2., max_mesh[d]);
		}
		double dt = cfl/sum_c_dx;
		dt = std::min(dt, final_time - curr_time);		

		// --- part 2: predict by Euler forward
		{	
			// before Euler forward, copy Element::ucoe_alpt to Element::ucoe_alpt_predict
			dg_solu.copy_ucoe_to_predict();
			dg_solu.copy_ucoe_ut_to_predict();

			DiffusionAlpt operator_ujp_vjp(dg_solu, oper_matx_alpt, sigma_ipdg);
			operator_ujp_vjp.assemble_matrix_flx_ujp_vjp();	

			dg_solu.set_source_zero();
			source_operator.resize_zero_vector();
			source_operator.assemble_vector();

			EulerODE2nd odesolver(operator_ujp_vjp, dt);
			odesolver.init();

			dg_solu.set_rhs_zero();

			// interpolation of k * u_x and k * u_y
			interp.var_coeff_gradu_Lagr_fast(coe_func, is_intp, fastLagr);
			
			diffuseRHS.rhs_flx_gradu();
			diffuseRHS.rhs_vol();

			// interpolation of k * u		
			interp.var_coeff_u_Lagr_fast(coe_func, is_intp_d0, fastLagr);

			diffuseRHS.rhs_flx_u();
			odesolver.rhs_to_eigenvec();
			
			// [u] * [v]
			odesolver.add_rhs_matrix(operator_ujp_vjp);			
			
			auto source_func = [&](double t)->Eigen::VectorXd { return sin(Const::PI*t) * source_operator.vec_b; };
			odesolver.step_rk_source(source_func, curr_time);

			// copy eigen vector ucoe to Element::ucoe_alpt
			// now prediction of numerical solution at t^(n+1) is stored in Element::ucoe_alpt
			odesolver.final();
		}

		// --- part 3: refine base on Element::ucoe_alpt
		dg_solu.refine();
		const int num_basis_refine = dg_solu.size_basis_alpt();

		// after refine, copy Element::ucoe_alpt_predict back to Element::ucoe_alpt
		dg_solu.copy_predict_to_ucoe();
		dg_solu.copy_predict_to_ucoe_ut();

		// --- part 4: time evolution
		// -sigma/h * [u] * [v]
		DiffusionAlpt operator_ujp_vjp(dg_solu, oper_matx_alpt, sigma_ipdg);
		operator_ujp_vjp.assemble_matrix_flx_ujp_vjp();	

		dg_solu.set_source_zero();
		source_operator.resize_zero_vector();
		source_operator.assemble_vector();

		RK4ODE2nd odesolver(operator_ujp_vjp, dt);
		odesolver.init();

		for (size_t stage = 0; stage < odesolver.num_stage; stage++)
		{		
			dg_solu.set_rhs_zero();

			// interpolation of k * u_x and k * u_y
			interp.var_coeff_gradu_Lagr_fast(coe_func, is_intp, fastLagr);
			
			diffuseRHS.rhs_flx_gradu();
			diffuseRHS.rhs_vol();

			// interpolation of k * u		
			interp.var_coeff_u_Lagr_fast(coe_func, is_intp_d0, fastLagr);

			diffuseRHS.rhs_flx_u();
			odesolver.rhs_to_eigenvec();
			
			// [u] * [v]
			odesolver.add_rhs_matrix(operator_ujp_vjp);			
			
			auto source_func = [&](double t)->Eigen::VectorXd { return sin(Const::PI*t) * source_operator.vec_b; };
			odesolver.step_stage_source(stage, source_func, curr_time);

			odesolver.final();
		}

		// --- part 5: coarsen
		dg_solu.coarsen();
		const int num_basis_coarsen = dg_solu.size_basis_alpt();		
		
		// record code running time
		if (num_time_step % 10 == 0)
		{
			record_time.time("running time");
			std::cout << "num of time steps: " << num_time_step << "; time step: " << dt << "; curr time: " << curr_time << std::endl
					<< "curr max mesh: " << dg_solu.max_mesh_level() << std::endl
					<< "num of basis after refine: " << num_basis_refine << "; num of basis after coarsen: " << num_basis_coarsen << std::endl << std::endl;
		}	

		// add current time and increase time steps
		curr_time += dt;
		num_time_step ++;
	}	
	std::cout << "--- evolution finished ---" << std::endl;

	record_time.time("total running time");
	std::cout << "total num of time steps: " << num_time_step << std::endl
			  << "total num of basis (last time step): " << dg_solu.size_basis_alpt() << std::endl;

	// calculate error between numerical solution and exact solution
	std::cout << "calculating error at final time" << std::endl;
	auto final_func = [&](double x, int d)->double 
					{
						if (d==0) { return sin(Const::PI * final_time) * sin(2 * Const::PI * x); }
						else { return cos(2 * Const::PI * x); }
					};
	
	// compute error at final time using Gauss quadrature in each finest element
	const int num_gauss_pt = 5;
	std::vector<double> err = dg_solu.get_error_separable_scalar(final_func, num_gauss_pt);
	std::cout << "L1, L2 and Linf error at final time: " << err[0] << ", " << err[1] << ", " << err[2] << std::endl;

	// compute L2 error at final time using (u-u_h)^2 = u^2 + u_h^2 - 2*u*u_h
	Quad quad_exact(DIM);
	auto final_exact_sol = [&](std::vector<double> x)->double 
		{ 
			double func = 1.;
			for (size_t d = 0; d < DIM; d++) { func *= final_func(x[d], d); }
			return func;
		};
	const double l2_norm_exact = quad_exact.norm_multiD(final_exact_sol, 3, 10)[1];
	double err2 = dg_solu.get_L2_error_split_separable_scalar(final_func, l2_norm_exact);
	std::cout << "L2 error at final time with split: " << err2 << std::endl;
	
	// std::vector<double> err(3, err2);
	IO inout(dg_solu);
	std::string file_name = "error_N" + std::to_string(NMAX) + ".txt";
	inout.write_error(NMAX, err, file_name);	
	// ------------------------------

	return 0;
}


double source_function_separabla_setup(double x, int dim, int num_separable, int var_index, int problem)
{	
	const double pi = Const::PI;
	if (problem==1)
	{		
		assert((dim>=0 && dim<2) && (num_separable>=0 && num_separable<3) && (var_index==0));

		// 1./3.*pi*pi*sin(2*pi*x)*cos(2*pi*y)*12*cos(2*pi*x)*cos(2*pi*y)
		if (num_separable==0)
		{
			if (dim==0) { return 1./3.*pi*pi*sin(2*pi*x)*12.*cos(2*pi*x); }
			else { return cos(2*pi*x)*cos(2*pi*x); }
		}
		// 1./3.*pi*pi*sin(2*pi*x)*cos(2*pi*y)*13
		else if (num_separable==1)
		{
			if (dim==0) { return 1./3.*pi*pi*sin(2*pi*x)*13.; }
			else { return cos(2*pi*x); }
		}
		// 1./3.*pi*pi*sin(2*pi*x) * (- 4*cos(2*pi*x)*sin(2*pi*y)*sin(2*pi*y)))
		else if (num_separable==2)
		{
			if (dim==0) { return 1./3.*pi*pi*sin(2*pi*x)*(-4)*cos(2*pi*x); }
			else { return sin(2*pi*x)*sin(2*pi*x); }
		}				
	}
	else if (problem==2)
	{
		assert((dim>=0 && dim<3) && (num_separable>=0 && num_separable<8) && (var_index==0));

		// 2/3 pi^2 Cos[2 pi x] Sin[2 pi x]
		if (num_separable==0)
		{
			if (dim==0) { return 2./3.*pi*pi*cos(2*pi*x)*sin(2*pi*x); }
			else if (dim==1 || dim==2) { return 1.; }
		}
		// 4/3 pi^2 Cos[2 pi x] Sin[2 pi x] Cos[4 pi y]
		else if (num_separable==1)
		{
			if (dim==0) { return 4./3.*pi*pi*cos(2*pi*x)*sin(2*pi*x); }
			else if (dim==1) { return cos(4*pi*x); }
			else if (dim==2) { return 1.; }
		}
		// pi^2 Cos[2 pi x] Sin[2 pi x] Cos[4 pi (y - z)]
		//  = pi^2 Cos[2 pi x] Sin[2 pi x] Cos[4 pi y] Cos[4 pi z]
		// 	+ pi^2 Cos[2 pi x] Sin[2 pi x] Sin[4 pi y] Sin[4 pi z]
		else if (num_separable==2)
		{
			if (dim==0) { return pi*pi*cos(2*pi*x)*sin(2*pi*x); }
			else if (dim==1) { return cos(4*pi*x); }
			else if (dim==2) { return cos(4*pi*x); }
		}
		else if (num_separable==3)
		{
			if (dim==0) { return pi*pi*cos(2*pi*x)*sin(2*pi*x); }
			else if (dim==1) { return sin(4*pi*x); }
			else if (dim==2) { return sin(4*pi*x); }
		}
		// 7 pi^2 Cos[2 pi y] Cos[2 pi z] Sin[2 pi x]
		else if (num_separable==4)
		{
			if (dim==0) { return 7*pi*pi*sin(2*pi*x); }
			else if (dim==1) { return cos(2*pi*x); }
			else if (dim==2) { return cos(2*pi*x); }
		}
		// 4/3 pi^2 Cos[2 pi x] Cos[4 pi z] Sin[2 pi x]
		else if (num_separable==5)
		{
			if (dim==0) { return 4./3.*pi*pi*cos(2*pi*x)*sin(2*pi*x); }
			else if (dim==1) { return 1.; }
			else if (dim==2) { return cos(4*pi*x); }
		}
		// pi^2 Cos[2 pi x] Cos[4 pi (y + z)] Sin[2 pi x]
		// = pi^2 Cos[2 pi x] Sin[2 pi x] Cos[4 pi y] Cos[4 pi z]
		// - pi^2 Cos[2 pi x] Sin[2 pi x] Sin[4 pi y] Sin[4 pi z]
		else if (num_separable==6)
		{
			if (dim==0) { return pi*pi*cos(2*pi*x)*sin(2*pi*x); }
			else if (dim==1) { return cos(4*pi*x); }
			else if (dim==2) { return cos(4*pi*x); }
		}
		else if (num_separable==7)
		{
			if (dim==0) { return pi*pi*cos(2*pi*x)*sin(2*pi*x); }
			else if (dim==1) { return - sin(4*pi*x); }
			else if (dim==2) { return sin(4*pi*x); }
		}	
	}
	else
	{
		std::cout << "problem option not correct" << std::endl; exit(1);
	}	
}

std::vector<int> source_num_separable(int problem)
{
	if (problem==1)
	{
		return {3};
	}
	else if (problem==2)
	{
		return {8};
	}	
	else
	{
		std::cout << "problem option not correct" << std::endl; exit(1);
	}	
}

double time_function_setup(double t, int problem)
{
	const double pi = Const::PI;

	if (problem==1 || problem==2)
	{
		return sin(pi * t);
	}
	else
	{
		std::cout << "problem option not correct" << std::endl; exit(1);
	}
}

double coeffcient_function_setup(std::vector<double> xvec, int d, int problem)
{	
	const double pi = Const::PI;
	double x = xvec[0];
	double y = xvec[1];
	
	if (problem==1)
	{
		return (cos(2*pi*x)*cos(2*pi*y) + 2.)/3.;
	}
	else if (problem==2)
	{
		double z = xvec[2];
		return (cos(2*pi*x)*cos(2*pi*y)*cos(2*pi*z) + 2.)/3.;
	}
	else
	{
		std::cout << "problem option not correct" << std::endl; exit(1);
	}		
}