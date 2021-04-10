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

// ------------------------------------------------------------------------------------------------
// solve 2D ZK equation (nonlinear ZK equation with source)
// with third-order IMEX and periodic boundary condition
// 
// accuracy test for full grid and sparse grid
// 
// u_t + u*u_x + u_xxx + u_xyy = s(x,y,t)
// with exact solution:
// u(x,y,t) = sin(2*pi*(x+y+t))
// and 
// s(x,y,t) = 2 \[Pi] Cos[2 \[Pi] (t + x + y)] (1 - 8 \[Pi]^2 + Sin[2 \[Pi] (t + x + y)])
// 			= (2*pi-16*pi^3) * (cos(2*pi*x) * cos(2*pi*(y+t)))
//				- (2*pi-16*pi^3) * (sin(2*pi*x) * sin(2*pi*(y+t)))
// 			 	+ pi * sin(4*pi*x) * cos(4*pi*(y+t))
// 			 	+ pi * cos(4*pi*x) * sin(4*pi*(y+t))
// ------------------------------------------------------------------------------------------------
int main(int argc, char *argv[])
{
	// static variables
	AlptBasis::PMAX = 3;

	LagrBasis::PMAX = 3;
	LagrBasis::msh_case = 1;

	HermBasis::PMAX = (AlptBasis::PMAX==3) ? 5 : 3;
	HermBasis::msh_case = 1;

	Element::PMAX_alpt = AlptBasis::PMAX;	// max polynomial degree for Alpert's basis functions
	Element::PMAX_intp = HermBasis::PMAX;	// max polynomial degree for interpolation basis functions
	Element::DIM = 2;			// dimension
	Element::VEC_NUM = 1;		// num of unknown variables in PDEs

	DGSolution::DIM = Element::DIM;
	DGSolution::VEC_NUM = Element::VEC_NUM;

	Interpolation::DIM = DGSolution::DIM;
	Interpolation::VEC_NUM = DGSolution::VEC_NUM;
	
	DGSolution::ind_var_vec = { 0 };
	DGAdapt::indicator_var_adapt = { 0 };

	Element::is_intp.resize(Element::VEC_NUM);
	for (size_t num = 0; num < Element::VEC_NUM; num++) { Element::is_intp[num] = std::vector<bool>(Element::DIM, true); }

	// constant variable
	const int DIM = Element::DIM;
	int NMAX = 4;
	const bool sparse = true;
	const std::string boundary_type = "period";
	double final_time = 0.01;
	const double cfl = 0.02;
	const bool is_adapt_find_ptr_alpt = true;	// variable control if need to adaptively find out pointers related to Alpert basis in DG operators
	const bool is_adapt_find_ptr_intp = true;	// variable control if need to adaptively find out pointers related to interpolation basis in DG operators

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
	
	const int N_init = NMAX;

	// adaptive parameter
	// if need to test code without adaptive, just set refine_eps large number 1e6, then no refine
	// and set coarsen_eta negative number -1, then no coarsen
	const double refine_eps = 1e10;
	const double coarsen_eta = -1;

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
					
	// initialization of DG solution
	DGAdapt dg_solu(sparse, N_init, NMAX, all_bas_alpt, all_bas_lagr, all_bas_herm, hash, refine_eps, coarsen_eta, is_adapt_find_ptr_alpt, is_adapt_find_ptr_intp);

	auto init_func_1 = [](double x, int d) { return (d==0) ? (sin(2.*Const::PI*x)) : (cos(2.*Const::PI*x)); };
	auto init_func_2 = [](double x, int d) { return (d==0) ? (cos(2.*Const::PI*x)) : (sin(2.*Const::PI*x)); };
	std::vector<std::function<double(double, int)>> init_func{init_func_1, init_func_2};	
	dg_solu.init_separable_scalar_sum(init_func);

	// ------------------------------
	const int max_mesh = dg_solu.max_mesh_level();
	const double dx = 1./pow(2., max_mesh);
	double dt = dx * cfl;
	if (AlptBasis::PMAX==3) { dt = pow(dx, 4./3.) * cfl; }
	int total_time_step = ceil(final_time/dt);	
	dt = (total_time_step==0) ? 0. : (final_time/total_time_step);

	// physical flux
	auto func_flux = [&](std::vector<double> u, int i, int d)->double { return FluxFunction::burgers_flux_scalar(u[0]); };
	
	// first derivative of physical flux
	auto func_flux_d1 = [&](std::vector<double> u, int i, int d, int i1)->double { return FluxFunction::burgers_flux_1st_derivative_scalar(u[0]); };

	// second derivative of physical flux
	auto func_flux_d2 = [&](std::vector<double> u, int i, int d, int i1, int i2)->double { return FluxFunction::burgers_flux_2nd_derivative_scalar(u[0]); };

	// third derivative of physical flux
	auto func_flux_d3 = [&](std::vector<double> u, int i, int d, int i1, int i2, int i3)->double { return 0.; };

	// fourth derivative of physical flux
	auto func_flux_d4 = [&](std::vector<double> u, int i, int d, int i1, int i2, int i3, int i4)->double { return 0.; };

	ZKAlpt dg_operator(dg_solu, oper_matx_alpt);
	dg_operator.assemble_matrix_scalar({1.0, 1.0});

	// flux integral of u^- * [v] and u^+ * [v] in x direction
	const double lxf_alpha = 1.1;
	HyperbolicAlpt linear(dg_solu, oper_matx_alpt);
	linear.assemble_matrix_flx_scalar(0, -1, lxf_alpha/2);
	linear.assemble_matrix_flx_scalar(0, 1, -lxf_alpha/2);

	// Hermite interpolation for nonlinear physical flux
	HermInterpolation interp_herm(dg_solu);

	HyperbolicSameFluxHermRHS fast_rhs_herm(dg_solu, oper_matx_herm);

	// interpolation is only needed in x direction
	std::vector< std::vector<bool> > is_intp_herm;
	is_intp_herm.push_back(std::vector<bool>{true, false});

	// fast hermite interpolation
	FastHermIntp fast_herm_intp(dg_solu, interp_herm.Her_pt_Alpt_1D);

	std::cout << "--- evolution started ---" << std::endl;
	// record code running time
	auto start_evolution_time = std::chrono::high_resolution_clock::now();
	
	IMEX43 odesolver(dg_operator, dt, "sparselu");

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
				// Hermite interpolation
				if (HermBasis::PMAX==3)
				{
					interp_herm.nonlinear_Herm_2D_fast(func_flux, func_flux_d1, func_flux_d2, is_intp_herm, fast_herm_intp);
				}
				else if (HermBasis::PMAX==5)
				{
					interp_herm.nonlinear_Herm_2D_PMAX5_scalar_fast(func_flux, func_flux_d1, func_flux_d2, func_flux_d3, func_flux_d4, is_intp_herm, fast_herm_intp);
				}
				
				// calculate rhs and update Element::ucoe_alpt
				dg_solu.set_rhs_zero();

				// nonlinear convection terms, only in x direction
				fast_rhs_herm.rhs_vol_scalar(0);
				fast_rhs_herm.rhs_flx_intp_scalar(0);

				// add to rhs in ode solver
				odesolver.set_rhs_zero();
				odesolver.add_rhs_to_eigenvec();
				odesolver.add_rhs_matrix(linear);				

				// source term
				double source_time = curr_time;
				if (num_stage == 3) { source_time += dt; }
				else if (num_stage == 4) { source_time += dt/2.; }
				// compute projection of source term f(x, y, t) with separable formulation
				auto source_func = [&](std::vector<double> x, int i)->double { return 2*Const::PI*cos(2*Const::PI*(source_time+x[0]+x[1]))*(1-8*pow(Const::PI,2.)+sin(2*Const::PI*(source_time+x[0]+x[1]))); };
				Domain2DIntegral source_operator(dg_solu, all_bas_alpt, source_func);
				source_operator.assemble_vector();
				odesolver.add_rhs_vector(source_operator.vec_b);			
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
				<< "; running time: " << duration.count()/1e6 << " seconds"
				<< std::endl;
	}
	
	std::cout << "--- evolution finished ---" << std::endl;
	std::cout << "total num of basis (last time step): " << dg_solu.size_basis_alpt() << std::endl;

	auto final_func_1 = [&](double x, int d) { return (d==0) ? (sin(2.*Const::PI*(x+final_time))) : (cos(2.*Const::PI*x)); };
	auto final_func_2 = [&](double x, int d) { return (d==0) ? (cos(2.*Const::PI*(x+final_time))) : (sin(2.*Const::PI*x)); };
	std::vector<std::function<double(double, int)>> final_func{final_func_1, final_func_2};
	
	const int num_gauss_pt = 5;
	std::vector<double> err = dg_solu.get_error_separable_scalar_sum(final_func, num_gauss_pt);
	std::cout << "L1, L2 and Linf error at final time: " << err[0] << ", " << err[1] << ", " << err[2] << std::endl;

	IO inout(dg_solu);
	std::string file_name = "error_N" + std::to_string(NMAX) + ".txt";
	inout.write_error(NMAX, err, file_name);	
	// ------------------------------

	return 0;
}
