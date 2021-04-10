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

// ------------------------------------------------------------------------------------------------
// solve 2D ZK equation with third-order IMEX and periodic boundary condition
// ------------------------------------------------------------------------------------------------
double initial_condition_set_up(std::vector<double> x, int problem);

int main(int argc, char *argv[])
{
	// problem 0: single soliton
	// problem 1: direct collision of two similar pulses solution, see fig. 15 in Xu and Shu 2002
	// problem 2: deviated collision of two similar pulses solution, see fig. 16 in Xu and Shu 2002
	// problem 3: direct collision of two dissimilar pulses solution, see fig. 17 in Xu and Shu 2002
	// problem 4: deviated collision of two dissimilar pulses solution, see fig. 18 in Xu and Shu 2002
	// problem 5: lump solution in "Evolution of two-dimensional lump nanosolitons for the Zakharov-Kuznetsov and electromigration equations" Chaos 2005.
	int problem = 5;

	// static variables
	AlptBasis::PMAX = 2;

	LagrBasis::PMAX = 3;
	LagrBasis::msh_case = 3;

	HermBasis::PMAX = 3;
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
	int NMAX = 8;
	int N_init = 2;
	const bool sparse = false;
	const std::string boundary_type = "period";
	double final_time = 0.51;
	double cfl = 0.01;
	if (problem==5)
	{
		cfl = 0.02;
	}	
	const bool is_adapt_find_ptr_alpt = true;	// variable control if need to adaptively find out pointers related to Alpert basis in DG operators
	const bool is_adapt_find_ptr_intp = true;	// variable control if need to adaptively find out pointers related to interpolation basis in DG operators
	const bool is_adapt_find_ptr_general = true;

	std::vector<double> output_time = linspace(0.01, 0.50, 50);

	// adaptive parameter
	// if need to test code without adaptive, just set refine_eps large number 1e6, then no refine
	// and set coarsen_eta negative number -1, then no coarsen
	double refine_eps = 1e-4;
	double coarsen_eta = 1e-5;
	
	OptionsParser args(argc, argv);
	args.AddOption(&NMAX, "-NM", "--max-mesh-level", "Maximum mesh level");
	args.AddOption(&N_init, "-N0", "--initial-mesh-level", "Mesh level in initialization");
	args.AddOption(&cfl, "-cfl", "--cfl-number", "CFL number");	
	args.AddOption(&final_time, "-tf", "--final-time", "Final time; start time is 0.");
	args.AddOption(&refine_eps, "-r", "--refine-epsilon", "refine parameter epsilon");
	args.AddOption(&coarsen_eta, "-c", "--coarsen-eta", "coarsen parameter eta");
	args.AddOption(&problem, "-p", "--problem", "problem type");

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
	OperatorMatrix1D<LagrBasis, LagrBasis> oper_matx_lagr_all(all_bas_lagr, all_bas_lagr, "period");
	OperatorMatrix1D<HermBasis, HermBasis> oper_matx_herm_all(all_bas_herm, all_bas_herm, "period");

	// initialization of DG solution
	DGAdaptIntp dg_solu(sparse, N_init, NMAX, all_bas_alpt, all_bas_lagr, all_bas_herm, hash, refine_eps, coarsen_eta, is_adapt_find_ptr_alpt, is_adapt_find_ptr_intp, is_adapt_find_ptr_general, oper_matx_lagr_all, oper_matx_herm_all);

	// for problem = 1, 2, 3, 4, computation domain shift from [0, 32]^2 to [0, 1]^2
	// u(x, t) = v(M*x, M*y, alpha*t) with M = alpha = 32
	// here v is the solution in Xu and Shu 2002, defined in [0, 32]^2
	// v_t + (3v^2)_x + v_xxx + v_xyy = 0
	// we can derive equation for u
	// u_t + (6*alpha/M) * u * u_x + alpha/M^3*(u_xxx + u_xyy) = 0
	double M_shift = 32;
	double alpha_t = 32;
	if (problem==3)
	{
		M_shift = 64;
		alpha_t = 64;
	}
	else if (problem==5)
	{
		M_shift = 80;
		alpha_t = 80;
	}
	// initial condition
	auto init_func = [&](std::vector<double> x, int i)->double 
		{
			std::vector<double> x_shift{ M_shift*x[0], M_shift*x[1] };
			if (problem==5)
			{
				x_shift = { M_shift*(x[0]-0.5), M_shift*(x[1]-0.5) };
			}			
			return initial_condition_set_up(x_shift, problem);
		};
	LagrInterpolation interp(dg_solu);
	dg_solu.init_adaptive_intp_Lag(init_func, interp);
	FastLagrInit fast_init(dg_solu, oper_matx_lagr);
	fast_init.eval_ucoe_Alpt_Lagr();

	dg_solu.coarsen();
	std::cout << "total num of basis (initial): " << dg_solu.size_basis_alpt() << std::endl;

	IO inout(dg_solu);
	inout.output_num("profile2D_0.000000.txt");
	inout.output_element_center("center2D_0.000000.txt");

	// physical flux
	auto func_flux = [&](std::vector<double> u, int i, int d)->double { return 6 * alpha_t/M_shift * FluxFunction::burgers_flux_scalar(u[0]); };
	
	// first derivative of physical flux
	auto func_flux_d1 = [&](std::vector<double> u, int i, int d, int i1)->double { return 6 * alpha_t/M_shift * FluxFunction::burgers_flux_1st_derivative_scalar(u[0]); };

	// second derivative of physical flux
	auto func_flux_d2 = [&](std::vector<double> u, int i, int d, int i1, int i2)->double { return 6 * alpha_t/M_shift * FluxFunction::burgers_flux_2nd_derivative_scalar(u[0]); };

	// flux integral of u^- * [v] and u^+ * [v] in x direction
	double lxf_alpha = 6 * 3.2;
	if (problem==5)
	{
		lxf_alpha = 6 * 0.4;
	}	
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

			HyperbolicAlpt linear(dg_solu, oper_matx_alpt);
			linear.assemble_matrix_flx_scalar(0, -1, lxf_alpha/2);
			linear.assemble_matrix_flx_scalar(0, 1, -lxf_alpha/2);

			ZKAlpt dg_operator(dg_solu, oper_matx_alpt);
			dg_operator.assemble_matrix_scalar({alpha_t/pow(M_shift, 3), alpha_t/pow(M_shift, 3)});

			IMEXEuler odesolver(dg_operator, dt, "sparselu");
			odesolver.init();

			// Hermite interpolation
			interp_herm.nonlinear_Herm_2D_fast(func_flux, func_flux_d1, func_flux_d2, is_intp_herm, fast_herm_intp);
			
			// calculate rhs and update Element::ucoe_alpt
			dg_solu.set_rhs_zero();

			fast_rhs_herm.rhs_vol_scalar(0);
			fast_rhs_herm.rhs_flx_intp_scalar(0);

			// add to rhs in ode solver
			odesolver.set_rhs_zero();
			odesolver.add_rhs_to_eigenvec();
			odesolver.add_rhs_matrix(linear);				

			odesolver.step_stage(0);

			odesolver.final();
		}

		// --- part 3: refine base on Element::ucoe_alpt
		dg_solu.refine();
		const int num_basis_refine = dg_solu.size_basis_alpt();

		// after refine, copy Element::ucoe_alpt_predict back to Element::ucoe_alpt
		dg_solu.copy_predict_to_ucoe();
	
		// --- part 4: time evolution
		HyperbolicAlpt linear(dg_solu, oper_matx_alpt);
		linear.assemble_matrix_flx_scalar(0, -1, lxf_alpha/2);
		linear.assemble_matrix_flx_scalar(0, 1, -lxf_alpha/2);

		ZKAlpt dg_operator(dg_solu, oper_matx_alpt);
		dg_operator.assemble_matrix_scalar({alpha_t/pow(M_shift, 3), alpha_t/pow(M_shift, 3)});

		IMEX43 odesolver(dg_operator, dt, "sparselu");
		odesolver.init();

		for (size_t num_stage = 0; num_stage < odesolver.num_stage; num_stage++)
		{
			// calculate rhs for explicit part only for stage = 2, 3, 4
			// no need for stage = 0, 1
			if (num_stage >= 2)
			{
				// Hermite interpolation
				interp_herm.nonlinear_Herm_2D_fast(func_flux, func_flux_d1, func_flux_d2, is_intp_herm, fast_herm_intp);
				
				// calculate rhs and update Element::ucoe_alpt
				dg_solu.set_rhs_zero();

				// nonlinear convection terms, only in x direction
				fast_rhs_herm.rhs_vol_scalar(0);
				fast_rhs_herm.rhs_flx_intp_scalar(0);

				// add to rhs in ode solver
				odesolver.set_rhs_zero();
				odesolver.add_rhs_to_eigenvec();
				odesolver.add_rhs_matrix(linear);	
			}
			odesolver.step_stage(num_stage);

			odesolver.final();
		}

		// --- part 5: coarsen
		dg_solu.coarsen();
		const int num_basis_coarsen = dg_solu.size_basis_alpt();		

		// add current time and increase time steps
		curr_time += dt;		

		if (num_time_step % 10 == 0)
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
		}
		
		auto min_time_iter = std::min_element(output_time.begin(), output_time.end());
		if ((curr_time <= *min_time_iter) && (curr_time+dt >= *min_time_iter))
		{
			inout.output_num("profile2D_" + std::to_string(curr_time) + ".txt");
			inout.output_element_center("center2D_" + std::to_string(curr_time) + ".txt");

			output_time.erase(min_time_iter);
		}

		num_time_step ++;
	}
	
	std::cout << "--- evolution finished ---" << std::endl;
	std::cout << "total num of basis (last time step): " << dg_solu.size_basis_alpt() << std::endl;

	inout.output_num("profile2D_final.txt");
	inout.output_element_center("center2D_final.txt");
	// ------------------------------

	return 0;
}

double initial_condition_set_up(std::vector<double> x, int problem)
{	
	const int num = 10;
	const std::vector<double> a_const{ -1.25529873, 0.21722635, 0.06452543, 0.00540862, -0.00332515, -0.00281281, -0.00138352, -0.00070289, -0.00020451, -0.00003053};
	
	double val = 0;
	if (problem == 0)
	{
		const double c = 4.;
		const std::vector<double> x_init{16, 16};				
		double r = pow(x[0]-x_init[0], 2.) + pow(x[1]-x_init[1], 2.);
		r = pow(r, 0.5);
		double arccot_r = Const::PI/2. - std::atan(pow(c,0.5)/2.*r);
		for (size_t n = 0; n < num; n++)
		{
			val += a_const[n] * (cos(2*(n+1)*arccot_r) - 1.);
		}
		val *= c/3.;
	}
	// two solitons
	else if ((problem==1) || (problem==2) || (problem==3) || (problem==4))
	{
		double c1 = 0.;
		double c2 = 0.;
		std::vector<double> x1_init;
		std::vector<double> x2_init;

		if (problem == 1)
		{
			c1 = 4.4;
			c2 = 4.;
			x1_init = std::vector<double>{6, 16};
			x2_init = std::vector<double>{16, 16};
		}
		else if (problem == 2)
		{
			c1 = 4.4;
			c2 = 4.;
			x1_init = std::vector<double>{6, 14};
			x2_init = std::vector<double>{16, 16};
		}
		else if (problem == 3)
		{
			c1 = 4.;
			c2 = 1.;
			x1_init = std::vector<double>{32, 32};
			x2_init = std::vector<double>{40, 32};
		}
		else if (problem == 4)
		{
			c1 = 4.;
			c2 = 1.;
			x1_init = std::vector<double>{8, 14};
			x2_init = std::vector<double>{16, 16};
		}
		else
		{
			std::cout << "problem type not correct in function initial_condition_set_up()" << std::endl; exit(1);
		}		

		double r1 = pow(x[0]-x1_init[0], 2.) + pow(x[1]-x1_init[1], 2.);
		r1 = pow(r1, 0.5);
		double arccot_r1 = Const::PI/2. - std::atan(pow(c1,0.5)/2.*r1);

		double r2 = pow(x[0]-x2_init[0], 2.) + pow(x[1]-x2_init[1], 2.);
		r2 = pow(r2, 0.5);
		double arccot_r2 = Const::PI/2. - std::atan(pow(c2,0.5)/2.*r2);

		for (size_t n = 0; n < num; n++)
		{
			val += c1/3. * a_const[n] * (cos(2*(n+1)*arccot_r1) - 1.);
			val += c2/3. * a_const[n] * (cos(2*(n+1)*arccot_r2) - 1.);
		}
	}
	else if (problem==5)
	{
		const double A = 0.4;
		const double kappa = 0.05;
		val = A * exp(-kappa*(x[0]*x[0]+x[1]*x[1]));
	}
	

	return val;
}