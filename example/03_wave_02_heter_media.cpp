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
#include "Timer.h"
#include "VecMultiD.h"

// ------------------------------------------------------------------------------------------------
// solve 2D and 3D wave equation with discontinuous coefficient and periodic boundary condition
// 
// u_tt = grad \cdot (k(x,y) grad u)
// ------------------------------------------------------------------------------------------------
double coeffcient_function_setup(std::vector<double> x, int d, int problem);

int main(int argc, char *argv[])
{
	// static variables
	AlptBasis::PMAX = 3;

	LagrBasis::PMAX = 3;
	LagrBasis::msh_case = 3;	// interpolation points are symmetric in case 3

	HermBasis::PMAX = 3;
	HermBasis::msh_case = 1;

	Element::PMAX_alpt = AlptBasis::PMAX;	// max polynomial degree for Alpert's basis functions
	Element::PMAX_intp = LagrBasis::PMAX;	// max polynomial degree for interpolation basis functions
	Element::DIM = 3;			// dimension
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
	int NMAX = 3;
	int N_init = NMAX;
	const bool sparse = false;
	double final_time = 0.1;
	double cfl = 0.05;
	const bool is_adapt_find_ptr_alpt = true;	// variable control if need to adaptively find out pointers related to Alpert basis in DG operators
	const bool is_adapt_find_ptr_intp = false;	// variable control if need to adaptively find out pointers related to interpolation basis in DG operators
	int problem = 1;
	const int output_step = 10;
	std::vector<double> output_time = linspace(0.01, 0.50, 50);

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
	OperatorMatrix1D<AlptBasis,AlptBasis> oper_matx_alpt(all_bas_alpt, all_bas_alpt, "period");
	OperatorMatrix1D<LagrBasis,AlptBasis> oper_matx_lagr(all_bas_lagr, all_bas_alpt, "period");
	OperatorMatrix1D<LagrBasis, LagrBasis> oper_matx_lagr_all(all_bas_lagr, all_bas_lagr, "period");
	OperatorMatrix1D<HermBasis, HermBasis> oper_matx_herm_all(all_bas_herm, all_bas_herm, "period");

	// initialization of DG solution
	DGAdaptIntp dg_solu(sparse, N_init, NMAX, all_bas_alpt, all_bas_lagr, all_bas_herm, hash, refine_eps, coarsen_eta, is_adapt_find_ptr_alpt, is_adapt_find_ptr_intp, oper_matx_lagr_all, oper_matx_herm_all);
	
	// initial condition
	auto init_func = [&](std::vector<double> x, int i)->double 
		{
			double r2 = 0;
			for (size_t d = 0; d < DIM; d++) { r2 += std::pow(x[d]-0.5, 2.); }
			return 100 * exp(-500*r2);
		};
	LagrInterpolation interp(dg_solu);
	dg_solu.init_adaptive_intp_Lag(init_func, interp);
	FastLagrInit fast_init(dg_solu, oper_matx_lagr);
	fast_init.eval_ucoe_Alpt_Lagr();

	dg_solu.copy_ucoe_to_ucoe_ut();
	dg_solu.set_ucoe_alpt_zero();
	
	// ------------------------------
	// 	This part is for solving constant coefficient for linear equation	
	std::vector<double> waveConst(DIM, 1.);
	double sigma_ipdg = 0.;
	if (DIM == 2) { sigma_ipdg = 10; }
	else if (DIM == 3) { sigma_ipdg = 20; }

	DiffusionRHS diffuseRHS(dg_solu, oper_matx_lagr);
	
	// coefficient function k
	auto coe_func = [&](std::vector<double> x, int d) ->double { return coeffcient_function_setup(x,d,problem); };

	// k-
	auto coe_func_minus = [&](std::vector<double> x, int d)->double 
		{
			std::vector<double> xminus = x;
			for (size_t d = 0; d < xminus.size(); d++) { xminus[d] -= 100 * Const::ROUND_OFF; }
			return coe_func(xminus, d);
		};

	// k+
	auto coe_func_plus = [&](std::vector<double> x, int d)->double 
		{
			std::vector<double> xplus = x;
			for (size_t d = 0; d < xplus.size(); d++) { xplus[d] += 100 * Const::ROUND_OFF; }
			return coe_func(xplus, d);
		};

	//	variable to control which flux need interpolation
	std::vector< std::vector<bool> > is_intp;
	std::vector<bool> vec_true(DIM, true);
	is_intp.push_back(vec_true);

	std::vector< std::vector<bool> > is_intp_d0;
	std::vector<bool> vec_false(DIM, false);
	vec_false[0] = true;
	is_intp_d0.push_back(vec_false);

	FastLagrIntp fastLagr(dg_solu, interp.Lag_pt_Alpt_1D, interp.Lag_pt_Alpt_1D_d1);
	
	std::cout << "--- evolution started ---" << std::endl;
	Timer record_time;
	double curr_time = 0.;
	int num_time_step = 0;
	IO inout(dg_solu);
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

			EulerODE2nd odesolver(operator_ujp_vjp, dt);
			odesolver.init();

			dg_solu.set_rhs_zero();

			// interpolation of k * u_x and k * u_y
			interp.var_coeff_gradu_Lagr_fast(coe_func, is_intp, fastLagr);
			
			diffuseRHS.rhs_flx_gradu();
			diffuseRHS.rhs_vol();

			// interpolation of k- * u
			interp.var_coeff_u_Lagr_fast(coe_func_minus, is_intp_d0, fastLagr);
			diffuseRHS.rhs_flx_k_minus_u();

			// interpolation of k+ * u
			interp.var_coeff_u_Lagr_fast(coe_func_plus, is_intp_d0, fastLagr);
			diffuseRHS.rhs_flx_k_plus_u();

			odesolver.rhs_to_eigenvec();
			
			// [u] * [v]
			odesolver.add_rhs_matrix(operator_ujp_vjp);			
			
			odesolver.step_stage(0);

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

		RK4ODE2nd odesolver(operator_ujp_vjp, dt);
		odesolver.init();

		for (size_t stage = 0; stage < odesolver.num_stage; stage++)
		{		
			dg_solu.set_rhs_zero();

			// interpolation of k * u_x and k * u_y
			interp.var_coeff_gradu_Lagr_fast(coe_func, is_intp, fastLagr);
			
			diffuseRHS.rhs_flx_gradu();
			diffuseRHS.rhs_vol();

			// interpolation of k- * u
			interp.var_coeff_u_Lagr_fast(coe_func_minus, is_intp_d0, fastLagr);
			diffuseRHS.rhs_flx_k_minus_u();

			// interpolation of k+ * u
			interp.var_coeff_u_Lagr_fast(coe_func_plus, is_intp_d0, fastLagr);
			diffuseRHS.rhs_flx_k_plus_u();
			
			odesolver.rhs_to_eigenvec();
			
			// [u] * [v]
			odesolver.add_rhs_matrix(operator_ujp_vjp);			
			
			odesolver.step_stage(stage);

			odesolver.final();
		}
		
		curr_time += dt;

		// --- part 5: coarsen
		dg_solu.coarsen();
		const int num_basis_coarsen = dg_solu.size_basis_alpt();		
		
		// record code running time
		if (num_time_step % output_step == 0)
		{
			record_time.time("running time");
			std::cout << "num of time steps: " << num_time_step << "; time step: " << dt << "; curr time: " << curr_time << std::endl
					<< "curr max mesh: " << dg_solu.max_mesh_level() << std::endl
					<< "num of basis after refine: " << num_basis_refine << "; num of basis after coarsen: " << num_basis_coarsen << std::endl << std::endl;				
		}	

		auto min_time_iter = std::min_element(output_time.begin(), output_time.end());
		if ((curr_time <= *min_time_iter) && (curr_time+dt >= *min_time_iter))
		{
			std::string file_name = "profile2D_" + std::to_string(curr_time) + ".txt";		
			if (DIM==2) { inout.output_num(file_name); }
			else if (DIM==3) { inout.output_num_cut_2D(file_name, 0.5, 2); }

			std::string file_name_center = "center2D_" + std::to_string(curr_time) + ".txt";
			inout.output_element_center(file_name_center);

			output_time.erase(min_time_iter);
		}		

		num_time_step ++;
	}
	std::cout << "--- evolution finished ---" << std::endl;

	inout.output_num("profile_final.txt");
	inout.output_element_center("center_final.txt");
	inout.output_element_support("suppt_final.txt");
	// ------------------------------

	return 0;
}




double coeffcient_function_setup(std::vector<double> xvec, int d, int problem)
{	
	const double pi = Const::PI;
	double x = xvec[0];
	double y = xvec[1];

	if (problem==1)
	{
		if (std::abs(x-0.5)<=0.15) { return 1./4.; }
		else { return 1.; }
	}
	else
	{
		std::cout << "problem option not correct" << std::endl; exit(1);
	}
}