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

// compare IMEX with explicit SSP:
// -NM 6 -N0 2 -s 0 -r 1e-3 -c 1e-4 -tf 0.2 -v 1 -imex 1 -cfl 0.2 -nu 2.0
// -NM 6 -N0 2 -s 0 -r 1e-3 -c 1e-4 -tf 0.2 -v 3 -imex 0 -sp 1 -cfl 0.01 -nu 2.0
// IMEX is more stable and faster than SSP
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
	double refine_eps = 1e-2;
	double coarsen_eta = refine_eps/10.;

	// variable control if need to adaptively find out pointers related to Alpert and interpolation basis in DG operators
	// if using artificial viscosity, we should set is_adapt_find_ptr_intp to be true
	bool is_adapt_find_ptr_alpt = true;
	bool is_adapt_find_ptr_intp = true;

	// constant kappa in shock detector
	// if se <= s0 + kappa, then no shock
	// s0: log10 of theory bound of l2 norm
	// se: log10 of numerical l2 norm
	// viscosity_option:
	//		0: no use artificial viscosity
	//		1: use artificial viscosity, assemble matrix, when calculate integral of \nu * u_x * v_x, find all related elements, see choice 1 in ArtificialViscosity::assemble_matrix_one_element
	//		2: use artificial viscosity, assemble matrix, when calculate integral of \nu * u_x * v_x, find all all parents' elements, see choice 2 in ArtificialViscosity::assemble_matrix_one_element
	//		3: use artificial viscosity, not assemble matrix, first interpolate \nu * u_x by Lagrange basis and then evaluate volume integral of \nu * u_x * v_x by fast LU algorithm
	int viscosity_option = 0;
	double shock_kappa = 0.0;
	double artificial_viscosity_nu0 = 1.;
	int support_overlap = 0;	// consider overlap of support (1) or not (0)		
	int is_imex_option = 1;		// // use IMEX (1) or explicit SSP RK (0)

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
	args.AddOption(&viscosity_option, "-v", "--artificial-viscosity-option", "artificial viscosity option");
	args.AddOption(&artificial_viscosity_nu0, "-nu", "--artificial-viscosity-value", "artificial viscosity nu0");
	args.AddOption(&is_imex_option, "-imex", "--implicit-explicit", "Use IMEX (1) or explicit SSP RK (0)");
	args.AddOption(&support_overlap, "-sp", "--support-viscosity-overlap", "consider overlap of support (1) or not (0 by default)");

	args.Parse();
	if (!args.Good())
	{
		args.PrintUsage(std::cout);
		return 1;
	}
	args.PrintOptions(std::cout);

	// check mesh level in initialization should be less or equal to maximum mesh level
	if (N_init>NMAX) { std::cout << "Mesh level in initialization should not be larger than Maximum mesh level" << std::endl; return 1; }
	if (is_imex_option==1 && viscosity_option==3) {std::cout << "In artificial viscosity option 3, we do not assemble matrix thus IMEX could not be applied" << std::endl; return 1; }
	bool sparse = ((is_init_sparse==1) ? true : false);
	bool overlap = ((support_overlap==1) ? true : false);
	bool is_imex = ((is_imex_option==1) ? true : false);

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

	// initial condition: a + b*sin(2*pi*(x+y+c))
	// exact solution at time t
	const double burgers_init_a = 0.;
	const double burgers_init_b = 1.;
	const double burgers_init_c = 0.;
	BurgersExact burgers(burgers_init_a, burgers_init_b, burgers_init_c);

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

	// output
	IO inout(dg_solu);

	// ------------------------------
	// Hermite interpolation for nonlinear physical flux
	// Lagrange interpolation for const*u_x and const*u_y in artificial viscosity term
	// const should vanish in smooth region and order h near shock
	HyperbolicSameFluxHermRHS fast_rhs_herm(dg_solu, oper_matx_herm);
	HyperbolicDiffFluxLagrRHS fast_rhs_lagr(dg_solu, oper_matx_lagr);
		
	HermInterpolation interp_herm(dg_solu);
	LargViscosInterpolation interp_lagr(dg_solu);

	// Burgers' equation have the same flux in x and y direction, thus interpolation is only needed in x direction
	std::vector< std::vector<bool> > is_intp_herm;
	is_intp_herm.push_back(std::vector<bool>{true, false});

	// Lagrange interpolation for both u_x and u_y
	std::vector< std::vector<bool> > is_intp_lagr;
	is_intp_lagr.push_back(std::vector<bool>(DIM, true));

	// fast hermite interpolation
	FastHermIntp fast_herm_intp(dg_solu, interp_herm.Her_pt_Alpt_1D);
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
		// update artificial viscosity element
		int num_visc_elem = 0;
		if (viscosity_option!=0)
		{			
			dg_solu.update_viscosity_element(shock_kappa);
			num_visc_elem = dg_solu.num_artific_visc_elem();
		}

		const std::vector<int> & max_mesh = dg_solu.max_mesh_level_vec();
		
		// dt = cfl/(c1/dx1 + c2/dx2 + ... + c_dim/dx_dim)
		double sum_c_dx = 0.;	// this variable stores (c1/dx1 + c2/dx2 + ... + c_dim/dx_dim)
		for (size_t d = 0; d < DIM; d++)
		{
			sum_c_dx += std::abs(wave_speed[d]) * std::pow(2., max_mesh[d]);
		}		
		double dt = cfl/sum_c_dx;
		if (num_visc_elem==0) { dt = cfl_hyper/sum_c_dx; }
		dt = std::min( dt, final_time - curr_time );

		// --- part 2: predict by Euler forward
		{	
			// before Euler forward, copy Element::ucoe_alpt to Element::ucoe_alpt_predict
			dg_solu.copy_ucoe_to_predict();

			HyperbolicAlpt linear(dg_solu, oper_matx_alpt);
			linear.assemble_matrix_flx_scalar(0, -1, lxf_alpha[0]/2);
			linear.assemble_matrix_flx_scalar(0, 1, -lxf_alpha[0]/2);
			linear.assemble_matrix_flx_scalar(1, -1, lxf_alpha[1]/2);
			linear.assemble_matrix_flx_scalar(1, 1, -lxf_alpha[1]/2);

			ForwardEuler odeSolver(linear, dt);
			odeSolver.init();

			// Hermite interpolation
			if (HermBasis::PMAX==3)
			{
				interp_herm.nonlinear_Herm_2D_fast(func_flux, func_flux_d1, func_flux_d2, is_intp_herm, fast_herm_intp);
			}
			else if (HermBasis::PMAX==5)
			{
				interp_herm.nonlinear_Herm_2D_PMAX5_scalar_fast(func_flux, func_flux_d1, func_flux_d2, func_flux_d3, func_flux_d4, is_intp_herm, fast_herm_intp);
			}		
			
			// calculate rhs
			dg_solu.set_rhs_zero();

			fast_rhs_herm.rhs_vol_scalar();
			fast_rhs_herm.rhs_flx_intp_scalar();

			odeSolver.set_rhs_zero();
			odeSolver.add_rhs_to_eigenvec();
			odeSolver.add_rhs_matrix(linear);

			// Euler forward time stepping
			odeSolver.step_stage(0);

			// copy eigen vector ucoe to Element::ucoe_alpt
			// now prediction of numerical solution at t^(n+1) is stored in Element::ucoe_alpt
			odeSolver.final();			
		}

		// --- part 3: refine base on Element::ucoe_alpt
		dg_solu.refine();
		const int num_basis_refine = dg_solu.size_basis_alpt();

		// after refine, copy Element::ucoe_alpt_predict back to Element::ucoe_alpt
		dg_solu.copy_predict_to_ucoe();

		// --- part 4: time evolution
		HyperbolicAlpt linear(dg_solu, oper_matx_alpt);
		linear.assemble_matrix_flx_scalar(0, -1, lxf_alpha[0]/2);
		linear.assemble_matrix_flx_scalar(0, 1, -lxf_alpha[0]/2);
		linear.assemble_matrix_flx_scalar(1, -1, lxf_alpha[1]/2);
		linear.assemble_matrix_flx_scalar(1, 1, -lxf_alpha[1]/2);
				
		// if there exist no artificial viscosity element, then use SSP-RK
		if (num_visc_elem == 0)
		{
			RK3SSP odeSolver(linear, dt);
			odeSolver.init();		

			for ( int stage = 0; stage < odeSolver.num_stage; ++stage )
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

				fast_rhs_herm.rhs_vol_scalar();
				fast_rhs_herm.rhs_flx_intp_scalar();

				// add to rhs in odeSolver
				odeSolver.set_rhs_zero();
				odeSolver.add_rhs_to_eigenvec();
				odeSolver.add_rhs_matrix(linear);

				odeSolver.step_stage(stage);

				odeSolver.final();
			}
		}
		// if there exist artificial viscosity element, then use IMEX or SSP
		else
		{	
			// maximum mesh level and smallest step size in space
			const int max_mesh_scalar = dg_solu.max_mesh_level();
			const double h = 1./pow(2., max_mesh_scalar);

			if (viscosity_option==1 || viscosity_option==2)
			{
				// assemble artificial viscosity operator matrix
				ArtificialViscosity artific_visc_operator(dg_solu, oper_matx_alpt, all_bas_alpt);
				artific_visc_operator.assemble_matrix( - artificial_viscosity_nu0 * h, viscosity_option);	
			
				if (is_imex)
				{			
					// use IMEX ODE solver
					IMEX43 odeSolver(artific_visc_operator, dt);
					odeSolver.init();

					for ( int stage = 0; stage < odeSolver.num_stage; ++stage )
					{			
						// calculate rhs for explicit part only for stage = 2, 3, 4
						// no need for stage = 0, 1
						if (stage>=2)
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

							fast_rhs_herm.rhs_vol_scalar();
							fast_rhs_herm.rhs_flx_intp_scalar();

							// add to rhs in odeSolver
							odeSolver.set_rhs_zero();
							odeSolver.add_rhs_to_eigenvec();
							odeSolver.add_rhs_matrix(linear);	
						}
						odeSolver.step_stage(stage);

						odeSolver.final();
					}
				}
				else
				{
					// use SSP RK3 ODE solver
					RK3SSP odeSolver(linear, dt);
					odeSolver.init();		

					for ( int stage = 0; stage < odeSolver.num_stage; ++stage )
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

						fast_rhs_herm.rhs_vol_scalar();
						fast_rhs_herm.rhs_flx_intp_scalar();

						// add to rhs in odeSolver
						odeSolver.set_rhs_zero();
						odeSolver.add_rhs_to_eigenvec();
						odeSolver.add_rhs_matrix(linear);
						odeSolver.add_rhs_matrix(artific_visc_operator);

						odeSolver.step_stage(stage);

						odeSolver.final();
					}				
				}	
			}
			else if (viscosity_option==3)
			{
				RK3SSP odeSolver(linear, dt);
				odeSolver.init();		
			
				for ( int stage = 0; stage < odeSolver.num_stage; ++stage )
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

					fast_rhs_herm.rhs_vol_scalar();
					fast_rhs_herm.rhs_flx_intp_scalar();

					// Lagrange interpolation for artificial viscosity term
					interp_lagr.support_gradu_Lagr_fast(dg_solu.viscosity_element, -artificial_viscosity_nu0*h, is_intp_lagr, fast_lagr_intp, overlap);
					fast_rhs_lagr.rhs_vol_scalar();

					// add to rhs in odeSolver
					odeSolver.set_rhs_zero();
					odeSolver.add_rhs_to_eigenvec();
					odeSolver.add_rhs_matrix(linear);

					odeSolver.step_stage(stage);

					odeSolver.final();
				}				
			}
		}

		// write shock support into file before coarsen (since coarsen may delete some element then cause nullptr in viscosity elements)
		if (num_time_step % output_time_interval == 0)
		{
			std::string file_name_shock = "shock_" + std::to_string(curr_time) + ".txt";
			inout.output_shock_support(file_name_shock);
		}

		// --- part 5: coarsen
		dg_solu.coarsen();
		const int num_basis_coarsen = dg_solu.size_basis_alpt();		

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
					<< "num of basis after refine: " << num_basis_refine
					<< "; num of basis after coarsen: " << num_basis_coarsen
					<< "; num of artificial viscosity elements: " << num_visc_elem
					<< std::endl << std::endl;
			
			// exact solution at current time
			auto exact_solu = [&](std::vector<double> x)->double { return burgers.exact_2d(x[0], x[1], curr_time); };

			std::string file_name = "profile2D_" + std::to_string(curr_time) + ".txt";
			inout.output_num_exa(file_name, exact_solu);
			
			file_name = "center2D_" + std::to_string(curr_time) + ".txt";
			inout.output_element_center(file_name);
		}		
	}

	std::cout << "--- evolution finished ---" << std::endl;
	
	record_time.time("running time");
	std::cout << "num of time steps: " << num_time_step
			<< "; num of basis: " << dg_solu.size_basis_alpt() << std::endl;

	auto final_func = [&](std::vector<double> x) {return burgers.exact_2d(x[0], x[1], final_time); };

	std::vector<double> err = dg_solu.get_error_no_separable_scalar(final_func, num_gauss_pt_compute_error);
	std::cout << "L1, L2 and Linf error at final time: " << err[0] << ", " << err[1] << ", " << err[2] << std::endl;
		
	std::string file_name = "error_N" + std::to_string(NMAX) + ".txt";
	inout.write_error(NMAX, err, file_name);
	// ------------------------------

	return 0;
}
