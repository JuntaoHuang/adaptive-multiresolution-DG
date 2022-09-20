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
#include "FastMultiplyLU.h"
#include "Timer.h"

int main(int argc, char *argv[])
{
	// constant variable
	const int DIM = 2;

	// static variables
	AlptBasis::PMAX = 1;

	LagrBasis::PMAX = 2;
	LagrBasis::msh_case = 2;

	HermBasis::PMAX = 3;
	HermBasis::msh_case = 1;

	Element::PMAX_alpt = AlptBasis::PMAX;	// max polynomial degree for Alpert's basis functions
	Element::PMAX_intp = LagrBasis::PMAX;	// max polynomial degree for interpolation basis functions
	Element::DIM = DIM;			// dimension
	Element::VEC_NUM = 1;		// num of unknown variables in PDEs

	DGSolution::DIM = Element::DIM;
	DGSolution::VEC_NUM = Element::VEC_NUM;

	Interpolation::DIM = DGSolution::DIM;
	Interpolation::VEC_NUM = DGSolution::VEC_NUM;

	DGSolution::ind_var_vec = { 0 };
	DGAdapt::indicator_var_adapt = { 0 };

	Element::is_intp.resize(Element::VEC_NUM);
	for (size_t num = 0; num < Element::VEC_NUM; num++) { Element::is_intp[num] = std::vector<bool>(Element::DIM, true); }

	// computation parameter
	int NMAX = 3;
	int N_init = NMAX;
	int is_init_sparse = 0;			// use full grid (0) or sparse grid (1) when initialization
	double final_time = 0.1;
	double cfl = 0.2;
	std::string boundary_type = "period";	// this variable will be used in constructors of class OperatorMatrix1D

	// adaptive parameter
	// if you need to test code without adaptive
	// just set refine_eps to be a large number (for example 1e10), then no refine
	// and set coarsen_eta to be a negative number -1, then no coarsen
	double refine_eps = 1e10;
	double coarsen_eta = -1;

	// variable control if need to adaptively find out pointers related to Alpert and interpolation basis in DG operators
	// if using artificial viscosity, we should set is_adapt_find_ptr_intp to be true
	bool is_adapt_find_ptr_alpt = true;
	bool is_adapt_find_ptr_intp = false;

	// output info in screen every time steps
	int output_time_interval = 100;

	// num of gauss points in computing error
	int num_gauss_pt_compute_error = 3;

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

	args.Parse();
	if (!args.Good())
	{
		args.PrintUsage(std::cout);
		return 1;
	}
	args.PrintOptions(std::cout);

	// check mesh level in initialization should be less or equal to maximum mesh level
	if (N_init>NMAX) { std::cout << "Mesh level in initialization should not be larger than Maximum mesh level" << std::endl; return 1; }
	bool sparse = ((is_init_sparse==1) ? true : false);

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

	OperatorMatrix1D<HermBasis, HermBasis> oper_matx_herm_herm(all_bas_herm, all_bas_herm, boundary_type);
	OperatorMatrix1D<LagrBasis, LagrBasis> oper_matx_lagr_lagr(all_bas_lagr, all_bas_lagr, boundary_type);

	// initialization of DG solution (distribution function f)
	DGAdaptIntp dg_f(sparse, N_init, NMAX, all_bas_alpt, all_bas_lagr, all_bas_herm, hash, refine_eps, coarsen_eta, is_adapt_find_ptr_alpt, is_adapt_find_ptr_intp, oper_matx_lagr_lagr, oper_matx_herm_herm);

	// initial condition for f = f(x, v) = sin(2*pi*(x+v))
	auto init_func_1 = [](double x, int d) { return (d==0) ? (sin(2.*Const::PI*x)) : (cos(2.*Const::PI*x)); };
	auto init_func_2 = [](double x, int d) { return (d==0) ? (cos(2.*Const::PI*x)) : (sin(2.*Const::PI*x)); };
	std::vector<std::function<double(double, int)>> init_func{init_func_1, init_func_2};	
	dg_f.init_separable_scalar_sum(init_func);
	// auto init_func = [=](std::vector<double> x, int i)
	// {
	// 	// // example 4.2 (solid body rotation) in Guo and Cheng. "A sparse grid discontinuous Galerkin method for high-dimensional transport equations and its application to kinetic simulations." SISC (2016)
	// 	// const std::vector<double> xc{0.75, 0.5};
	// 	// const double b = 0.23;

	// 	// example 4.3 (deformational flow)
	// 	const std::vector<double> xc{0.65, 0.5};
	// 	const double b = 0.35;
				
	// 	double r_sqr = 0.;
	// 	for (int d = 0; d < DIM; d++) { r_sqr += pow(x[d] - xc[d], 2.); };
	// 	double r = pow(r_sqr, 0.5);
	// 	if (r <= b) { return pow(b, DIM-1) * pow(cos(Const::PI*r/(2.*b)), 6.); }
	// 	else { return 0.; }
	// };
	// dg_f.init_adaptive_intp(init_func);

	// initialization of electric field E = E(x,v) with v in the auxiliary dimension
	const int auxiliary_dim = 1;
	const bool sparse_E = false;	// use full grid in x
	DGAdapt dg_E(sparse_E, N_init, NMAX, auxiliary_dim, all_bas_alpt, all_bas_lagr, all_bas_herm, hash, refine_eps, coarsen_eta, is_adapt_find_ptr_alpt, is_adapt_find_ptr_intp);
	
	// initial condition for E = cos(2*pi*x)
	auto init_func_E = [](double x, int d) { return (d==0) ? (cos(2.*Const::PI*x)) : (1.); };
	dg_E.init_separable_scalar(init_func_E);
	
	// ------------------------------
	HyperbolicDiffFluxLagrRHS fast_rhs_lagr(dg_f, oper_matx_lagr);		

	// fast Lagrange interpolation
    LagrInterpolation interp_lagr(dg_f);
	FastLagrIntp fast_lagr_intp(dg_f, interp_lagr.Lag_pt_Alpt_1D, interp_lagr.Lag_pt_Alpt_1D_d1);

	// constant in global Lax-Friedrich flux	
	const double lxf_alpha = 1.;
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
		const std::vector<int> & max_mesh = dg_f.max_mesh_level_vec();
		
		// dt = cfl/(c1/dx1 + c2/dx2 + ... + c_dim/dx_dim)
		double sum_c_dx = 0.;	// this variable stores (c1/dx1 + c2/dx2 + ... + c_dim/dx_dim)
		for (size_t d = 0; d < DIM; d++)
		{
			sum_c_dx += std::abs(wave_speed[d]) * std::pow(2., max_mesh[d]);
		}		
		double dt = cfl/sum_c_dx;
		dt = std::min( dt, final_time - curr_time );
		
        // --- part 4: time evolution
		// linear operator for f
		HyperbolicAlpt linear(dg_f, oper_matx_alpt);
		// x direction: u^- * [v] * alpha / 2
		linear.assemble_matrix_flx_scalar(0, -1, lxf_alpha/2);
		// x direction: - u^+ * [v] * alpha / 2
		linear.assemble_matrix_flx_scalar(0, 1, -lxf_alpha/2);		
		// y direction: u^- * [v] * alpha / 2
		linear.assemble_matrix_flx_scalar(1, -1, lxf_alpha/2);
		// y direction: - u^+ * [v] * alpha / 2
		linear.assemble_matrix_flx_scalar(1, 1, -lxf_alpha/2);

        RK3SSP odeSolver_f(linear, dt);
        odeSolver_f.init();

		RK3SSP odeSolver_E(dg_E, dt);
		odeSolver_E.init();

        for ( int stage = 0; stage < odeSolver_f.num_stage; ++stage )
        {
			// --- step 1: update RHS for f ---
            // Lagrange interpolation
            LagrInterpolation interp(dg_f);
            // variable to control which flux need interpolation
            // the first index is # of unknown variable, the second one is # of dimension
            std::vector< std::vector<bool> > is_intp;
            is_intp.push_back(std::vector<bool>(DIM, true));
            
            // f = f(x, v, t) in 1D1V
            // interpolation for v * f and E * f
			// f_t + v * f_x + E * f_v = 0
			const std::vector<int> zero_derivative(DIM, 0);
            auto coe_func = [&](std::vector<double> x, int d) -> double 
            {
                if (d==0) { return x[1]; }
                else { return dg_E.val(x, zero_derivative)[0]; }
            };
            interp.var_coeff_u_Lagr_fast(coe_func, is_intp, fast_lagr_intp);

            // calculate rhs and update Element::ucoe_alpt
            dg_f.set_rhs_zero();

            fast_rhs_lagr.rhs_vol_scalar();
            fast_rhs_lagr.rhs_flx_intp_scalar();

            // add to rhs in odeSolver_f
            odeSolver_f.set_rhs_zero();
            odeSolver_f.add_rhs_to_eigenvec();
            odeSolver_f.add_rhs_matrix(linear);

            // source term for f
			{			
            double source_time = curr_time;
            if (stage == 1) { source_time += dt; }
            else if (stage == 2) { source_time += dt/2.; }
			auto source_func = [&](std::vector<double> x, int i)->double
			{	
				// s(x,v,t) = diff(F, t) + v * diff(F, x) + cos(2*pi*x) * diff(F,v)
				// = 2*pi*cos(2*pi*(t + v + x))*(v + 1 + cos(2*pi*x))
				return 2*Const::PI*(1. + x[1] + cos(2*Const::PI*x[0])) * cos(2*Const::PI*(source_time + x[0] + x[1]));
			};
            Domain2DIntegral source_operator(dg_f, all_bas_alpt, source_func);
            source_operator.assemble_vector();
            odeSolver_f.add_rhs_vector(source_operator.vec_b);
            }

            odeSolver_f.step_stage(stage);
			
			// --- step 2: update RHS for E ---
			dg_E.set_rhs_zero();

			const std::vector<int> moment_order{0};
			const std::vector<double> moment_order_weight{-1.};
			const int num_vec = 0;
			dg_E.compute_moment_full_grid(dg_f, moment_order, moment_order_weight, num_vec);

            odeSolver_E.set_rhs_zero();
			odeSolver_E.add_rhs_to_eigenvec();

			// // source term for E
			// {			
			// double source_time = curr_time;
			// if (stage == 1) { source_time += dt; }
			// else if (stage == 2) { source_time += dt/2.; }
			// auto source_func = [&](std::vector<double> x, int i)->double
			// {	
			// 	// Cos[2 Pi (t + x)] - 2 Pi t Sin[2 Pi (t + x)]
			// 	return cos(2*Const::PI*(source_time+x[0])) - 2 * Const::PI * source_time * sin(2*Const::PI*(source_time+x[0]));
			// };
			// Domain2DIntegral source_operator(dg_E, all_bas_alpt, source_func);
			// source_operator.assemble_vector();
			// odeSolver_E.add_rhs_vector(source_operator.vec_b);
			// }

			odeSolver_E.step_stage(stage);

			// --- step 3: copy ODESolver::ucoe to Element::ucoe_alpt for f and set ODESolver::rhs to be zero ---
            odeSolver_f.final();

			// --- step 4: copy ODESolver::ucoe to Element::ucoe_alpt for E and set ODESolver::rhs to be zero ---
			odeSolver_E.final();
        }

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
					<< std::endl << std::endl;
		}		
	}

	std::cout << "--- evolution finished ---" << std::endl;
	
	record_time.time("running time");
	std::cout << "num of time steps: " << num_time_step
			<< "; num of basis: " << dg_f.size_basis_alpt() << std::endl;

	// compute error for f = f(x, v, t) = sin(2*pi*(x+v+t))
	auto final_func_f = [&](std::vector<double> x) -> double 
	{	
		return sin(2*Const::PI*(x[0] + x[1] + final_time));
	};
	std::vector<double> err_f = dg_f.get_error_no_separable_scalar(final_func_f, num_gauss_pt_compute_error);
	std::cout << "L1, L2 and Linf error (for f) at final time: " << err_f[0] << ", " << err_f[1] << ", " << err_f[2] << std::endl;

	// compute error for E
	auto final_func_E = [&](std::vector<double> x) -> double 
	{	
		return cos(2*Const::PI*(x[0]));
	};
	std::vector<double> err_E = dg_E.get_error_no_separable_scalar(final_func_E, num_gauss_pt_compute_error);
	std::cout << "L1, L2 and Linf error (for E) at final time: " << err_E[0] << ", " << err_E[1] << ", " << err_E[2] << std::endl;

	return 0;
}


// error table
// P1 full grid
// f
// 0.101028, 0.120891, 0.294999
// 0.0230843, 0.0303998, 0.102846
// 0.00571958, 0.00751011, 0.0261115
// 0.00141824, 0.00187159, 0.00664366
// 0.000353593, 0.000467581, 0.00167002
// 2.1298    1.9916    1.5202
// 2.0129    2.0172    1.9777
// 2.0118    2.0046    1.9746
// 2.0039    2.0010    1.9921
// E
// 0.0626626, 0.0630444, 0.0706295
// 0.0148119, 0.0161324, 0.0235879
// 0.00365265, 0.00405665, 0.00629323
// 0.000910058, 0.00101565, 0.00159831
// 0.000227321, 0.000254008, 0.00040112
// 2.0808    1.9664    1.5822
// 2.0197    1.9916    1.9062
// 2.0049    1.9979    1.9773
// 2.0012    1.9995    1.9944
// P1 sparse grid (mesh case = 2)
// f
// 0.311954, 0.358668, 0.904762
// 0.0993162, 0.124651, 0.473364
// 0.0378423, 0.0483275, 0.23112
// 0.0112165, 0.0149412, 0.0969459
// 0.00328285, 0.00435172, 0.0301211
// 0.000909835, 0.00123083, 0.0105595
// 1.6512    1.5248    0.9346
// 1.3920    1.3670    1.0343
// 1.7544    1.6935    1.2534
// 1.7726    1.7796    1.6864
// 1.8513    1.8220    1.5122
// E
// 0.0626672, 0.0630475, 0.0704511
// 0.0148075, 0.0161344, 0.0237851
// 0.0036511, 0.00406216, 0.00664047
// 0.000910425, 0.00101986, 0.00174653
// 0.000228857, 0.00025697, 0.000464127
// 5.76129e-05, 6.4772e-05, 0.000115212
// 2.0814    1.9663    1.5666
// 2.0199    1.9898    1.8407
// 2.0037    1.9939    1.9268
// 1.9921    1.9887    1.9119
// 1.9900    1.9882    2.0102
// ------------------------------------
// P2 full grid
// f
// 0.0106806, 0.013337, 0.035273
// 0.00201144, 0.0025501, 0.00707526
// 0.00029867, 0.000389814, 0.00131439
// 4.3911e-05, 5.88262e-05, 0.000223879
// 6.37332e-06, 8.77692e-06, 3.95652e-05
// 2.4087    2.3868    2.3177
// 2.7516    2.7097    2.4284
// 2.7659    2.7283    2.5536
// 2.7845    2.7447    2.5004
// E
// 0.000787902, 0.000808144, 0.00101353
// 4.26431e-05, 4.90922e-05, 7.82713e-05
// 2.6574e-06, 3.2923e-06, 7.52786e-06
// 4.5342e-07, 5.22664e-07, 9.98019e-07
// 9.39112e-08, 1.07284e-07, 1.86366e-07
// 4.2076    4.0410    3.6948
// 4.0042    3.8983    3.3782
// 2.5511    2.6551    2.9151
// 2.2715    2.2844    2.4209
// P2 sparse grid (mesh case = 3)
// f
// 0.0534594, 0.0639747, 0.147033
// 0.00902356, 0.011655, 0.0580205
// 0.00190811, 0.00242373, 0.0113415
// 0.000342251, 0.000434715, 0.00193663
// 6.16721e-05, 7.97812e-05, 0.000450397
// 1.16745e-05, 1.56318e-05, 9.06835e-05
// 2.5667    2.4566    1.3415
// 2.2416    2.2656    2.3550
// 2.4790    2.4791    2.5500
// 2.4724    2.4459    2.1043
// 2.4013    2.3516    2.3123
// E
// 0.000732281, 0.000770196, 0.00118776
// 4.40985e-05, 5.28759e-05, 9.65438e-05
// 1.10801e-05, 1.35063e-05, 2.6849e-05
// 2.61067e-06, 3.26672e-06, 7.19117e-06
// 4.57668e-07, 5.40771e-07, 1.15532e-06
// 8.21309e-08, 9.67378e-08, 2.29496e-07
// 4.0536    3.8645    3.6209
// 1.9928    1.9690    1.8463
// 2.0855    2.0477    1.9006
// 2.5120    2.5948    2.6379
// 2.4783    2.4829    2.3318
// ------------------------------------
// P3 full grid
// f
// 0.00233844, 0.00293609, 0.00774432
// 8.83374e-05, 0.00011098, 0.000341811
// 4.13877e-06, 5.39494e-06, 1.6042e-05
// 2.38861e-07, 3.16203e-07, 8.34617e-07
// 1.52864e-08, 2.01127e-08, 5.89419e-08
// 4.7264    4.7255    4.5019
// 4.4158    4.3625    4.4133
// 4.1150    4.0927    4.2646
// 3.9659    3.9747    3.8237
// E
// 0.000828705, 0.000837562, 0.000941511
// 4.89012e-05, 5.33172e-05, 7.87132e-05
// 3.01435e-06, 3.34806e-06, 5.2252e-06
// 1.8796e-07, 2.09762e-07, 3.30979e-07
// 1.1737e-08, 1.31153e-08, 2.07834e-08
// 4.0829    3.9735    3.5803
// 4.0200    3.9932    3.9130
// 4.0033    3.9965    3.9807
// 4.0013    3.9994    3.9932
// P3 sparse grid (mesh case = 2)
// f
// 0.00596427, 0.00727649, 0.0177637
// 0.000444615, 0.000548954, 0.00200915
// 4.17579e-05, 5.2199e-05, 0.0002164
// 3.11907e-06, 4.06284e-06, 2.81118e-05
// 2.49684e-07, 3.32749e-07, 3.11742e-06
// 2.36786e-08, 3.24892e-08, 3.27087e-07
// 3.7457    3.7285    3.1443
// 3.4124    3.3946    3.2148
// 3.7429    3.6835    2.9445
// 3.6429    3.6100    3.1728
// 3.3984    3.3564    3.2526
// E
// 0.000833125, 0.000842194, 0.000965363
// 4.86069e-05, 5.29358e-05, 7.89966e-05
// 2.94865e-06, 3.28415e-06, 5.12659e-06
// 1.86091e-07, 2.07766e-07, 3.37673e-07
// 1.15551e-08, 1.29422e-08, 2.19647e-08
// 7.23124e-10, 8.11597e-10, 1.42085e-09
// 4.0993    3.9918    3.6112
// 4.0430    4.0107    3.9457
// 3.9860    3.9825    3.9243
// 4.0094    4.0048    3.9424
// 3.9981    3.9952    3.9504