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

// numerical result for full grid:
// for i in {2..7}; do ./07_vlasov_ampere -s 0 -NM $i -N0 $i -p 1000; done
// 
// P2 + LagrBasis::msh_case = 1
// L1, L2 and Linf error at final time:
// 0.00977663, 0.0145172, 0.0321203
// 0.00147612, 0.00303159, 0.0146294
// 0.000166637, 0.00064198, 0.00661077
// 2.08798e-05, 8.31085e-05, 0.00104469
// 3.65917e-06, 1.51696e-05, 0.000181649
// 6.90303e-07, 2.99507e-06, 3.38135e-05
// order
// 2.7275    2.2596    1.1346
// 3.1470    2.2395    1.1460
// 2.9965    2.9495    2.6617
// 2.5125    2.4538    2.5238
// 2.4062    2.3405    2.4255
// P2 + LagrBasis::msh_case = 2
// L1, L2 and Linf error at final time:
// 0.00511512, 0.007072, 0.0152497
// 0.00142072, 0.00270807, 0.0126668
// 0.000163363, 0.000623661, 0.00641348
// 2.02479e-05, 8.01085e-05, 0.00100447
// 3.6186e-06, 1.49575e-05, 0.000178568
// order
// 1.8481    1.3849    0.2677
// 3.1205    2.1184    0.9819
// 3.0122    2.9607    2.6747
// 2.4843    2.4211    2.4919

// numerical result for sparse grid:
// for i in {5..10}; do ./07_vlasov_ampere -s 1 -NM $i -N0 $i -p 1000; done
// 
// 0.00196079, 0.00414682, 0.0412254
// 0.000458176, 0.000981532, 0.00976287
// 5.71906e-05, 0.000135878, 0.00151081
// 1.02366e-05, 2.5748e-05, 0.000585272
// 1.47486e-06, 4.05604e-06, 0.000116428
// 2.72721e-07, 7.92076e-07, 1.81119e-05
// order
// 2.0975    2.0789    2.0782
// 3.0021    2.8527    2.6920
// 2.4820    2.3998    1.3681
// 2.7951    2.6663    2.3297
// 2.4351    2.3564    2.6844

// deformational flow (full grid)
// 0.00707721, 0.0097736, 0.0233013
// 0.000995681, 0.0022954, 0.011081
// 0.000122556, 0.000292421, 0.00166943
// 1.70688e-05, 4.24906e-05, 0.000316131
// 2.63512e-06, 6.84403e-06, 9.68338e-05
// order
// 2.8294    2.0901    1.0723
// 3.0222    2.9726    2.7307
// 2.8440    2.7828    2.4008
// 2.6954    2.6342    1.7069
// sparse grid
// LagrBasis::msh_case = 1
// 80.0892, 113.378, 312.923
// 1.91581e+07, 2.6025e+07, 7.58415e+07
// 8.37956e+23, 1.42582e+24, 6.74819e+24
// 3.34031e+61, 7.23456e+61, 5.12755e+62
// 8.9904e+140, 2.30897e+141, 2.25564e+142
// LagrBasis::msh_case = 2
// 0.0362431, 0.0511725, 0.11858
// 0.0532136, 0.0906722, 0.383674
// 0.0531663, 0.100788, 0.597923
// 0.798468, 2.40259, 18.6084
// 384969, 1.29615e+06, 1.19112e+07
// LagrBasis::msh_case = 3
// 0.0146614, 0.0173568, 0.0328734
// 0.0102148, 0.01536, 0.0544565
// 0.0044042, 0.00740571, 0.0466604
// 0.00159391, 0.0028898, 0.0176127
// 0.000400531, 0.000811244, 0.00503487
// 7.19094e-05, 0.000170648, 0.00158929
int main(int argc, char *argv[])
{
	// constant variable
	const int DIM = 2;

	// static variables
	AlptBasis::PMAX = 2;

	LagrBasis::PMAX = 3;
	LagrBasis::msh_case = 3;

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
	double final_time = 1.5;
	double cfl = 0.2;
	int rk_order = 3;
	std::string boundary_type = "period";	// this variable will be used in constructors of class OperatorMatrix1D

	// adaptive parameter
	// if test code without adaptive
	// just set refine_eps to be a large number (for example 1e10), then no refine
	// and set coarsen_eta to be a negative number -1, then no coarsen
	double refine_eps = 1e10;
	double coarsen_eta = -1;

	// variable control if need to adaptively find out pointers related to Alpert and interpolation basis in DG operators
	// if using artificial viscosity, we should set is_adapt_find_ptr_intp to be true
	bool is_adapt_find_ptr_alpt = true;
	bool is_adapt_find_ptr_intp = false;

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

	// // project initial function into numerical solution
	// auto init_func_1 = [](double x, int d) { return (d==0) ? (cos(2.*Const::PI*x)) : (cos(2.*Const::PI*x)); };
	// auto init_func_2 = [](double x, int d) { return (d==0) ? (-sin(2.*Const::PI*x)) : (sin(2.*Const::PI*x)); };
	// std::vector<std::function<double(double, int)>> init_func{init_func_1, init_func_2};	
	// dg_f.init_separable_scalar_sum(init_func);
	auto init_func = [=](std::vector<double> x, int i)
	{
		// // example 4.2 (solid body rotation) in Guo and Cheng. "A sparse grid discontinuous Galerkin method for high-dimensional transport equations and its application to kinetic simulations." SISC (2016)
		// const std::vector<double> xc{0.75, 0.5};
		// const double b = 0.23;

		// example 4.3 (deformational flow)
		const std::vector<double> xc{0.65, 0.5};
		const double b = 0.35;
				
		double r_sqr = 0.;
		for (int d = 0; d < DIM; d++) { r_sqr += pow(x[d] - xc[d], 2.); };
		double r = pow(r_sqr, 0.5);
		if (r <= b) { return pow(b, DIM-1) * pow(cos(Const::PI*r/(2.*b)), 6.); }
		else { return 0.; }
	};
	dg_f.init_adaptive_intp(init_func);

	// // initialization of electric field E
	// const int auxiliary_dim = 1;
	// DGAdapt dg_electric(sparse, N_init, NMAX, auxiliary_dim, all_bas_alpt, all_bas_lagr, all_bas_herm, hash, refine_eps, coarsen_eta, is_adapt_find_ptr_alpt, is_adapt_find_ptr_intp);

	// // E(x,y) = cos(2*pi*x)
	// auto init_func_electric = [](double x, int d) { return (d==0) ? (cos(2.*Const::PI*x)) : (1.); };
	// dg_electric.init_separable_scalar(init_func_electric);
	
    // output
	IO inout(dg_f);

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
		// linear operator
		HyperbolicAlpt linear(dg_f, oper_matx_alpt);
		// x direction: u^- * [v] * alpha / 2
		linear.assemble_matrix_flx_scalar(0, -1, lxf_alpha/2);
		// x direction: - u^+ * [v] * alpha / 2
		linear.assemble_matrix_flx_scalar(0, 1, -lxf_alpha/2);		
		// y direction: u^- * [v] * alpha / 2
		linear.assemble_matrix_flx_scalar(1, -1, lxf_alpha/2);
		// y direction: - u^+ * [v] * alpha / 2
		linear.assemble_matrix_flx_scalar(1, 1, -lxf_alpha/2);

        RK3SSP odeSolver(linear, dt);
        odeSolver.init();		

        for ( int stage = 0; stage < odeSolver.num_stage; ++stage )
        {			

// Timer record_time;

            // Lagrange interpolation
            LagrInterpolation interp(dg_f);
            // variable to control which flux need interpolation
            // the first index is # of unknown variable, the second one is # of dimension
            std::vector< std::vector<bool> > is_intp;
            is_intp.push_back(std::vector<bool>(DIM, true));
            
            // f = f(x, v, t) in 1D1V
            // interpolation for f * v
            // f_t + v * f_x + E * f_v = 0
			// test: f_t + sin(2*pi*v) * f_x + cos(2*pi*x) * f_v = 0
            // auto coe_func = [&](std::vector<double> x, int d) -> double 
            // {
            //     if (d==0) 
			// 	{ 
			// 		return sin(2*Const::PI*(x[1])); 
			// 	}
            //     else 
			// 	{
			// 		// return cos(2*Const::PI*(x[0]));
			// 		std::vector<int> zero_derivative(DIM, 0);
			// 		return dg_electric.val(x, zero_derivative)[0];					
			// 	}
            // };
			// // solid body rotation: f_t + (-x2 + 0.5) * f_x1 + (x1 - 0.5) * f_x2 = 0
            // auto coe_func = [&](std::vector<double> x, int d) -> double 
            // {
            //     if (d==0) { return -x[1] + 0.5; }
            //     else { return x[0] - 0.5; }
            // };
			// deformational flow: f_t + ( sin(pi*x1)^2 * sin(2*pi*x2) * g(t) ) * f_x1 + ( -sin(pi*x2)^2 * sin(2*pi*x1) * g(t) ) * f_x2 = 0
            double gt_time = curr_time;
            if (stage == 1) { gt_time += dt; }
            else if (stage == 2) { gt_time += dt/2.; }
			double gt = cos(Const::PI*gt_time/final_time);
            auto coe_func = [&](std::vector<double> x, int d) -> double 
            {				
                if (d==0) { return pow(sin(Const::PI*x[0]), 2.) * sin(2*Const::PI*x[1]) * gt; }
                else { return - pow(sin(Const::PI*x[1]), 2.) * sin(2*Const::PI*x[0]) * gt; }
            };
            interp.var_coeff_u_Lagr_fast(coe_func, is_intp, fast_lagr_intp);
// record_time.time("interpolation");
// record_time.reset();

            // calculate rhs and update Element::ucoe_alpt
            dg_f.set_rhs_zero();

            fast_rhs_lagr.rhs_vol_scalar();
            fast_rhs_lagr.rhs_flx_intp_scalar();
// record_time.time("bilinear form for nonlinear term");
// record_time.reset();

            // add to rhs in odeSolver
            odeSolver.set_rhs_zero();
            odeSolver.add_rhs_to_eigenvec();
            odeSolver.add_rhs_matrix(linear);
// record_time.time("bilinear form for linear term");
// record_time.reset();

            // // source term
            // double source_time = curr_time;
            // if (stage == 1) { source_time += dt; }
            // else if (stage == 2) { source_time += dt/2.; }
            // // compute projection of source term f(x, y, t) with separable formulation
            // // auto source_func = [&](std::vector<double> x, int i)->double { return 2*Const::PI*(x[1]+1.)*cos(2*Const::PI*(source_time+x[0]+x[1])); };
			// auto source_func = [&](std::vector<double> x, int i)->double { return (sin(2*Const::PI*x[1]) + cos(2*Const::PI*x[0]) + 1.) * (-2*Const::PI) * sin(2*Const::PI*(source_time+x[0]+x[1])); };
            // Domain2DIntegral source_operator(dg_f, all_bas_alpt, source_func);
            // source_operator.assemble_vector();
            // odeSolver.add_rhs_vector(source_operator.vec_b);
// record_time.time("source term");
            
            odeSolver.step_stage(stage);
            odeSolver.final();
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

	auto final_func = [&](std::vector<double> x) -> double 
	{	
		// test: f_t + f_x + f_v = 0
		// return sin(2*Const::PI*(x[0] + 2 * x[1] - 3 * final_time));
		// return cos(2*Const::PI*(x[0] + x[1] - 2 * final_time));
		// return cos(2*Const::PI*(x[0] + x[1] + final_time));
		
		// solid body rotation
		return init_func(x, 0);
	};

	std::vector<double> err = dg_f.get_error_no_separable_scalar(final_func, num_gauss_pt_compute_error);
	std::cout << "L1, L2 and Linf error at final time: " << err[0] << ", " << err[1] << ", " << err[2] << std::endl;
		
	std::string file_name = "error_N" + std::to_string(NMAX) + ".txt";
	inout.write_error(NMAX, err, file_name);

	return 0;
}
