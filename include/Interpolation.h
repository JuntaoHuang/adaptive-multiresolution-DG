#pragma once
#include "DGSolution.h"
#include "FastMultiplyLU.h"

struct pwts
{
	std::vector<std::vector<double>> wt;
	std::vector<int> p_k;
	std::vector<int> p_i;
	std::vector<int> p_num;
};


class Interpolation
{
	public:
		Interpolation(DGSolution & dgsolution): dgsolution_ptr(&dgsolution) {};
			
		~Interpolation() {};

		static int DIM;
		static int VEC_NUM;

	protected:

    DGSolution* const dgsolution_ptr;

	std::map<int, pwts> pw1d;
    
	int hash_key1d(const int n, const int j);
};


///---------------------> for Lagrange interpolation

class LagrInterpolation:
    public Interpolation
{
	public:
		LagrInterpolation(DGSolution & dgsolution): 
		Interpolation(dgsolution)
		{
			eval_Lag_pt_at_Alpt_1D();

			eval_Lag_pt_at_Alpt_1D_d1();
		};
		~LagrInterpolation() {};

		// Lagrange interpolation for a given function coefficient_func * u_x or coefficient_func * u_y in 2D
		// u is linear combination of Alpert basis
		// this will update fucoe_intp in class Element
		void var_coeff_gradu_Lagr(std::function<double(std::vector<double>, int)> coefficient_func, std::vector< std::vector<bool> > is_intp);

		void var_coeff_gradu_Lagr_fast(std::function<double(std::vector<double>, int)> coe_func, 
                   					   std::vector< std::vector<bool> > is_intp, FastLagrIntp & fastLagr);		

		void var_coeff_u_Lagr(std::function<double(std::vector<double>, int)> coefficient_func, std::vector< std::vector<bool> > is_intp);

		// this function will update fucoe_intp in class Element
		// is_intp: variable to control which flux need interpolation
		// the first index is # of unknown variable, the second one is # of dimension
		void var_coeff_u_Lagr_fast(std::function<double(std::vector<double>, int)> coefficient_func, 
		                           std::vector< std::vector<bool> > is_intp, FastLagrIntp & fastLagr);
								   		
		void var_coeff_u_Lagr_fast_coarse_grid(std::function<double(std::vector<double>, int)> coe_func, std::vector< std::vector<bool> > is_intp, FastLagrIntp & fastLagr, const int mesh_nmax);

		// this function will compute interpolation for 1D1V Vlasov equation, e.g. f_t + v * f_x + E * f_v = 0
		// compute interpolation basis of v * f in the first dimension and E * f in the second dimension
		void interp_Vlasov_1D1V(DGSolution & E, FastLagrIntp & fastLagr_f, FastLagrIntp & fastLagr_E);

		// overload for rescaled 1D1V Vlasov equation
		// this function will compute interpolation for 1D1V Vlasov equation, e.g. f_t + coe_v * f_x + coe_E * f_v = 0
		// compute interpolation basis of coe_v * f in the first dimension and coe_E * f in the second dimension
		void interp_Vlasov_1D1V(DGSolution & E, std::function<double(double)> coe_v, std::function<double(double)> coe_E, FastLagrIntp & fastLagr_f, FastLagrIntp & fastLagr_E);

		// this function will compute interpolation for 1D2V Vlasov equation
		// f_t + v2 * f_x2 + (E1 + v2 * B3) * f_v1 + (E2 - v1 * B3) * f_v2 = 0
		void interp_Vlasov_1D2V(DGSolution & dg_BE, FastLagrIntp & fastLagr_f, FastLagrIntp & fastLagr_BE);
		
		// overload for rescaled 1D2V Vlasov equation
		// this function will compute interpolation for 1D2V Vlasov equation
		// f_t + coe_x2 * f_x2 + coe_v1 * f_v1 + coe_v2 * f_v2 = 0
		// coe_x2: function of v2
		// coe_v1: function of (v2, E1, B3)
		// coe_v2: function of (v1, E2, B3)
		void interp_Vlasov_1D2V(DGSolution & dg_BE, std::function<double(double)> coe_x2, std::function<double(double, double, double)> coe_v1, std::function<double(double, double, double)> coe_v2, FastLagrIntp & fastLagr_f, FastLagrIntp & fastLagr_BE);

		// this function will compute interpolation for 2D2V Vlasov equation
		// f_t + v1 * f_x1 + v2 * f_x2 + E1 * f_x1 + E2 * f_x2 = 0
		void interp_Vlasov_2D2V(DGSolution & dg_E, FastLagrIntp & fastLagr_f, FastLagrIntp & fastLagr_BE);

		// Lagrange interpolation for a given function func(u), u is linear combination of Alpert basis
		// this will update VecMultiD<double> fp_intp in class Element
		void nonlinear_Lagr(std::function<double(std::vector<double>, int, int)> func, std::vector< std::vector<bool> > is_intp);

		void nonlinear_Lagr_fast(std::function<double(std::vector<double>, int, int)> func, 
            					 std::vector< std::vector<bool> > is_intp, FastLagrIntp & fastLagr);
		void nonlinear_Lagr_fast_full(std::function<double(std::vector<double>, int, int, std::vector<double>)> func,
			std::vector< std::vector<bool> > is_intp, FastLagrIntp & fastLagr);
		void nonlinear_Lagr_HJ_fast(std::function<double(std::vector<double>, int, int)> func, FastLagrIntp & fastLagr);

		// Initialization based on Lagrange interpolation to Alpert basis
		// this will update ucoe_alpt in the class Element
		void init_from_Lagr_to_alpt(std::function<double(std::vector<double>, int)> func, 
									const OperatorMatrix1D<LagrBasis, AlptBasis> & matrix);

		void init_from_Lagr_to_alpt_fast(std::function<double(std::vector<double>, int)> func,
									     FastLagrInit & fastLagr_init);

		// lagrange interpolate source term, then transform to coefficients of alpert basis and add it to Element::rhs
		void source_from_lagr_to_rhs(std::function<double(std::vector<double>, int)> func, FastLagrInit & fastLagr);

		// this function only works for full grid
		void init_from_Lagr_to_alpt_fast_full(std::function<double(std::vector<double>, int)> func,
									     FastLagrFullInit & fastLagr_init);

		// Initialization based on adaptive Lagrange interpolation
		// this will update ucoe_intp of new added elements in dg solution
		void init_coe_ada_intp_Lag(std::function<double(std::vector<double>, int)> func);

		// compute coefficients of Alpt basis from coefficients of Lagrange basis

		void eval_coe_alpt_from_coe_Lag(const OperatorMatrix1D<LagrBasis, AlptBasis> & matrix);

		// store the function value of Alpt basis value at Lagrange interpolation pt (with respect to degree order)
		std::vector<std::vector<double>> Lag_pt_Alpt_1D;

		// store the first derivative of Alpt basis value at Lagrange interpolation pt (with respect to degree order)
		std::vector<std::vector<double>> Lag_pt_Alpt_1D_d1;	

protected:
	// compute the function value of 1D Alpt basis at Lagrange interpolation point 

	void eval_Lag_pt_at_Alpt_1D();

	// compute the first derivative of 1D Alpt basis at Lagrange interpolation point 

	void eval_Lag_pt_at_Alpt_1D_d1();

	// compute numerical solution and flux function at Lagrange basis interpolation points

	void eval_up_fp_Lag(std::function<double(std::vector<double>, int, int)> func, std::vector< std::vector<bool> > is_intp);

	// compute flux function at Lagrange basis interpolation points

	void eval_fp_Lag(std::function<double(std::vector<double>, int, int)> func, std::vector< std::vector<bool> > is_intp);
	void eval_fp_Lag_full(std::function<double(std::vector<double>, int, int, std::vector<double> x)> func, std::vector< std::vector<bool> > is_intp);
	// compute flux function at Lagrange basis interpolation points
	// for HJ equation
	// just compute case: VEC_NUM = 0, d = 0
	void eval_fp_Lag_HJ(std::function<double(std::vector<double>, int, int)> func);

	// compute numerical solution of nonlinear function in artificial viscosity of burgers' equation at Lagrange basis interpolation points
	// used in LU decompotion version

	// compute numerical solution of nonlinear function in artificial viscosity of burgers' equation at Lagrange basis interpolation points

	void eval_coe_grad_u_Lag(std::function<double(std::vector<double>, int)> coe_func, std::vector< std::vector<bool> > is_intp, int d0);
	
	void eval_coe_u_Lag(std::function<double(std::vector<double>, int)> coe_func, std::vector< std::vector<bool> > is_intp);

	// compute numerical solution of nonlinear function in wave at Lagrange basis interpolation points

	void wave_eval_coe_grad_u_Lag(std::function<double(std::vector<double>, int)> coefficient_func, std::vector< std::vector<bool> > is_intp);

	void wave_eval_coe_u_Lag(std::function<double(std::vector<double>, int)> coefficient_func, std::vector< std::vector<bool> > is_intp);

	// read function value at Lagrange point from exact function directly
	// this function will update Element::up_intp
	void eval_up_exact_Lag(std::function<double(std::vector<double>, int)> func);

	// read function value at Lagrange point from exact function directly for adaptive interpolation initialization
	void eval_up_exact_ada_Lag(std::function<double(std::vector<double>, int)> func);

	// find points and weights which will affact the coefficients of the Lagrange interpolation points
	void set_pts_wts_1d_ada_Lag(int l, int j);

	// u: compute hierarchical coefficients of Lagrange basis based on function value at Lagrange point 
	void eval_up_to_coe_D_Lag();

	// u: compute hierarchical coefficients of Lagrange basis based on function value at Lagrange point
	// for adaptive interpolation
	void eval_up_to_coe_D_ada_Lag();

	// fu: compute hierarchical coefficients of Lagrange basis based on function value at Lagrange point 
	void eval_fp_to_coe_D_Lag(std::vector< std::vector<bool> > is_intp);

	// fu: compute hierarchical coefficients of Lagrange basis based on function value at Lagrange point 
	// for HJ equation
	// just compute case: VEC_NUM = 0, d = 0
	void eval_fp_to_coe_D_Lag_HJ();

	// function to compute Lagrange point value at pos[D] based on Alpt basis
	void eval_point_val_Alpt_Lag(std::vector<double> & pos, std::vector<int> & m1, std::vector<double> & val);
	
	// function to compute first derivative value at pos[D] based on Alpt basis
	// u_x, u_y, ...

	void eval_der1_val_Alpt_Lag(std::vector<double> & pos, std::vector<int> & m1, int d0, double & val);

	// function to compute point value at pos[D] based on Lagrange basis

    double eval_point_val_Lagr(std::vector<double> & pos);

	// compare solution based on Lagrange interpolation

	void compare_solution_Lag();

};

// Lagrange interpolation for artificial viscosity term
class LargViscosInterpolation:
	public LagrInterpolation
{
public:
	LargViscosInterpolation(DGSolution & dgsolution): LagrInterpolation(dgsolution) {};
	~LargViscosInterpolation() {};

	// Lagrange interpolation of const * u_x and const * u_y depending support element		
	// if interpolation points is in support elements, then we interpolation const * u_x and const * u_y;
	// otherwise, zero
	// overlap:
	// 		false (by default): not consider overlap of supports
	// 		true: consider overlap number of supports
	// this will update fucoe_intp in class Element
	// this will be applied in calculate artificial viscosity term
	void support_gradu_Lagr_fast(const std::unordered_set<Element*> & support_element, const double coefficient, std::vector< std::vector<bool> > is_intp, FastLagrIntp & fastLagr, const bool overlap = false);

private:
	void eval_artificial_viscos_grad_u_Lag(const std::unordered_set<Element*> & support_element, const double coefficient, std::vector< std::vector<bool> > is_intp, int d0, const bool overlap);
	
	// return true if a given point in a interval in 1D
	// xl and xr: left and right end point of this interval
	static bool is_pt_in_interval(const double pt, const double xl, const double xr);
	// overload: interval[0] and interval[1]: left and right end point
	static bool is_pt_in_interval(const double pt, const std::vector<double> & interval);
	// overload for multidimension
	static bool is_pt_in_interval(const std::vector<double> & pt, const std::vector<double> & xl, const std::vector<double> & xr);

	// return true if a given point in multidimension in a set of elements in multidimension
	static bool is_pt_in_set_element(const std::vector<double> & pt, const std::unordered_set<Element*> & element_set);	

	// return a point in number of a set of elements
	static int num_pt_in_set_element(const std::vector<double> & pt, const std::unordered_set<Element*> & element_set);
};



////---------------------->for Hermite interpolation

class HermInterpolation:
    public Interpolation
{
	public:
		HermInterpolation(DGSolution & dgsolution): 
		Interpolation(dgsolution)
		{
			eval_Her_pt_at_Alpt_1D();
		};
		~HermInterpolation() {};

		void nonlinear_Herm_1D(std::function<double(std::vector<double>, int, int)> func,
	                          std::function<double(std::vector<double>, int, int, int)> func_d1,
							  std::vector< std::vector<bool> > is_intp);

		// Hermite interpolation for a given function f(u), u is linear combination of Alpert basis
		// this will update VecMultiD<double> fp_intp in class Element
		void nonlinear_Herm_2D(std::function<double(std::vector<double>, int, int)> func,
						std::function<double(std::vector<double>, int, int, int)> func_1st_derivative,
						std::function<double(std::vector<double>, int, int, int, int)>  func_2nd_derivative,
						std::vector< std::vector<bool> > is_intp);

		void nonlinear_Herm_2D_fast(std::function<double(std::vector<double>, int, int)> func,
						std::function<double(std::vector<double>, int, int, int)> func_1st_derivative,
						std::function<double(std::vector<double>, int, int, int, int)>  func_2nd_derivative,
						std::vector< std::vector<bool> > is_intp, FastHermIntp & fastHerm);

		void nonlinear_Herm_2D_PMAX5_scalar_fast(std::function<double(std::vector<double>, int, int)> func,
	                          		std::function<double(std::vector<double>, int, int, int)> func_d1,
	                          		std::function<double(std::vector<double>, int, int, int, int)>  func_d2,
									std::function<double(std::vector<double>, int, int, int, int, int)>  func_d3,
									std::function<double(std::vector<double>, int, int, int, int, int, int)>  func_d4,
									std::vector< std::vector<bool> > is_intp, FastHermIntp & fastHerm);

		// Hermite interpolation for 1D2V Vlasov equation
		void interp_Herm_Vlasov_1D2V(DGSolution & dg_BE, FastHermIntp & fastHerm_f, FastHermIntp & fastHerm_BE);		
		
		// Initialization based on Hermite interpolation to Alpert basis
		// this will update up_alpt in the class Element
		void init_from_Herm_to_alpt(std::function<double(std::vector<double>, int, std::vector<int>)> func,
									const OperatorMatrix1D<HermBasis, AlptBasis> & matrix);

		void init_from_Herm_to_alpt_fast(std::function<double(std::vector<double>, int, std::vector<int>)> func,
										 FastHermInit & fastHerm_init);

		// Initialization based on adaptive Hermite interpolation
		// this will update ucoe_intp of new added elements in dg solution
		void init_coe_ada_intp_Herm(std::function<double(std::vector<double>, int, std::vector<int>)> func);

		// compute coefficients of Alpt basis from coefficients of Hermite basis
		
		void eval_coe_alpt_from_coe_Her(const OperatorMatrix1D<HermBasis, AlptBasis> & matrix);

		// store the Alpt basis value at Hermite interpolation pt (with respect to degree order)
		// function or derivative value
		std::vector<std::vector<double>> Her_pt_Alpt_1D;

	private:
	
	// compute the value of 1D hermite interpolation point with Alpt basis
	
	void eval_Her_pt_at_Alpt_1D();

	// compute numerical solution for u at Hermite basis interpolation points

	void eval_up_Her();	

	// compute numerical solution for f(u) at Hermite basis interpolation points
	// func[i][j][k]: 
	// i: the number of vector
	// j: j-th dimension
	// k-->0: function ------> f(u)
	// k-->1: 1st_order derivative  -----> f'(u)
	// k-->2: 2nd_order derivative  -----> f''(u)

	void eval_fp_Her_2D(std::function<double(std::vector<double>, int, int)> func,
	                    std::function<double(std::vector<double>, int, int, int)> func_d1,
	                    std::function<double(std::vector<double>, int, int, int, int)>  func_d2,
					    std::vector< std::vector<bool> > is_intp);

	void eval_fp_Her_2D_PMAX5_scalar(std::function<double(std::vector<double>, int, int)> func,
	                            std::function<double(std::vector<double>, int, int, int)> func_d1,
	                            std::function<double(std::vector<double>, int, int, int, int)> func_d2,
								std::function<double(std::vector<double>, int, int, int, int, int)> func_d3,
								std::function<double(std::vector<double>, int, int, int, int, int, int)> func_d4,
								std::vector< std::vector<bool> > is_intp);

	void eval_fp_Her_1D(std::function<double(std::vector<double>, int, int)> func,
	                    std::function<double(std::vector<double>, int, int, int)> func_d1,
						std::vector< std::vector<bool> > is_intp);

	// read function value at Hermite point from exact function directly

	void eval_up_exact_Her(std::function<double(std::vector<double>, int, std::vector<int>)> func);

	// read function value at Hermite point from exact function directly for adaptive initialization

	void eval_up_exact_ada_Her(std::function<double(std::vector<double>, int, std::vector<int>)> func);

	// from degree to get the index of point and derivative 
	// void degree_to_point_derivative(std::vector<int> & pp, std::vector<int> & p, std::vector<int> & l);
	
	void deg_pt_deri_1d(const int pp, int & p, int & l);

	// find points and weights which will affact the coefficients of the Hermite interpolation points

	void set_pts_wts_1d_ada_Her(const int l, const int j);

	// u: compute hierarchical coefficients of Hermite basis based on function value at Hermite point
	
	void eval_up_to_coe_D_Her();

	// u: compute hierarchical coefficients of Hermite basis based on function value at Hermite point
	// for adaptive initialization
	
	void eval_up_to_coe_D_ada_Her();

	// fu: compute hierarchical coefficients of Hermite basis based on function value at Hermite point 
	
	void eval_fp_to_coe_D_Her(std::vector< std::vector<bool> > is_intp);

	// function to compute Hermite point value at pos[D] based on Alpt basis

    void eval_point_val_Alpt_Her(std::vector<double> & pos, std::vector<int> & m1, std::vector<double> & val);

	// function to compute point value at pos[D] based on Hermite basis

	double eval_point_val_Her(std::vector<double> & pos, std::vector<int> & p);

	// compare solution based on Hermite interpolation

	void compare_solution_Her();
	
	// plot solution

	void plot_error(std::function<double(double, int)> func);
    
	void eval_fp_Vlasov_1D2V_P3();

	// TO BE COMPLETED
	void eval_fp_Vlasov_1D2V_P5();
};


// namespace for store commonly used flux (linear or nonlinear functions of u)
namespace FluxFunction
{
	// linear flux for scalar function
	double linear_flux_scalar(const double u, int dim, const std::vector<double> & coefficient);

	double linear_flux_1st_derivative_scalar(const double u, const int dim, const std::vector<double> & coefficient);

	// Burgers' flux u*u/2
	double burgers_flux_scalar(const double u);

	double burgers_flux_1st_derivative_scalar(const double u);

	double burgers_flux_2nd_derivative_scalar(const double u);

	// sin(u), KPP problem
	double sin_flux(const double u);

	double sin_flux_1st_derivative(const double u);

	double sin_flux_2nd_derivative(const double u);

	// cos(u), KPP problem
	double cos_flux(const double u);

	double cos_flux_1st_derivative(const double u);

	double cos_flux_2nd_derivative(const double u);

	// u^2/(u^2 + (1 - u)^2), Buckley–Leverett equation
	double buckley_leverett_dim_x(const double u);

	double buckley_leverett_dim_x_1st_derivative(const double u);

	double buckley_leverett_dim_x_2nd_derivative(const double u);

	// (u^2 (1 - 5 (1 - u)^2))/(u^2 + (1 - u)^2), Buckley–Leverett equation
	double buckley_leverett_dim_y(const double u);

	double buckley_leverett_dim_y_1st_derivative(const double u);

	double buckley_leverett_dim_y_2nd_derivative(const double u);
}