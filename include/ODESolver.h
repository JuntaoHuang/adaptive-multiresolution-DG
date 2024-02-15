#pragma once
#include "BilinearForm.h"
#include "HamiltonJacobi.h"
#include "Timer.h"
class ODESolver
{
public:
    // constructor for the case when bilinearform is given
    ODESolver(BilinearForm & bilinearform);
    
    // constructor for the case when only dof is given and bilinearform is not given
    ODESolver(DGSolution & dg);
    
    ~ODESolver() {};

    // transfer eigenvector ODESolver::ucoe to Element::ucoe_alpt and set eigenvector ODESolver::rhs to be zero
    virtual void final() { eigenvec_to_ucoe(); rhs.setZero(); };
	template<class T>
	void final_HJ(HamiltonJacobiLDG<T> & HJ) { HJ.copy_eigenvec_to_phi(ucoe); rhs.setZero(); };
	//virtual void final_HJ(HamiltonJacobiLDG<HJOutflowAlpt> & HJ) { HJ.copy_eigenvec_to_phi(ucoe); rhs.setZero(); };
	//virtual void final_HJ(HamiltonJacobiLDG<HyperbolicAlpt> & HJ) { HJ.copy_eigenvec_to_phi(ucoe); rhs.setZero(); };

    // add Element::rhs to eigenvec ODESolver::rhs
    void add_rhs_to_eigenvec();

    // add eigenvec ODESolver::rhs to Element::rhs
    void add_eigenvec_to_rhs() const;

    // multiply ODESolver::ucoe by BilinearForm matrix and add it to ODESolver::rhs
    void add_rhs_matrix(const BilinearForm & bilinearform) { rhs += bilinearform.mat * ucoe; };

	void add_rhs_vector(const Eigen::VectorXd & vec) { rhs += vec; };

    // multiply ODESolver::fucoe (coefficient of interpolation basis) by BilinearForm matrix and add it to ODESolver::rhs
    void add_rhs_matrix_intp(const BilinearForm & bilinearform) { rhs += bilinearform.mat * fucoe; }; 

    // set eigenvector ODESolver::rhs to be zero
    void set_rhs_zero() { rhs.setZero(); };

    // transfer Element::fucoe_intp (in dim-th dimension component) to eigenvector ODESolver::fucoe
    void fucoe_to_eigenvec(const int dim);
	// overload the function for HJ equations
	void fucoe_to_eigenvec_HJ(); 

    // transfer between eigen vector ODESolver::ucoe and Element::ucoe_alpt in class DGSolution
    void ucoe_to_eigenvec();
    void eigenvec_to_ucoe() const;

    // transfer between eigen vector ODESolver::rhs and Element::rhs in class DGSolution    
	void rhs_to_eigenvec();
	// overload for HJ equation
	void rhs_to_eigenvec(std::string eqn);
    void eigenvec_to_rhs() const;

    BilinearForm* const dgoperator_ptr;

	DGSolution* const dgsolution_ptr;
    // vector of size the same with num of alpert basis
    Eigen::VectorXd ucoe;
    Eigen::VectorXd rhs;

    // vector of size the same with num of interpolation basis
    Eigen::VectorXd fucoe;
    
	/// dof is given by the function get_dof() in class DGSolution, determined by size of ind_var_vec
    const int dof;
};



// ----------------------------------------------
// 
// below is explicit rk method
// 
// ----------------------------------------------
class ExplicitRK:
    public ODESolver
{
public:
    ExplicitRK(BilinearForm & BilinearForm, const double dt_): ODESolver(BilinearForm), dt(dt_)
        {
            ucoe_tn.resize(dof);
            ucoe_tn.setZero();
        };
    ExplicitRK(DGSolution & dg, const double dt_): ODESolver(dg), dt(dt_)
        {
            ucoe_tn.resize(dof);
            ucoe_tn.setZero();
        };
    ~ExplicitRK() {};

    // transfer Element::ucoe_alpt to eigenvector ODESolver::ucoe and then copy it to ODESolver::ucoe_tn
    virtual void init() { ucoe_to_eigenvec(); ucoe_tn = ucoe; };

	template<class T>
    void init_HJ(const HamiltonJacobiLDG<T> & HJ) { ucoe = HJ.get_phi_eigenvec(); ucoe_tn = ucoe; };
	template<class T>
	void compute_gradient_HJ(HamiltonJacobiLDG<T> & HJ) { HJ.HJ_apply_operator_to_vector(ucoe); };
	// only works for DG operator keep unchanged
	virtual void step_rk() = 0;

	// rhs should be updated before using this function
	virtual void step_stage(const int stage) = 0;

// protected:

	// transfer from Element::ucoe_alpt to ODESolver::ucoe_tn
	void ucoe_tn_to_eigenvec();

	Eigen::VectorXd ucoe_tn;

	const double dt;
};

class ForwardEuler :
	public ExplicitRK
{
public:
	ForwardEuler(BilinearForm & BilinearForm, const double dt) : ExplicitRK(BilinearForm, dt), num_stage(1) {};
	ForwardEuler(DGSolution & dg, const double dt) : ExplicitRK(dg, dt), num_stage(1) {};
	~ForwardEuler() {};

	// only works for DG operator keep unchanged
	virtual void step_rk() override;

	// rhs should be updated before using this function
	virtual void step_stage(const int stage) override;

	const int num_stage;    // number of stages: 1
};


class RK2SSP :
	public ExplicitRK
{
public:
	RK2SSP(BilinearForm & BilinearForm, const double dt) : ExplicitRK(BilinearForm, dt), num_stage(2) { u1.resize(dof); };
	RK2SSP(DGSolution & dg, const double dt) : ExplicitRK(dg, dt), num_stage(2) { u1.resize(dof); };
	~RK2SSP() {};

	// only works for DG operator keep unchanged
	virtual void step_rk() override;

	// rhs should be updated before using this function
	virtual void step_stage(const int stage) override;

	const int num_stage;    // number of stages: 2

private:
	Eigen::VectorXd u1;
};

// Midpoint method
class RK2Midpoint:
	public ExplicitRK
{
public:
	RK2Midpoint(BilinearForm & BilinearForm, const double dt) : ExplicitRK(BilinearForm, dt), num_stage(2) { u1.resize(dof); };
	RK2Midpoint(DGSolution & dg, const double dt) : ExplicitRK(dg, dt), num_stage(2) { u1.resize(dof); };
	~RK2Midpoint() {};

	// only works for DG operator keep unchanged
	virtual void step_rk() override;

	// rhs should be updated before using this function
	virtual void step_stage(const int stage) override;

	const int num_stage;    // number of stages: 2

private:
	Eigen::VectorXd u1;
};


class RK3SSP :
	public ExplicitRK
{
public:
	RK3SSP(BilinearForm & BilinearForm, const double dt) : ExplicitRK(BilinearForm, dt), num_stage(3) { u1.resize(dof); u2.resize(dof); };
	RK3SSP(DGSolution & dg, const double dt) : ExplicitRK(dg, dt), num_stage(3) { u1.resize(dof); u2.resize(dof); };
	~RK3SSP() {};

	// only works for DG operator keep unchanged
	virtual void step_rk() override;

	// only works for DG operator keep unchanged, with source term (also keep unchange with time)
	void step_rk_source(const Eigen::VectorXd & source);

	// rhs should be updated before using this function
	virtual void step_stage(const int stage) override;

	const int num_stage;    // number of stages: 3

private:
	Eigen::VectorXd u1;
	Eigen::VectorXd u2;
};

// Heun's method for linear problem
class RK3HeunLinear:
	public ExplicitRK
{
public:
	RK3HeunLinear(BilinearForm & BilinearForm, const double dt) : ExplicitRK(BilinearForm, dt), num_stage(3) { u1.resize(dof); u2.resize(dof); };
	RK3HeunLinear(DGSolution & dg, const double dt) : ExplicitRK(dg, dt), num_stage(3) { u1.resize(dof); u2.resize(dof); };
	~RK3HeunLinear() {};

	// only works for DG operator keep unchanged
	virtual void step_rk() override;

	// only works for DG operator keep unchanged, with source term (also keep unchange with time)
	void step_rk_source(const Eigen::VectorXd & source);

	// rhs should be updated before using this function
	virtual void step_stage(const int stage) override;

	const int num_stage;    // number of stages: 3

private:
	Eigen::VectorXd u1;
	Eigen::VectorXd u2;
};

class RK4 :
	public ExplicitRK
{
public:
	RK4(BilinearForm & BilinearForm, const double dt) : ExplicitRK(BilinearForm, dt), num_stage(4) { u1.resize(dof); u2.resize(dof); u3.resize(dof); };
	RK4(DGSolution & dg, const double dt) : ExplicitRK(dg, dt), num_stage(4) { u1.resize(dof); u2.resize(dof); u3.resize(dof); };
	~RK4() {};

	virtual void step_rk() override;

	const int num_stage;    // number of stages: 4

private:
	Eigen::VectorXd u1;
	Eigen::VectorXd u2;
	Eigen::VectorXd u3;
};


/**
 * @brief 	Solve second order ODE u_tt = L(u) by rewrite it into systems
 * 			u_t = v
 * 			v_t = L(u)
 * 			and then apply explicit RK
 */
class RKODE2nd:
	public ExplicitRK
{
public:
	RKODE2nd(BilinearForm & BilinearForm, const double dt): ExplicitRK(BilinearForm, dt) 
		{
			vcoe_tn.resize(dof);
			vcoe_tn.setZero();
			vcoe.resize(dof);
			vcoe.setZero();
		};
	~RKODE2nd() {};

	virtual void init() override;	

	virtual void final() override;

protected:
	// transfer from Element::ucoe_ut to ODESolver::vcoe
	void ucoe_ut_to_eigenvec();

	// transfer from ODESolver::vcoe to Element::ucoe_ut
	void eigenvec_to_ucoe_ut();

	Eigen::VectorXd vcoe;		// store coefficient of v = u_t
	Eigen::VectorXd vcoe_tn;	// store coefficient of v = u_t at t = t^n
};


class EulerODE2nd:
	public RKODE2nd
{
public:
	EulerODE2nd(BilinearForm & BilinearForm, const double dt): RKODE2nd(BilinearForm, dt), num_stage(1) {};
	~EulerODE2nd() {};

	virtual void step_rk() override;

	void step_rk_source(std::function<Eigen::VectorXd(double)> source_func, double time_tn);
	
	// rhs should be updated before using this function
	virtual void step_stage(const int stage) override;	

	const int num_stage;    // number of stages: 1
};

class RK2ODE2nd:
	public RKODE2nd
{
public:
	RK2ODE2nd(BilinearForm & BilinearForm, const double dt): RKODE2nd(BilinearForm, dt), num_stage(2) {};
	~RK2ODE2nd() {};

	virtual void step_rk() override;

	void step_rk_source(std::function<Eigen::VectorXd(double)> source_func, double time_tn);

	// rhs should be updated before using this function
	virtual void step_stage(const int stage) override;

	void step_stage_source(const int stage, std::function<Eigen::VectorXd(double)> source_func, double time_tn);

	const int num_stage;    // number of stages: 2
};

class RK3ODE2nd:
	public RKODE2nd
{
public:
	RK3ODE2nd(BilinearForm & BilinearForm, const double dt): RKODE2nd(BilinearForm, dt), num_stage(3) {};
	~RK3ODE2nd() {};

	virtual void step_rk() override;

	void step_rk_source(std::function<Eigen::VectorXd(double)> source_func, double time_tn);

	// rhs should be updated before using this function
	virtual void step_stage(const int stage) override;

	void step_stage_source(const int stage, std::function<Eigen::VectorXd(double)> source_func, double time_tn);

	const int num_stage;    // number of stages: 3
};

class RK4ODE2nd:
	public RKODE2nd
{
public:
	RK4ODE2nd(BilinearForm & BilinearForm, const double dt): RKODE2nd(BilinearForm, dt), num_stage(4) { resize_ku_kv(); };
	~RK4ODE2nd() {};

	virtual void step_rk() override;

	void step_rk_source(std::function<Eigen::VectorXd(double)> source_func, double time_tn);
	
	// rhs should be updated before using this function
	virtual void step_stage(const int stage) override;

	void step_stage_source(const int stage, std::function<Eigen::VectorXd(double)> source_func, double time_tn);

	const int num_stage;    // number of stages: 4

private:
	void resize_ku_kv();

	Eigen::VectorXd k1u;
	Eigen::VectorXd k2u;
	Eigen::VectorXd k3u;
	Eigen::VectorXd k4u;
	
	Eigen::VectorXd k1v;
	Eigen::VectorXd k2v;
	Eigen::VectorXd k3v;
	Eigen::VectorXd k4v;
};

// ----------------------------------------------
// 
// below is explicit multistep method
// 
// ----------------------------------------------
class ExplicitMultiStep :
	public ODESolver
{
public:
	ExplicitMultiStep(BilinearForm & BilinearForm, const double dt_) : ODESolver(BilinearForm), dt(dt_) { resize_variable(); };
	ExplicitMultiStep(DGSolution & dg, const double dt_) : ODESolver(dg), dt(dt_) { resize_variable(); };
	~ExplicitMultiStep() {};

	// unhide base class method because of overload
	using ODESolver::ucoe_to_eigenvec;
	using ODESolver::eigenvec_to_ucoe;

	// transfer between eigen vector ODESolver::ucoe_t_m1 and Element::ucoe_alpt_t_m1 in class DGSolution
	void ucoe_to_eigenvec_t_m1();
	void eigenvec_to_ucoe_t_m1() const;

	// coefficient in the (n-1)-th time step
	Eigen::VectorXd ucoe_t_m1;

protected:
	const double dt;

private:
	void resize_variable();
};

// solve u_tt = A * u + b
// 
// second order time stepping:
// 
// u^(n+1) = 2*u^n - u^(n-1) + dt^2 * ( A * u^n + b^n )
class Newmard2nd :
	public ExplicitMultiStep
{
public:
	Newmard2nd(BilinearForm & BilinearForm, const double dt_) : ExplicitMultiStep(BilinearForm, dt_) {};
	Newmard2nd(DGSolution & dg, const double dt_) : ExplicitMultiStep(dg, dt_) {};
	~Newmard2nd() {};

	// b = 0, update by using matrix multiply by vector
	void step_ms();

	// b != 0, update by using matrix multiply by vector
	void step_ms(const Eigen::VectorXd & vec_b);

	// b = 0, update by using rhs directly
	// rhs should be updated before using this function
	void step_ms_rhs();

	// b != 0, update by using rhs directly
	// rhs should be updated before using this function
	void step_ms_rhs(const Eigen::VectorXd & vec_b);
};

// solve u_tt = A * u + b
// 
// fourth order time stepping:
// (u^(n+1) - 2 * u^n + u^(n-1))/(dt^2) = (A * u^n + b^n) + dt^2/12*(A^2 * u^n + A * b^n + (b'')^n)
// 
// it can also be reformulated as prediction-correction form:
// prediction: 
//      (u^(predict) - 2 * u^n + u^(n-1))/(dt^2) = A * u^n + b^n
// correction:
//      u^(n+1) = u^(predict) + dt^4/12 * (A * v + (b'')^n)
//      with v = (u^(predict) - 2 * u^n + u^(n-1))/(dt^2)
class Newmard4th :
	public ExplicitMultiStep
{
public:
	Newmard4th(BilinearForm & BilinearForm, const double dt_) : ExplicitMultiStep(BilinearForm, dt_) {};
	~Newmard4th() {};

	// b = 0, update by using matrix multiply by vector
	void step_ms();

	// b != 0, update by using matrix multiply by vector
	void step_ms(const Eigen::VectorXd & vec_b, const Eigen::VectorXd & vec_b_2nd_derivative);

	// b = 0, update by using rhs directly
	// rhs and rhs_2nd should be updated before using this function
	void step_ms_rhs();

	// b != 0, update by using rhs directly
	// rhs and rhs_2nd should be updated before using this function
	void step_ms_rhs(const Eigen::VectorXd & vec_b, const Eigen::VectorXd & vec_b_2nd_derivative);

	// this vector store A * u^n, i.e., apply linear operator to u^n
	Eigen::VectorXd rhs_Au;

	// this vector store A^2 * u^n, i.e., apply linear operator twice to u^n
	Eigen::VectorXd rhs_A2u;

	// this vector store A * b^n, i.e., apply linear operator to source term b
	Eigen::VectorXd rhs_Ab;

	// a vector temporary store some variable
	Eigen::VectorXd vec_tmp;

	// swap two eigen vector
	void swap_vec(Eigen::VectorXd & vec_a, Eigen::VectorXd & vec_b);
};

// ----------------------------------------------
// 
// below is implicit-explicit Runge-Kutta (IMEX) method
// 
// ----------------------------------------------
class IMEX :
	public ExplicitRK
{
public:
	// explicit and implicit part are both provided by matrix
	IMEX(BilinearForm & bilinearForm_explicit, BilinearForm & bilinearForm_implicit, const double dt) :
		ExplicitRK(bilinearForm_explicit, dt),
		dgoperator_ptr_explicit(&bilinearForm_explicit),
		dgoperator_ptr_implicit(&bilinearForm_implicit),
		_dt(dt) {};

	// only provide implicit part matrix
	IMEX(BilinearForm & bilinearForm_implicit, const double dt) :
		ExplicitRK(bilinearForm_implicit, dt),
		dgoperator_ptr_explicit(nullptr),
		dgoperator_ptr_implicit(&bilinearForm_implicit),
		_dt(dt) {};

	~IMEX() {};

	// transfer between eigen vector rhs_explicit and Element::rhs in class DGSolution
	void rhs_explicit_to_eigenvec();
	void eigenvec_to_rhs_explicit() const;

protected:

	BilinearForm* const dgoperator_ptr_explicit;
	BilinearForm* const dgoperator_ptr_implicit;

	Eigen::VectorXd rhs_explicit;     // explicit part directly given as rhs

	const double _dt;
};

/**
 * @brief 	use first order implicit-explict Euler backward-forward to solve u_t = F(u) + G(u)
 * 			u^(n+1) = u^n + dt * F(u^n) + dt * G(u^(n+1))
 */
class IMEXEuler:
	public IMEX
{
public:
	IMEXEuler(BilinearForm & bilinearForm_explicit, BilinearForm & bilinearForm_implicit, const double dt, const std::string linear_solver = "sparselu");
	IMEXEuler(BilinearForm & bilinearForm_implicit, const double dt, const std::string linear_solver = "sparselu");
	~IMEXEuler() {};

	// only works for DG operator keep unchanged
	virtual void step_rk() override { std::cout << "IMEXEuler::step_rk() to be completed" << std::endl; };
	
	// rhs should be updated before using this function
	virtual void step_stage(const int stage) override;

	const int num_stage;    // number of stages: 1

private:
	Eigen::SparseMatrix<double> implicit_linear_matx;

	const std::string linear_solver_type;

	void init_linear_solver();
	void init_sparseLU_solver();

	Eigen::SparseLU<Eigen::SparseMatrix<double>> sparselu;	// LU factorization direct solver
};


/**
 * @brief 	use implicit-explicit Runge-Kutta to solve u_t = F(u) + G(u)
 * 			where F is explicit part and G is implicit part
 * 
 * @note
 * 	Pareschi and Russo, Implicit–Explicit Runge–Kutta Schemes and Applications to Hyperbolic Systems with Relaxation, 2005, JSC
 * 	Table VI, SSP3(4,3,3)scheme 
 * 	implicit 4 stages, explicit 3 stages, order 3.
 * 
 * 	stage 0
 * 		u1 = un + dt * alpha * G(u1)
 * 	stage 1
 * 		u2 = un + dt * ( -alpha * G(u1) + alpha * G(u2) )
 * 	stage 2
 * 		u3 = un + dt * F(u2) + dt * ( (1-alpha) * G(u2) + alpha * G(u3) )
 * 	stage 3
 * 		u4 = un + dt * ( 1/4 * F(u2) + 1/4 * F(u3) )+ dt * ( beta * G(u1) + eta * G(u2) + (1/2-beta-eta-alpha) * G(u3) + alpha * G(u4) )
 * 	stage 4 (final stage)
 * 		u^(n+1) = un + dt * ( 1/6 * F(u2) + 1/6 * F(u3) + 2/3 * F(u4) )+ dt * ( 1/6 * G(u2) + 1/6 * G(u3) + 2/3 * G(u4) )
 */
class IMEX43 :
	public IMEX
{
public:
	/**
	 * @brief Construct a new IMEX43 object with given both explicit and implicit linear part
	 * 
	 * @param bilinearForm_explicit 
	 * @param bilinearForm_implicit 
	 * @param dt 
	 * @param linear_solver 
	 */
	IMEX43(BilinearForm & bilinearForm_explicit, BilinearForm & bilinearForm_implicit, const double dt, const std::string linear_solver = "cg");
	/**
	 * @brief Construct a new IMEX43 object with given only implicit linear part
	 * 
	 * @param bilinearForm_implicit 
	 * @param dt 
	 * @param linear_solver 
	 */
	IMEX43(BilinearForm & bilinearForm_implicit, const double dt, const std::string linear_solver = "cg");
	~IMEX43() {};

	// only works for DG operator keep unchanged
	virtual void step_rk() override;

	// rhs should be updated before using this function
	virtual void step_stage(const int stage) override;

	const int num_stage;    // number of stages: 5

private:
	Eigen::SparseMatrix<double> implicit_linear_matx;

	Eigen::VectorXd u1;
	Eigen::VectorXd u2;
	Eigen::VectorXd u3;
	Eigen::VectorXd u4;

	Eigen::VectorXd Gu1;
	Eigen::VectorXd Gu2;
	Eigen::VectorXd Gu3;
	Eigen::VectorXd Gu4;

	Eigen::VectorXd Fu2;
	Eigen::VectorXd Fu3;
	Eigen::VectorXd Fu4;

	const std::string linear_solver_type;

	void resize_variable();
	
	void init_linear_solver();
	void init_cg_solver();
	void init_bicg_solver();
	void init_sparseLU_solver();

	Eigen::ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Lower | Eigen::Upper> cg;  // conjugate gradient solver
	Eigen::BiCGSTAB<Eigen::SparseMatrix<double>, Eigen::IncompleteLUT<double> > bicg;  // bi conjugate gradient stabilized solver
	Eigen::SparseLU<Eigen::SparseMatrix<double>> sparselu;	// LU factorization direct solver
	
	static const double alpha;
	static const double beta;
	static const double eta;
};