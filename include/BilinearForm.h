#pragma once
#include "DGSolution.h"

class BilinearForm
{
public:
    BilinearForm(DGSolution & dgsolution):
        dgsolution_ptr(&dgsolution), row(0), col(0), mat() {};
    ~BilinearForm() {};
    
    DGSolution* const dgsolution_ptr;
     
    // size of matrix, row and col
    int row;
    int col;

    // add another BilinerForm (matrix) to this one
    void add(const BilinearForm & bilinearform);

    Eigen::SparseMatrix<double> mat;

    // rhs = matrix * ucoe
    void multi(const Eigen::VectorXd & ucoe, Eigen::VectorXd & rhs) const { rhs = mat * ucoe; };
	
};


class BilinearFormAlpt:
    public BilinearForm
{
public:
    BilinearFormAlpt(DGSolution & dgsolution, OperatorMatrix1D<AlptBasis, AlptBasis> & oper_matx_alpt):
        BilinearForm(dgsolution), oper_matx_alpt_ptr(&oper_matx_alpt) { resize_zero_matrix(dgsolution.VEC_NUM); };

	BilinearFormAlpt(DGSolution & dgsolution, OperatorMatrix1D<AlptBasis, AlptBasis> & oper_matx_alpt, int const n) :
		BilinearForm(dgsolution), oper_matx_alpt_ptr(&oper_matx_alpt) {
		resize_zero_matrix(n);
	};

    ~BilinearFormAlpt() {};
        
    /**
     * @brief resize the matrix (when adaptive) and set to zero
     */
    void resize_zero_matrix(const int n = 1);

protected:
    OperatorMatrix1D<AlptBasis, AlptBasis>* const oper_matx_alpt_ptr;
    
    // calculate volume and flux integral of Alpert basis with given dimension
	// only work for PDE operators involving derivative in one dimension, e.g., \iint u * v_x, \iint u_x * v_x, not for \iint u_x * v_y
    void assemble_matrix_alpt(const double operatorCoefficient, const int dim, const VecMultiD<double> & mat_operator, const VecMultiD<double> & mat_mass, const std::string integral_type, const int index_solu_variable = 0, const int index_test_variable = 0);
	// overload, for PDE operators involving derivative in multiple dimension, e.g., \iint u * v_xyy
	// this will be applied in class ZKAlpt
	void assemble_matrix_alpt(const double operatorCoefficient, const int dim, const std::vector<const VecMultiD<double>*> & mat_1D_array, const std::string & integral_type, const int index_solu_variable = 0, const int index_test_variable = 0);
	
	// overload, for PDE operators involving multiple terms involving interface
	// this will be applied in jump terms in ZK equation
	void assemble_matrix_alpt(const double operatorCoefficient, const std::vector<const VecMultiD<double>*> & mat_1D_array, const int index_solu_variable = 0, const int index_test_variable = 0);

	// overload, it can either take matrix coefficient or vector coefficient (diagonal)
	// GENERATE CORRECT RESULTS, BUT NOT OPTIMIZED
    void assemble_matrix_system_alpt(const std::vector<std::vector<double>> & operatorCoefficient, const int dim, const VecMultiD<double> & mat_operator, const VecMultiD<double> & mat_mass, const std::string integral_type, const double coef = 1.);
    void assemble_matrix_system_alpt(const std::vector<double> & operatorCoefficient, const int dim, const VecMultiD<double> & mat_operator, const VecMultiD<double> & mat_mass, const std::string integral_type, const double coef = 1.);
};


// ultra weak DG for Kdv eqn
class KdvAlpt:
    public BilinearFormAlpt
{
public:
    KdvAlpt(DGSolution & dgsolution, OperatorMatrix1D<AlptBasis, AlptBasis> & oper_matx_alpt):
        BilinearFormAlpt(dgsolution, oper_matx_alpt) {};
    ~KdvAlpt() {};

    // this function assemble matrix for linear KdV equations
    // coefficient is a constant
    void assemble_matrix_scalar(const std::vector<double> & eqnCoefficient);
	// overload
	void assemble_matrix_scalar(const double eqnCoefficient = 1.0);
};

/**
 * @brief ultraweak DG for ZK equation in 2D: u_t + u_xxx + u_xyy = 0
 * 
 */
class ZKAlpt:
	public BilinearFormAlpt
{
public:
	ZKAlpt(DGSolution & dgsolution, OperatorMatrix1D<AlptBasis, AlptBasis> & oper_matx_alpt):
        BilinearFormAlpt(dgsolution, oper_matx_alpt) {};
	~ZKAlpt() {};

	// assemble matrix for operator in ZK equation
	// 
	// option 0 (default):
	// 		u_xyy weak formulation by Liu Yong, (k+1) convergence order
	// 
	// option 1:
	// 		u_xyy weak formulation with penalty, (k+1) convergence order
	// 
	// option 2:
	// 		u_xyy weak formulation without penalty, only k convergence order
	void assemble_matrix_scalar(const std::vector<double> & eqnCoefficient, const int option = 0);
};

/**
 * @brief ultraweak DG for Schrodinger equation in
 * 		1D
 * 			i * u_t + u_xx = 0
 * 		and 2D
 * 			i * u_t + (u_xx + u_yy) = 0
 */
class SchrodingerAlpt:
	public BilinearFormAlpt
{
public:
	SchrodingerAlpt(DGSolution & dgsolution, OperatorMatrix1D<AlptBasis, AlptBasis> & oper_matx_alpt):
        BilinearFormAlpt(dgsolution, oper_matx_alpt) {};
	~SchrodingerAlpt() {};

	void assemble_matrix(const double eqnCoefficient = 1.0);

	void assemble_matrix_couple(const double eqnCoefficient = 1.0);
};

class HyperbolicAlpt:
    public BilinearFormAlpt
{
public:
    // coefficientOperator: coefficient for scalar hyperbolic equation
    HyperbolicAlpt(DGSolution & dgsolution, OperatorMatrix1D<AlptBasis, AlptBasis> & oper_matx_alpt):
        BilinearFormAlpt(dgsolution, oper_matx_alpt) {};
	HyperbolicAlpt(DGSolution & dgsolution, OperatorMatrix1D<AlptBasis, AlptBasis> & oper_matx_alpt, const int n) :
		BilinearFormAlpt(dgsolution, oper_matx_alpt, n) {};
    ~HyperbolicAlpt() {};

    // this function assemble matrix for linear scalar hyperbolic equations
    // coefficient is a 1D vector, size of dim, which stores coefficients of equation
    void assemble_matrix_scalar(const std::vector<double> & eqnCoefficient);

    // flux integral of u^- * [v] or u^+ * [v]
    // sign: -1, left limit; 1, right limit
    void assemble_matrix_flx_scalar(const int dim, const int sign, const double coefficient = 1.);

	// flux integral of u^- * [v] or u^+ * [v]
	// sign: -1, left limit; 1, right limit
	// coefficient： matrix of size VEC_NUM * VEC_NUM, [i][j] means the coefficient of j-th unknown in i-th equation
	void assemble_matrix_flx_system(const int dim, const int sign, const std::vector<std::vector<double>> & coefficient, const double coef = 1.);
	void assemble_matrix_flx_system(const int dim, const int sign, const std::vector<double> & coefficient, const double coef = 1.);
	// flux integral of [u] * [v]
	void assemble_matrix_flx_jump_system(const int dim, const std::vector<double> & coefficient);


	// volume integral of c * u * v_x
	// coefficient denote c
	// dim denote derivative direction
	void assemble_matrix_vol_scalar(const int dim, const double coefficient = 1.);
	void assemble_matrix_vol_system(const int dim, const std::vector<std::vector<double>> & coefficient);

	// coefficient is a 3D vector
	// coefficient[i][j][k] denotes coefficient of i-th dim, j-th solution variable, k-th test variable
	void assemble_matrix(const VecMultiD<double> & volCoefficient, const VecMultiD<double> & fluxLeftCoefficient, const VecMultiD<double> & fluxRightCoefficient);

	// this function assemble matrix for ux terms in schrodinger system
    // coefficient is a double, which stores coefficient of ux in the equation
	void assemble_matrix_schrodinger(const double eqnCoefficient = 1.0);
};

class HJOutflowAlpt:
	public HyperbolicAlpt
{
public:
	HJOutflowAlpt(DGSolution & dgsolution, OperatorMatrix1D<AlptBasis, AlptBasis> & oper_matx_alpt_period, OperatorMatrix1D<AlptBasis, AlptBasis> & oper_matx_alpt_inside, int const n = 1);
	~HJOutflowAlpt() {};

	// flux near the boundary is taken to be u^+ at x=0 and u^- at x=1
	// if sign = -1, flux inside the domain is taken to be u^-
	// if sign = +1, flux inside the domain is taken to be u^+
	void assemble_matrix_flx_scalar(const int dim, const int sign, const double coefficient = 1.0);

private:
	OperatorMatrix1D<AlptBasis, AlptBasis>* const oper_matx_alpt_inside_ptr;

	VecMultiD<double> mat1D_flx_lft;
	VecMultiD<double> mat1D_flx_rgt;
};

// bilinear form for diffusion operator using interior penalty DG method
class DiffusionAlpt :
	public BilinearFormAlpt
{
public:
	DiffusionAlpt(DGSolution & dgsolution, OperatorMatrix1D<AlptBasis, AlptBasis> & oper_matx_alpt, const double sigma_ipdg_) :
		BilinearFormAlpt(dgsolution, oper_matx_alpt), sigma_ipdg(sigma_ipdg_) {};
	~DiffusionAlpt() {};

	// this function assemble matrix for linear scalar diffusion equations with constant coefficients
	// periodic boundary condition is imposed, if oper_matx_alpt is taken period
	// zero Neumann boundary condition is imposed, if oper_matx_alpt is taken inside
	void assemble_matrix_scalar(const std::vector<double> & eqnCoefficient);

	// volume integral of - k * grad u * grad v
	void assemble_matrix_vol_gradu_gradv(const std::vector<double> & eqnCoefficient);

	// flux integral of - k * {grad u} * [v]
	void assemble_matrix_flx_gradu_vjp(const std::vector<double> & eqnCoefficient);

	// flux integral of - k * [u] * {grad v}
	void assemble_matrix_flx_ujp_gradv(const std::vector<double> & eqnCoefficient);

	// flux integral of - sigma/dx * [u] * [v]
	void assemble_matrix_flx_ujp_vjp();

private:
	const double sigma_ipdg;
};


// bilinear form for diffusion operator with zero Dirichlet boundary condition
class DiffusionZeroDirichletAlpt :
	public BilinearFormAlpt
{
public:
    DiffusionZeroDirichletAlpt(DGSolution & dgsolution, OperatorMatrix1D<AlptBasis, AlptBasis> & oper_matx_alpt, const double sigma_ipdg_, const double eqnCoefficient_);
    ~DiffusionZeroDirichletAlpt() {};

    // this function assemble matrix for linear scalar diffusion equations with constant coefficients
    void assemble_matrix_scalar();

private:
    const double sigma_ipdg;
    const double eqnCoefficient;
    VecMultiD<double> mat1D_flx;
};



class BilinearFormIntp :
	public BilinearForm
{
public:
	BilinearFormIntp(DGSolution & dgsolution, OperatorMatrix1D<LagrBasis, AlptBasis> & oper_matx_lagr, OperatorMatrix1D<HermBasis, AlptBasis> & oper_matx_herm) :
		BilinearForm(dgsolution), oper_matx_lagr_ptr(&oper_matx_lagr), oper_matx_herm_ptr(&oper_matx_herm)
	{
		row = dgsolution_ptr->size_basis_alpt() * dgsolution_ptr->VEC_NUM;
		col = dgsolution_ptr->size_basis_intp() * dgsolution_ptr->VEC_NUM;
		mat.resize(row, col);
		mat.setZero();
	}
	BilinearFormIntp(DGSolution & dgsolution, OperatorMatrix1D<LagrBasis, AlptBasis> & oper_matx_lagr, OperatorMatrix1D<HermBasis, AlptBasis> & oper_matx_herm, const int n) :
		BilinearForm(dgsolution), oper_matx_lagr_ptr(&oper_matx_lagr), oper_matx_herm_ptr(&oper_matx_herm)
	{
		row = dgsolution_ptr->size_basis_alpt() * n;
		col = dgsolution_ptr->size_basis_intp() * n;
		mat.resize(row, col);
		mat.setZero();
	}
	~BilinearFormIntp() {};

	// protected:
	OperatorMatrix1D<LagrBasis, AlptBasis>* const oper_matx_lagr_ptr;
	OperatorMatrix1D<HermBasis, AlptBasis>* const oper_matx_herm_ptr;

	// calculate volume and flux integral of interpolation basis with given dimension
	void assemble_matrix_intp(const double operatorCoefficient, const int dim, const VecMultiD<double> & mat_operator, const VecMultiD<double> & mat_mass, const std::string integral_type, const int index_solu_variable = 0, const int index_test_variable = 0);
	// the overload function only considers the mass part. This is useful for the HJ equations
	void assemble_matrix_intp(const double operatorCoefficient, const VecMultiD<double> & mat_mass, const int index_solu_variable = 0, const int index_test_variable = 0);
};

// nonlinear hyperbolic equation with hermite interpolation
class HyperbolicHerm :
	public BilinearFormIntp
{
public:
	HyperbolicHerm(DGSolution & dgsolution, OperatorMatrix1D<LagrBasis, AlptBasis> & oper_matx_lagr, OperatorMatrix1D<HermBasis, AlptBasis> & oper_matx_herm) :
		BilinearFormIntp(dgsolution, oper_matx_lagr, oper_matx_herm) {};
	~HyperbolicHerm() {};

	// assemble matrix of nonlinear terms in weak formulation
	// volume integral of f(u) * v_x
	// + flux integral of 1/2 * f(u^-) * [v]
	// + flux integral of 1/2 * f(u^+) * [v]
	void assemble_matrix_scalar(const int dim);

	// volume integral of c * f(u) * v_x
	// coefficient denote c
	// dim denote derivative direction
	void assemble_matrix_vol_scalar(const int dim, const double coefficient = 1.);

	// flux integral of f(u^-) * [v] or f(u^+) * [v]
	// sign: -1, left limit; 1, right limit
	void assemble_matrix_flx_scalar(const int dim, const int sign, const double coefficient = 1.);
};


// nonlinear Hamilton-Jacobi equations with Lagrange interpolation
class HamiltonJacobiLagr :
	public BilinearFormIntp
{
public:
	HamiltonJacobiLagr(DGSolution & dgsolution, OperatorMatrix1D<LagrBasis, AlptBasis> & oper_matx_lagr, OperatorMatrix1D<HermBasis, AlptBasis> & oper_matx_herm) :
		BilinearFormIntp(dgsolution, oper_matx_lagr, oper_matx_herm) {};
	HamiltonJacobiLagr(DGSolution & dgsolution, OperatorMatrix1D<LagrBasis, AlptBasis> & oper_matx_lagr, OperatorMatrix1D<HermBasis, AlptBasis> & oper_matx_herm, int const n) :
		BilinearFormIntp(dgsolution, oper_matx_lagr, oper_matx_herm, n) {};
	~HamiltonJacobiLagr() {};


	// volume integral of c * H(u) * v
	// coefficient denote c
	void assemble_matrix_vol_scalar(const double coefficient = 1.);

};

// diffusion operator with variable coefficient
// volume integral of (k * grad u) * grad v + flux integral of {k * grad u} * [v]
// here (k * grad u) is interpolated with Lagrange or Hermite basis
class DiffusionIntpGradU :
	public BilinearFormIntp
{
public:
	DiffusionIntpGradU(DGSolution & dgsolution, OperatorMatrix1D<LagrBasis, AlptBasis> & oper_matx_lagr, OperatorMatrix1D<HermBasis, AlptBasis> & oper_matx_herm) :
		BilinearFormIntp(dgsolution, oper_matx_lagr, oper_matx_herm) {};
	~DiffusionIntpGradU() {};

	void assemble_matrix(const int dim, const std::string & interp_type);

	// volume integral of (k * grad u) * grad v
	void assemble_matrix_vol_lagr(const int dim, const double coefficient = 1.);

	void assemble_matrix_vol_herm(const int dim, const double coefficient = 1.);

	// flux integral of {k * grad u} * [v]
	void assemble_matrix_flx_lagr(const int dim, const double coefficient = 1.);

	void assemble_matrix_flx_herm(const int dim, const double coefficient = 1.);
};

// diffusion operator with variable coefficient
// flux integral of - {k * grad v} * [u]
// - {k v_x} * [u]
// = - 1/2 * ((k-) * (vx-) + (k+) * (vx+)) * [(u+) - (u-)]
// = - 1/2 * ( ((k-)(u+)) * (vx-) + ((k+)(u+)) * (vx+) - ((k-)(u-)) * (vx-) - ((k+)(u-)) * (vx+) )
// = - 1/2 * ( ((k-)(u+)) * (vx-) - ((k-)(u-)) * (vx-) ) 
//  - 1/2 * ( ((k+)(u+)) * (vx+) - ((k+)(u-)) * (vx+) )
class DiffusionIntpU :
	public BilinearFormIntp
{
public:
	// if sign = -1, this matrix denotes terms involving k-
	// i.e. - 1/2 * ( ((k-)(u+)) * (vx-) - ((k-)(u-)) * (vx-) )
	// if sign = 1, this matrix denotes terms involving k+
	// i.e. - 1/2 * ( ((k+)(u+)) * (vx+) - ((k+)(u-)) * (vx+) )
	DiffusionIntpU(DGSolution & dgsolution, OperatorMatrix1D<LagrBasis, AlptBasis> & oper_matx_lagr, OperatorMatrix1D<HermBasis, AlptBasis> & oper_matx_herm, const int sign_) :
		BilinearFormIntp(dgsolution, oper_matx_lagr, oper_matx_herm), sign(sign_) {};
	~DiffusionIntpU() {};

	void assemble_matrix(const int dim, const std::string & interp_type);

private:
	const int sign;
};


// calculate matrix corresponding to artificial viscosity in DG operator
class ArtificialViscosity :
	public BilinearFormAlpt
{
public:
	ArtificialViscosity(DGSolution & dgsolution, OperatorMatrix1D<AlptBasis, AlptBasis> & oper_matx_alpt, AllBasis<AlptBasis> & all_basis_alpt) :
		BilinearFormAlpt(dgsolution, oper_matx_alpt), all_basis_alpt_ptr(&all_basis_alpt) {};
	~ArtificialViscosity() {};

	// two options about all related elements
	// option 1: find all elements that have intersection with the given viscosity elements
	// option 2: find all parents' (or parents' parents) elements
	// Note: they are the same in 1D case
	// Note: choice 2 is a subset of choice 1
	void assemble_matrix(const double viscosity_coefficient, const int viscosity_option = 1);

private:

	// store mass matrix \int_{support_of_w} u * v and stiffness matrix \int_{support_of_w} u_x * v_x for calculating artificial viscosity
	// here basis functions u, v, w are all in 1D
	// key: vector<int>{i, j, k}, i, j, k denote order of u, v, w in all 1D basis functions, respectively
	// value: mass (u*v) or stiff (u_x*v_x) matrix
	static std::map<std::vector<int>, double> visc_mass;
	static std::map<std::vector<int>, double> visc_stiff;

	AllBasis<AlptBasis>* all_basis_alpt_ptr;

	// find all parents (or parents' parents...) elements of a given element
	// including the given element itself
	void find_all_parents(Element* elem, std::unordered_set<Element*> & parent_elem) const;

	// assemble matrix for a given viscosity element
	void assemble_matrix_one_element(Element* visc_elem, const double viscosity_coefficient, const int viscosity_option, const int index_solu_variable = 0, const int index_test_variable = 0);
};