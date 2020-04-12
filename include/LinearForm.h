#pragma once
#include "BilinearForm.h"

class LinearForm
{
public:
    LinearForm(DGSolution & dgsolution, AllBasis<AlptBasis> & all_bas_alpt, const int gauss_quad_num_ = 10);
    ~LinearForm() {};

    Eigen::VectorXd vec_b;

    /**
     * @brief resize the vector (when adaptive) and set to zero
     */
    void resize_zero_vector();

    void copy_source_to_eigenvec();

    void copy_eigenvec_to_source();

protected:

    DGSolution* const dgsolution_ptr;

    AllBasis<AlptBasis>* const all_bas_alpt_ptr;
    
    Quad gauss_quad_1D;

    Quad gauss_quad_2D;

    Quad gauss_quad_3D;

    /// number of gauss quadrature points when calculating integral
    const int gauss_quad_num;

    /**
     * @brief integral of product (f, v), (f, v_x), (f, v_xx) etc. of a given 1D function f and an alpert basis functions or its derivative
     * 
     * @param func function f
     * @param index_alpt_basis global order in all Alpert's basis functions AllBasis<AlptBasis>
     * @param derivative_degree derivative degree of basis function v (0 by default)
     * @return double integral of (f, v), (f, v_x), (f, v_xx) etc.
     */
    double prod_func_bas_1D(std::function<double(double)> func, const int index_alpt_basis, const int derivative_degree = 0) const;

    /**
     * @brief integral of product of a given 2D function (in nonseparable form) and an alpert basis functions
     * 
     * @param func function f in nonseparable form
     * @param index_alpt_basis 
     * @return double 
     */
    double prod_func_bas_2D_no_separable(std::function<double(std::vector<double>)> func, const std::vector<int> & index_alpt_basis, const std::vector<int> & derivative_degree = std::vector<int>(2, 0)) const;

    /**
     * @brief integral of product of a given 2D function (in separable form) and an alpert basis functions
     * 
     * @param func 
     * @param index_alpt_basis 
     * @return double 
     */
    double prod_func_bas_2D_separable(std::function<double(double, int)> func, const std::vector<int> & index_alpt_basis) const { return 0.; };

    /**
     * @brief integral of product of a given 2D function (a summation of separable form) and an alpert basis functions
     * 
     * @param func 
     * @param index_alpt_basis 
     * @return double 
     */
    double prod_func_bas_2D_separable_sum(std::vector<std::function<double(double, int)>> func, const std::vector<int> & index_alpt_basis) const { return 0.; };

    /**
     * @brief integral of product of a given 3D function (in nonseparable form) and an alpert basis functions
     * 
     * @param func function f in nonseparable form
     * @param index_alpt_basis 
     * @return double 
     */
    double prod_func_bas_3D_no_separable(std::function<double(std::vector<double>)> func, const std::vector<int> & index_alpt_basis, const std::vector<int> & derivative_degree = std::vector<int>(3, 0)) const;
};



// integral of (f, v) on 1D domain
class Domain1DIntegral:
    public LinearForm
{
public:
    // constructor for a function of x (dim of x is d)
    Domain1DIntegral(DGSolution & dgsolution, AllBasis<AlptBasis> & all_bas_alpt, std::function<double(double, int)> func_x_, const int gauss_quad_num = 10):
        LinearForm(dgsolution, all_bas_alpt, gauss_quad_num), func_x(func_x_) {};
    ~Domain1DIntegral() {};

    // assemble vector for a function of x, i.e. calculate (f, v)
    void assemble_vector();

private:
    std::function<double(double, int)> func_x;
};

/**
 * @brief integral of (f, v) on 2D domain
 * 
 */
class Domain2DIntegral:
    public LinearForm
{
public:
    // constructor for a function of x (dim of x is d)
    Domain2DIntegral(DGSolution & dgsolution, AllBasis<AlptBasis> & all_bas_alpt, std::function<double(std::vector<double>, int)> func_x_, const int gauss_quad_num = 10):
        LinearForm(dgsolution, all_bas_alpt, gauss_quad_num), func_x(func_x_) {};
    ~Domain2DIntegral() {};

    /**
     * @brief assemble vector for a function of x, i.e. calculate (f, v)
     * 
     */
    void assemble_vector();

private:
    std::function<double(std::vector<double>, int)> func_x;
};

/**
 * @brief integral of (f, v) on 3D domain
 * 
 */
class Domain3DIntegral:
    public LinearForm
{
public:
    // constructor for a function of x (dim of x is d)
    /**
     * @brief Construct a new Domain 3D Integral object
     * 
     * @param dgsolution 
     * @param all_bas_alpt 
     * @param func_x_       int denote index of unknown variable
     * @param gauss_quad_num 
     */
    Domain3DIntegral(DGSolution & dgsolution, AllBasis<AlptBasis> & all_bas_alpt, std::function<double(std::vector<double>, int)> func_x_, const int gauss_quad_num = 10):
        LinearForm(dgsolution, all_bas_alpt, gauss_quad_num), func_x(func_x_) {};
    ~Domain3DIntegral() {};

    /**
     * @brief assemble vector for a function of x, i.e. calculate (f, v)
     * 
     */
    void assemble_vector();

private:
    std::function<double(std::vector<double>, int)> func_x;
};

/**
 * @brief only works for source term in (summation of) seperable form
 * 
 */
class DomainIntegral:
    public LinearForm
{
public:
    /**
     * @brief Construct a new Domain Integral object
     * 
     * @param dgsolution 
     * @param all_bas_alpt 
     * @param func          first index: unknown variable; second index: num of separable function
     * @param gauss_quad_num 
     */
    DomainIntegral(DGSolution & dgsolution, AllBasis<AlptBasis> & all_bas_alpt, std::vector<std::vector<std::function<double(double, int)>>> func, const int gauss_quad_num = 10);    
    // overload
    DomainIntegral(DGSolution & dgsolution, AllBasis<AlptBasis> & all_bas_alpt, std::function<double(double x, int dim, int num_separable, int var_index)> func, std::vector<int> num_separable, const int gauss_quad_num = 10);
    ~DomainIntegral() {};

    void assemble_vector();

private:
    // 1st index: index of unknown variable
    // 2nd index: index of separable function
    // VecMultiD<double>: size of (dim) * (num of all alpert basis), coefficients of separable functions in each dimension
    std::vector<std::vector<VecMultiD<double>>> coeff;
};

/**
 * @brief assemble vector for the delta source \delta(x-x0)
 * 
 */
class DomainDeltaSource:
    public LinearForm
{
public:
    DomainDeltaSource(DGSolution & dgsolution, AllBasis<AlptBasis> & all_bas_alpt, const std::vector<double> & x0);
    ~DomainDeltaSource() {};

    void assemble_vector(const double coefficient = 1.);

private:
    // location of delta source
    std::vector<double> point_x0;
    
    // store value of Alpert basis function in 1D at point x0
    // size of dim * num_basis_1D
    std::vector<std::vector<double>> value_basis_1D_point_x0;

    /**
     * @brief calculate value of Alpert basis function in 1D at point x0
     * 
     * @param x0 location of delta source
     * @return std::vector<double> value of Alpert basis function in 1D at point x0
     */
    std::vector<double> evaluate_basis_1D(const double x0);

    /**
     * @brief value of an Alpert basis function in tensor product of 1D basis at the point x0
     * 
     * @param x0 
     * @param order_global_basis 
     * @return double 
     */
    double val_basis_multiD(const std::vector<double> & x0, const std::vector<int> & order_global_basis) const;
};

/**
 * @brief integral of (f, v) or (f, v_x) on boundary of 1D, i.e., integral on a 1D interval [0, 1] (only for 2D problem)
 * 
 */
class Boundary1DIntegral:
    public LinearForm
{
public:
    // constructor
    Boundary1DIntegral(DGSolution & dgsolution, AllBasis<AlptBasis> & all_bas_alpt, const int gauss_quad_num = 10):
        LinearForm(dgsolution, all_bas_alpt, gauss_quad_num) { assert(dgsolution.DIM==2); };
    ~Boundary1DIntegral() {};

    /**
     * @brief calculate integral of (f, v) or (f, v_x) on boundary of 1D
     * 
     * @param func_1D       a function from 1D to 1D
     * @param boundary_dim  boundary dimension (if 0, then boundary in x direction, i.e., left or right boundary; if 1, then boundary in y direction, i.e., down or up boundary)
     * @param boundary_sign if -1, left boundary; if 1, right boundary
     * @param derivative    derivative of basis function, vector of size dim (each component denotes derivative degree in each dimension)
     */
    void assemble_vector_boundary1D(std::function<double(double)> func_1D, const int boundary_dim, const int boundary_sign, const std::vector<int> & derivative);
};

/**
 * @brief integral of (f, v) or (f, v_x) on boundary of 2D, i.e., integral on a 2D square [0, 1]^2 (only for 3D problem)
 * 
 */
class Boundary2DIntegral:
    public LinearForm
{
public:
    Boundary2DIntegral(DGSolution & dgsolution, AllBasis<AlptBasis> & all_bas_alpt, const int gauss_quad_num = 10):
        LinearForm(dgsolution, all_bas_alpt, gauss_quad_num) { assert(dgsolution.DIM==3); };
    ~Boundary2DIntegral() {};

    /**
     * @brief calculate integral of (f, v) or (f, v_x) on boundary of 2D
     * 
     * @param func_2D       a function from 2D to 1D
     * @param boundary_dim  boundary dimension (if 0, then boundary in x direction, i.e., left or right boundary; if 1, then boundary in y direction, i.e., down or up boundary)
     * @param boundary_sign if -1, left boundary; if 1, right boundary
     * @param derivative    derivative of basis function, vector of size dim (each component denotes derivative degree in each dimension)
     * 
     * @note func_2D is a function from 2D to 1D, order of variable should be correct
     */
    void assemble_vector_boundary2D(std::function<double(std::vector<double>)> func_2D, const int boundary_dim, const int boundary_sign, const std::vector<int> & derivative); 
};