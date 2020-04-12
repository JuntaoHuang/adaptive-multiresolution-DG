#pragma once
#include "DGAdapt.h"
#include "VecMultiD.h"

/**
 * @brief Compute matrix-vector multiplication in multidimensions using fast LU algorithm
 * 
 */
class FastMultiplyLU
{
public:
    FastMultiplyLU(DGSolution & dgsolution): dgsolution_ptr(&dgsolution) {};
    ~FastMultiplyLU() {};

protected:
    /**
     * @brief do transformation (or matrix-vector multiplication) in 1D with given matrix and operator_type
     * 
     * @param mat_1D 
     * @param LU                    "L" (lower and include diagnoal), "U" (strictly upper) or "full" (full) part of 1D transformation matrix
     * @param operator_type         if vol (use pointers related to volume term); if flx (use pointers related to flux term)
     * @param dim_transform         dimension which we do 1D transformation
     * @param pmax_trans_from       max polynomial degree transform from
     * @param pmax_trans_to         max polynomial degree transform to
     * @param coefficient       
     * @param vec_index_trans_from  index of unknown variable we do transformation from
     * @param vec_index_trans_to    index of unknown variable we do transformation to
     * 
     * @note  if transform from alpert basis to interpolation basis, then pmax_trans_from = Element::PMAX_alpt and pmax_trans_to = Element::PMAX_intp
     */
    void transform_1D(const VecMultiD<double> & mat_1D, const std::string LU, const std::string operator_type, const int dim_transform, 
                    const int pmax_trans_from, const int pmax_trans_to, const double coefficient = 1.0, 
                    const int vec_index_trans_from = 0, const int vec_index_trans_to = 0);

    // resize ucoe_trans_from and ucoe_trans_to in each element before do transformation
    // with given size of vector
    void resize_ucoe_transfrom(const std::vector<int> & size_trans_from, const int vec_index = 0);
    void resize_ucoe_transto(const std::vector<int> & size_trans_to, const int vec_index = 0);

    // copy value in Element::ucoe_trans_to to Element::ucoe_trans_from
    void copy_transto_to_transfrom(const int vec_index = 0);

    /**
     * @brief generate a series of vector (size of (dim + 1) * dim) which denotes transformation size with given transformation dimension order
     * 
     * @param size_trans_from 
     * @param size_trans_to 
     * @param dim 
     * @param transform_order a vector of size the same with dim
     * @return std::vector<std::vector<int>> 
     * 
     * @note   Here is an example: if take size_trans_from = 5, size_trans_to = 4, dim = 3, transform_order = {1, 2, 0}
     *          then returned vector is 
     *          vec[0] = {5, 5, 5}
     *          vec[1] = {5, 4, 5}
     *          vec[2] = {5, 4, 4}
     *          vec[3] = {4, 4, 4}
     */
    std::vector<std::vector<int>> series_vec_transform_size(const int size_trans_from, const int size_trans_to, const int dim, const std::vector<int> & transform_order);

    /**
     * @brief generate a seriez of vector which denote transformation dimension order and LU order
     * 
     * @param dim 
     * @param dim_order_transform 
     * @param LU_order_transform
     * 
     * @note    Here is an example:
     *          If dim = 2, then result is:
     *          dim_order_transform:
     *              [0] = {0 1}
     *              [1] = {1 0}
     *          LU_order_transform:
     *              [0] = {L full}
     *              [1] = {full U}
     *          If dim = 3, then result is:
     *          dim_order_transform:
     *              [0] = {0 1 2}
     *              [1] = {0 2 1}
     *              [2] = {1 2 0}
     *              [3] = {2 0 1}
     *          LU_order_transform:
     *              [0] = {L L full}
     *              [1] = {L full U}
     *              [2] = {L full U}
     *              [3] = {full U U}
     */
    void generate_transform_order(const int dim, std::vector<std::vector<int>> & dim_order_transform, std::vector<std::vector<std::string>> & LU_order_transform);

    DGSolution* const dgsolution_ptr;    
};


/**
 * @brief   Fast transform coefficients of Alpert basis (stored in Element::ucoe_alpt) 
 *          to values (and derivatives) at Lagrange or Hermite interpolation points (stored in Element::up_intp)
 *          This is the first step in interpolation. 
 *          The second step is to transform the values (and derivatives) at interpolation points to coefficients of interpolation basis,
 *          which will be done in class Interpolation.
 */
class FastInterpolation:
    public FastMultiplyLU
{
public:
    FastInterpolation(DGSolution & dgsolution): FastMultiplyLU(dgsolution) {};
    ~FastInterpolation() {};

protected:
    /**
     * @brief transform Element::ucoe_alpt to Element::up_intp in any dimension
     */
    void transform_ucoealpt_to_upintp(const std::vector<const VecMultiD<double>*> & alpt_basis_pt_1D);

    // overload for matrix are the same
    void transform_ucoealpt_to_upintp(const VecMultiD<double> & alpt_basis_pt_1D);

private:
    // copy value in ucoe_alpt to ucoe_trans_from
    void copy_ucoealpt_to_transfrom(const int vec_index = 0);

    // add value in ucoe_trans_to to up_intp
    void add_transto_to_upintp(const int vec_index = 0);

    // clear Element::up_intp to zero
    void upintp_set_zero(const int vec_index = 0);
    
    /**
     * @brief   Do 1D transformation in a given dimension.
     *          If the first step, then copy Element::ucoe_alpt to Element::transform_from and then transform to Element::transform_to.
     *          If not the middle step, then copy Element::transform_to (from 1D transformtion in the last step) to Element::transform_from and then transform to Element::transform_to.
     *          If the last step, then there is an extra step in the last, add Element::transform_to to Element::up_intp.
     *          This will update Element::transform_to (if not last step) or Element::up_intp (if last step)
     * 
     * @param mat_1D 
     * @param LU 
     * @param size_trans_from 
     * @param size_trans_to 
     * @param dim 
     * @param is_first_step     if true, denote the first step
     * @param is_last_step      if true, denote the last step
     */
    void transform_1D_from_ucoealpt_to_upintp(const VecMultiD<double> & mat_1D, const std::string LU, const std::vector<int> & size_trans_from, const std::vector<int> & size_trans_to, const int dim, const bool is_first_step, const bool is_last_step);

    /**
     * @brief Do multidimension transformation in a dimension by dimension approach.
     * 
     * @param mat_1D_array          size of dim, store 1D transformation matrix in all dimensions, in the order from 0 to (dim-1)
     * @param dim_order_transform   size of dim, denote order of transformation in 1D
     * @param LU                    size of dim, use "L", "U", or "full" in each transformation 1D 
     * 
     * @note    Here is an example in 3D:
     *          mat_1D_array = {mat_1D_x, mat_1D_y, mat_1D_z}, dim_order_transform = {1, 0, 2}, matx_order = {"L", "full", "U"}.
     *          Then this means first do transformation in y direction with lower part of mat_1D_y, 
     *          and then do transformation in x direction with full mat_1D_x
     *          and last do transformation in z direction with upper part of mat_1D_z
     */
    void transform_multiD_partial_sum(const std::vector<const VecMultiD<double>*> & mat_1D_array, const std::vector<int> & dim_order_transform, const std::vector<std::string> & LU);
};


/**
 * @brief   Fast transform coefficients of Alpert basis (stored in Element::ucoe_alpt) 
 *          to values at Lagrange interpolation points (stored in Element::up_intp)
 *          This is the first step in Lagrange interpolation.
 */
class FastLagrIntp:
    public FastInterpolation
{
public:
    FastLagrIntp(DGSolution & dgsolution, const std::vector<std::vector<double>> & Lag_pt_Alpt_1D,
                 const std::vector<std::vector<double>> & Lag_pt_Alpt_1D_d1);
    ~FastLagrIntp() {};

    // transform Element::ucoe_alpt to Element::up_intp
    void eval_up_Lagr();

    // transform Element::ucoe_alpt to Element::u_xd0_intp ---> u_xd0 means that derivative along xd0 direction
    void eval_der_up_Lagr(const int d0);

protected:
    // Alpert basis function in 1D evaluated at Lagrange interpolation points
    VecMultiD<double> alpt_basis_Lagr_pt;

    // Alpert basis function in 1D evaluated at Lagrange interpolation points
    VecMultiD<double> alpt_basis_der_Lagr_pt;
};


/**
 * @brief   Fast transform coefficients of Alpert basis (stored in Element::ucoe_alpt) 
 *          to values and derivatives at Hermite interpolation points (stored in Element::up_intp)
 *          This is the first step in Hermite interpolation.
 */
class FastHermIntp:
    public FastInterpolation
{
public:
    FastHermIntp(DGSolution & dgsolution, const std::vector<std::vector<double>> & Her_pt_Alpt_1D);
    ~FastHermIntp() {};

    // transform Element::ucoe_alpt to Element::up_intp
    void eval_up_Herm();

protected:
    // Alpert basis function in 1D evaluated at Hermite interpolation points
    VecMultiD<double> alpt_basis_herm_pt;
};

/**
 * @brief   Fast transform coefficients of interpolation basis (stored in Element::ucoe_intp)
 *          to coefficients of Alpert basis (stored in Element::ucoe_alpt)
 *          This class will be applied in the initialization for non-separable initial values.
 *          First use (adaptive) interpolation basis to interpolate the given function 
 *          and then transform to coefficients of Alpert basis.
 */
class FastInitial:
    public FastMultiplyLU
{
public:
    FastInitial(DGSolution & dgsolution): FastMultiplyLU(dgsolution) {};
    ~FastInitial() {};

protected:
    // copy value in ucoe_intp to ucoe_trans_from
    void copy_ucoeintp_to_transfrom(const int vec_index = 0);

    // add value in ucoe_trans_to to ucoe_alpt
    void add_transto_to_ucoealpt(const int vec_index = 0);

    // clear Element::ucoe_alpt to zero
    void ucoealpt_set_zero(const int vec_index = 0);

    /**
     * @brief fast transform Element::ucoe_intp to Element::ucoe_alpt in any dimension
     * 
     * @param inner_product_1D  inner product of Alpert basis and interpolation basis in 1D
     */
    void transform_ucoeintp_to_ucoealpt(const std::vector<const VecMultiD<double>*> & inner_product_1D);

    // overload for matrix are the same
    void transform_ucoeintp_to_ucoealpt(const VecMultiD<double> & inner_product_1D);

    /**
     * @brief fast transform Element::ucoe_intp to Element::ucoe_alpt in 2D
     * 
     * @param inner_product_1D  inner product of Alpert basis and interpolation basis in 1D
     */
    void transform_ucoeintp_to_ucoealpt_2D(const VecMultiD<double> & inner_product_1D);
    
private:
    /**
     * @brief   Do 1D transformation in a given dimension.
     *          If the first step, then copy Element::ucoe_intp to Element::transform_from and then transform to Element::transform_to.
     *          If not the middle step, then copy Element::transform_to (from 1D transformtion in the last step) to Element::transform_from and then transform to Element::transform_to.
     *          If the last step, then there is an extra step in the last, add Element::transform_to to Element::ucoe_alpt.
     *          This will update Element::transform_to (if not last step) or Element::ucoe_alpt (if last step)
     * 
     * @param mat_1D 
     * @param LU 
     * @param size_trans_from 
     * @param size_trans_to 
     * @param dim 
     * @param is_first_step     if true, denote the first step
     * @param is_last_step      if true, denote the last step
     */
    void transform_1D_from_ucoeintp_to_ucoealpt(const VecMultiD<double> & mat_1D, const std::string LU, const std::vector<int> & size_trans_from, const std::vector<int> & size_trans_to, const int dim, const bool is_first_step, const bool is_last_step);

    /**
     * @brief Do multidimension transformation in a dimension by dimension approach.
     * 
     * @param mat_1D_array          size of dim, store 1D transformation matrix in all dimensions, in the order from 0 to (dim-1)
     * @param dim_order_transform   size of dim, denote order of transformation in 1D
     * @param LU                    size of dim, use "L", "U", or "full" in each transformation 1D 
     * 
     * @note    Here is an example in 3D:
     *          mat_1D_array = {mat_1D_x, mat_1D_y, mat_1D_z}, dim_order_transform = {1, 0, 2}, matx_order = {"L", "full", "U"}.
     *          Then this means first do transformation in y direction with lower part of mat_1D_y, 
     *          and then do transformation in x direction with full mat_1D_x
     *          and last do transformation in z direction with upper part of mat_1D_z
     */
    void transform_multiD_partial_sum(const std::vector<const VecMultiD<double>*> & mat_1D_array, const std::vector<int> & dim_order_transform, const std::vector<std::string> & LU);
};


/**
 * @brief   Fast transform coefficients of Hermite interpolation basis (stored in Element::ucoe_intp)
 *          to coefficients of Alpert basis (stored in Element::ucoe_alpt)
 *          inherit from class FastInitial    
 */
class FastHermInit:
    public FastInitial
{
public:
    FastHermInit(DGSolution & dgsolution, const OperatorMatrix1D<HermBasis, AlptBasis> & matrix);
    ~FastHermInit() {};

    // transform Element::ucoe_intp to Element::ucoe_alpt in any dimension
    void eval_ucoe_Alpt_Herm();

    // transform Element::ucoe_intp to Element::ucoe_alpt in 2D
    void eval_ucoe_Alpt_Herm_2D();

protected:
    // Integral of Hermite basis function in 1D and Alpert basis function in 1D
    VecMultiD<double> Herm_basis_alpt_basis_int;
};


/**
 * @brief   Fast transform coefficients of Lagrange interpolation basis (stored in Element::ucoe_intp)
 *          to coefficients of Alpert basis (stored in Element::ucoe_alpt)
 *          inherit from class FastInitial    
 */
class FastLagrInit:
    public FastInitial
{
public:
    FastLagrInit(DGSolution & dgsolution, const OperatorMatrix1D<LagrBasis, AlptBasis> & matrix);
    ~FastLagrInit() {};

    // transform Element::ucoe_intp to Element::ucoe_alpt in any dimension
    void eval_ucoe_Alpt_Lagr();

    // transform Element::ucoe_intp to Element::ucoe_alpt in 2D
    void eval_ucoe_Alpt_Lagr_2D();

protected:
    // Integral of Lagrange basis function in 1D and Alpert basis function in 1D
    VecMultiD<double> Lagr_basis_alpt_basis_int;
};

// transform Element::ucoe_intp to Element::ucoe_alpt using Lagrange interpolation
// this works for full grid in any dimension
class FastLagrFullInit:
    public FastLagrInit
{
public:
    FastLagrFullInit(DGSolution & dgsolution, const OperatorMatrix1D<LagrBasis, AlptBasis> & matrix);
    ~FastLagrFullInit() {};

    // transform Element::ucoe_intp to Element::ucoe_alpt
    void eval_ucoe_Alpt_Full();

private:
    // return if it is full grid
    bool is_full_grid();
};

/**
 * @brief Compute rhs using fast LU algorithm directly without assemble matrix
 * 
 */
class FastRHS:
    public FastMultiplyLU
{
public:
    FastRHS(DGSolution & dgsolution): FastMultiplyLU(dgsolution) {};
    ~FastRHS() {};

protected:

    void transform_fucoe_to_rhs(const std::vector<const VecMultiD<double>*> & mat_1D, const std::vector<std::string> & operator_type, const int dim_interp, const double coefficient = 1.0, const int vec_index = 0);

    // calculate rhs from coefficients of interpolation basis 
    // given transformation matrix (x and y directions) in 1D
    // LU-algorithm: 
    //      split mat_x = L_x + U_x and keep mat_y not split
    //      step 1. perform 1D transformation in x direction with L_x, then 1D transformation in y direction with mat_y
    //      step 2. perform 1D transformation in y direction with mat_y, then 1D transformation in y direction with U_x
    // operator_type_x: "vol" or "flx", denote mat_x is volume or flux integral
    // operator_type_y: "vol" or "flx", denote mat_y is volume or flux integral
    // dim_interp: index of fucoe_intp
    // coefficient: multiply by a given constant (1.0 by default)
    // 
    // NOTE: this function only works for 2D case
    //      this function will ADD (not copy) computational result to Element::rhs
    void rhs_2D(const VecMultiD<double> & mat_x, const VecMultiD<double> & mat_y, const std::string operator_type_x, const std::string operator_type_y, const int dim_interp, const double coefficient = 1.0);

private:

    void transform_1D_from_fucoe_to_rhs(const VecMultiD<double> & mat_1D, const std::string & LU, const std::string & operator_type, 
            const std::vector<int> & size_trans_from, const std::vector<int> & size_trans_to, const int dim_transform_1D,
            const bool is_first_step, const bool is_last_step, const int dim_interp, const int vec_index, const double coefficient = 1.0);

    void transform_multiD_partial_sum(const std::vector<const VecMultiD<double>*> & mat_1D_array, const std::vector<int> & dim_order_transform, 
            const std::vector<std::string> & LU_order_transform, const std::vector<std::string> & operator_type, const int dim_interp, const int vec_index, const double coefficient = 1.0);

    // set Element::rhs to zero
    void set_rhs_zero(const int vec_index = 0);

    // copy value in fucoe_intp in dim_interp to ucoe_trans_from 
    void copy_fucoe_to_transfrom(const int dim_interp, const int vec_index = 0);

    // add value in Element::ucoe_trans_to to Element::rhs
    void add_transto_to_rhs(const int vec_index = 0);

    // void transform_1D_from_fucoe_to_rhs(const VecMultiD<double> & mat_1D, const std::string LU, const std::string operator_type, const std::vector<int> & size_trans_from, const std::vector<int> & size_trans_to, const int dim, const bool is_first_step);

    // L_x * mat_y
    void rhs_2D_sum_1st(const VecMultiD<double> & mat_x, const VecMultiD<double> & mat_y, const std::string operator_type_x, const std::string operator_type_y, const int dim_interp, const double coefficient = 1.0);

    // mat_y * U_x
    void rhs_2D_sum_2nd(const VecMultiD<double> & mat_x, const VecMultiD<double> & mat_y, const std::string operator_type_x, const std::string operator_type_y, const int dim_interp, const double coefficient = 1.0);
};


// scalar hyperbolic equation (in 2D) with the same physical flux (e.g. Burgers equation) in x and y directions
// use Hermite interpolation basis
class HyperbolicSameFluxHermRHS:
    public FastRHS
{
public:
    HyperbolicSameFluxHermRHS(DGSolution & dgsolution, OperatorMatrix1D<HermBasis, AlptBasis> & oper_matx_herm): 
        FastRHS(dgsolution), oper_matx_herm_ptr(&oper_matx_herm) {};
    ~HyperbolicSameFluxHermRHS() {};

    // calculate rhs of volume integral term for scalar equation
    void rhs_vol_scalar();

    // overload for only one given dimension
    void rhs_vol_scalar(const int dim);

    // calculate rhs of flux integral term involving Hermite interpolation basis
    void rhs_flx_intp_scalar();

    // overload for only one given dimension
    void rhs_flx_intp_scalar(const int dim);

private:
    OperatorMatrix1D<HermBasis, AlptBasis>* const oper_matx_herm_ptr;
};

// scalar hyperbolic equation (in 2D) with different physical flux (e.g. KPP problem) in x and y directions
// use Hermite interpolation basis
class HyperbolicDiffFluxHermRHS:
    public FastRHS
{
public:
    HyperbolicDiffFluxHermRHS(DGSolution & dgsolution, OperatorMatrix1D<HermBasis, AlptBasis> & oper_matx_herm): 
        FastRHS(dgsolution), oper_matx_herm_ptr(&oper_matx_herm) {};
    ~HyperbolicDiffFluxHermRHS() {};

    // calculate rhs of volume integral term for scalar equation
    void rhs_vol_scalar();

    // calculate rhs of flux integral term involving Hermite interpolation basis
    void rhs_flx_intp_scalar();

private:
    OperatorMatrix1D<HermBasis, AlptBasis>* const oper_matx_herm_ptr;
};

// scalar hyperbolic equation (in 2D) with the same physical flux (e.g. Burgers equation) in x and y directions
// use Lagrange interpolation basis
class HyperbolicSameFluxLagrRHS:
    public FastRHS
{    
public:
    HyperbolicSameFluxLagrRHS(DGSolution & dgsolution, OperatorMatrix1D<LagrBasis, AlptBasis> & oper_matx_lagr): 
        FastRHS(dgsolution), oper_matx_lagr_ptr(&oper_matx_lagr) {};
    ~HyperbolicSameFluxLagrRHS() {};

    // calculate rhs of volume integral term for scalar equation
    void rhs_vol_scalar();

    // calculate rhs of flux integral term involving Hermite interpolation basis
    void rhs_flx_intp_scalar();

private:
    OperatorMatrix1D<LagrBasis, AlptBasis>* const oper_matx_lagr_ptr;
};

// scalar hyperbolic equation (in 2D) with different physical flux (e.g. KPP problem) in x and y directions
// use Lagrange interpolation basis
class HyperbolicDiffFluxLagrRHS:
    public FastRHS
{    
public:
    HyperbolicDiffFluxLagrRHS(DGSolution & dgsolution, OperatorMatrix1D<LagrBasis, AlptBasis> & oper_matx_lagr): 
        FastRHS(dgsolution), oper_matx_lagr_ptr(&oper_matx_lagr) {};
    ~HyperbolicDiffFluxLagrRHS() {};

    // calculate rhs of volume integral term for scalar equation
    void rhs_vol_scalar();

    // calculate rhs of flux integral term involving Hermite interpolation basis
    void rhs_flx_intp_scalar();

private:
    OperatorMatrix1D<LagrBasis, AlptBasis>* const oper_matx_lagr_ptr;
};

// use Lagrange interpolation basis to approximate k * grad u and k * u
class DiffusionRHS:
    public FastRHS
{
public:
    DiffusionRHS(DGSolution & dgsolution, OperatorMatrix1D<LagrBasis, AlptBasis> & oper_matx_lagr):
        FastRHS(dgsolution), oper_matx_lagr_ptr(&oper_matx_lagr) {};
    ~DiffusionRHS() {};

    // volume integral of (k * grad u) * grad v
    void rhs_vol();

    // flux integral of {k * grad u} * [v] (terms involving grad u)
    void rhs_flx_gradu();

    // flux integral of - {k * grad v} * [u] (terms involving u)
    // x direction: 
    // - {k v_x} * [u]  = - 1/2 * ((k-) * (vx-) + (k+) * (vx+)) * [(u+) - (u-)]
    //                  = - 1/2 * ( ((k-)(u+)) * (vx-) + ((k+)(u+)) * (vx+) - ((k-)(u-)) * (vx-) - ((k+)(u-)) * (vx+) )
    //                  = - 1/2 * ( ((k-)(u+)) * (vx-) - ((k-)(u-)) * (vx-) ) - 1/2 * ( ((k+)(u+)) * (vx+) - ((k+)(u-)) * (vx+) )
    //                  = - 1/2 * [k- u] * vx- - 1/2 * [k+ u] * vx+
    //                  ( if ignore difference of k- and k+ )
    //                  = - 1/2 * [k * u] * (vx- + vx+)
    //                  = - [k * u] * {vx}
    // similar result applies for y direction
    void rhs_flx_u();

    // - 1/2 * [k- u] * vx-
    void rhs_flx_k_minus_u();

    // - 1/2 * [k+ u] * vx+
    void rhs_flx_k_plus_u();

private:
    OperatorMatrix1D<LagrBasis, AlptBasis>* const oper_matx_lagr_ptr;
};

/**
 * @brief Fast algorithm for computing the rhs for Hamilton Jacobian equations
 * 
 */
class FastRHSHamiltonJacobi:
    public FastRHS
{
public:
    FastRHSHamiltonJacobi(DGSolution & dgsolution, OperatorMatrix1D<LagrBasis, AlptBasis> & oper_matx_lagr): 
        FastRHS(dgsolution), oper_matx_lagr_ptr(&oper_matx_lagr) {};
    ~FastRHSHamiltonJacobi() {};

    // volume integral of nonlinear Hamiltonian term
    void rhs_nonlinear();

private:
    OperatorMatrix1D<LagrBasis, AlptBasis>* const oper_matx_lagr_ptr;
};