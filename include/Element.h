#pragma once
#include "libs.h"
#include "VecMultiD.h"
#include "AllBasis.h"
#include "AlptBasis.h"
#include "LagrBasis.h"
#include "HermBasis.h"
#include "Hash.h"

class Element
{
public:
	Element(const std::vector<int> & level_, const std::vector<int> & suppt_, AllBasis<AlptBasis> & all_bas, Hash & hash);
	~Element() {};

	static int DIM;		// dimension of our problem
	static int VEC_NUM;	// num of unknown variables (or equations)
	static std::vector<std::vector<bool>> is_intp;

	const std::vector<int> level;
	const std::vector<int> suppt;

	// left and right end point of support in multidimension
	// this information is also contained in supp_interv
	std::vector<double> xl;		
	std::vector<double> xr;
		
	std::vector<std::vector<double>> dis_point;		// discontinuous point in each dimension
	std::vector<std::vector<double>> supp_interv;	// interval of support in each dimension
	
	std::vector<int> order_elem;	// order for this element, size of dim, consists of order of element in 1D dimension
	int hash_key;					// hash key for this element
	
	// global order of Alpert and interpolation basis in DGSolution
	// this will be used in assemble matrix for DG bilinear form
	// e.g. element.order_alpt_basis_in_dg[variable_index].at(order_local_basis);
	// it is ordered by first loop over dgmap, second loop over all unknown variable, last loop over all basis
	// see details in void DGSolution::update_order_all_basis_in_dgmap()
	// it should be updated after adaptive
	// really need to store the order  
	std::vector<VecMultiD<int>> order_alpt_basis_in_dg;	
	std::vector<VecMultiD<int>> order_intp_basis_in_dg;

	// local order of Alpert and interpolation basis
	// i.e. order of basis in this element, which is also degree of polynomial
	// size of (num of multidimension basis * DIM)
	// order_global_alpt[n]: vector of size DIM (i_1, i_2, ..., i_DIM)
	// n-th multidimension basis in this element, which is a tensor product of DIM many 1D basis
	// i_d denote the polynomia degree of 1D basis
	// it keeps unchaged in adaptive
	std::vector<std::vector<int>> order_local_alpt;		
	std::vector<std::vector<int>> order_local_intp;

	// global order of Alpert and interpolation basis 
	// i.e. order of basis in all 1D Alpert and interpolation basis (AllBasis)
	// size of (num of multidimension basis * DIM)
	// order_global_alpt[n]: vector of size DIM (i_1, i_2, ..., i_DIM)
	// n-th multidimension basis in this element, which is a tensor product of DIM many 1D basis
	// i_d denote the global order of 1D basis in d-th dimension in all 1D basis
	// it keeps unchaged in adaptive
	std::vector<std::vector<int>> order_global_alpt;	
	std::vector<std::vector<int>> order_global_intp;

	// coefficients of Alpert's basis
	std::vector<VecMultiD<double>> ucoe_alpt;
	std::vector<VecMultiD<double>> ucoe_alpt_t_m1;	// coefficients at t_(n-1), this will be used in multistep class
	std::vector<VecMultiD<double>> ucoe_alpt_predict;
	std::vector<VecMultiD<double>> ucoe_alpt_predict_t_m1;
	std::vector<VecMultiD<double>> ucoe_tn;		// coefficients at t_n, this will be used in RK class
	std::vector<VecMultiD<double>> ucoe_ut;		// coefficients of v = u_t
	std::vector<VecMultiD<double>> ucoe_ut_predict;		// coefficients of v = u_t
	std::vector<VecMultiD<double>> rhs;
	std::vector<VecMultiD<double>> source;		// store coefficients of source term with the form of a function of x and t

	// store coefficients in fast matrix-vector multiplication
	std::vector<VecMultiD<double>> ucoe_trans_from;
	std::vector<VecMultiD<double>> ucoe_trans_to;
	// ------------------------------------------------------------------------------------------------------------------------
	// 
	// this part will be used for interpolation
	// 
	// ------------------------------------------------------------------------------------------------------------------------
	// fucoe_intp will store coefficients of interpolation basis for f(u)
	// the first index of vector denote unknown variables
	// the second index of vector denote the dimension
	std::vector< std::vector< VecMultiD<double> > > fucoe_intp;
	std::vector< std::vector< VecMultiD<std::vector<double> > > > fucoe_intp_inter;
	std::vector< std::vector< VecMultiD<double> > > fp_intp;

	// coefficients of interpolation basis for u
	// the index of vector 
	std::vector< VecMultiD<double> > ucoe_intp;
	std::vector< VecMultiD<std::vector<double> > > ucoe_intp_inter;
	std::vector< VecMultiD<double> > up_intp;
		
	// ------------------------------------------------------------------------------------------------------------------------
	// 
	// this part will be used in class DGadapt
	// 
	// ------------------------------------------------------------------------------------------------------------------------
	// number of total children and parents
	int num_total_chd, num_total_par;

	// map store existing children and parents elements
	// key: hash key of existing children and parents
	// value: pointer to existing children and parents
	// we do not store num of existing children and parents, just use size of map
	std::unordered_map<int,Element*> hash_ptr_chd;
	std::unordered_map<int,Element*> hash_ptr_par;
	
	bool new_add;
	// ------------------------------------------------------------------------------------------------------------------------	
	// 
	// this part will used to determine elements that will contribute to integral of volume and flux terms in DG operators
	// 
	// ------------------------------------------------------------------------------------------------------------------------
	// store pointers (which are of size DIM) to all other elements that will contribute to integral of volume and flux term when current element is taken as a test function
	// e.g. ptr_vol_alpt[i] stores pointers to the elements that contribute to integral of volume in the i-th dimension when using alpert basis
	// current element has intersection with other element in the i-th dimension and has the same index in other dimensions due to orthogonality
	// the interpolation basis is different because of no orthogonality. in other dimension, not the same index
	// note that ptr_vol_intp is the same for different dimensions
	std::vector<std::unordered_set<Element*>> ptr_vol_alpt;
	std::vector<std::unordered_set<Element*>> ptr_flx_alpt;
	std::vector<std::unordered_set<Element*>> ptr_vol_intp;
	std::vector<std::unordered_set<Element*>> ptr_flx_intp;

	// return true if current element has interaction with another element in the sense of volume and flux integral
	// volume integral with both Alpert basis. It depends on dimension, since in other dimensions which are not equal to given dimension, the element has to be the same index because of orthogonal
	bool is_vol_alpt(const Element & elem, const int dim) const;
	// volume integral with one interpolation basis and one Alpert basis. It does not depend on dimension due to no orthogonal
	bool is_vol_intp(const Element & elem) const;
	// flux integral with Alpert basis. The difference from the volume integral is that two intervals in 1D can intersect or adjacent
	bool is_flx_alpt(const Element & elem, const int dim) const;
	// flux integral with interpolation basis.
	bool is_flx_intp(const Element & elem, const int dim) const;

	// ------------------------------------------------------------------------------------------------------------------------
	// return number of all Alpert basis and interpolation basis in this element
	int size_alpt() const;
	int size_intp() const;
	
	/**
	 * @brief return value (or derivatives) of collection of all Alpert basis (with coefficients ucoe_alpt) in this element at any point x
	 * 
	 * @param x 
	 * @param all_bas 
	 * @param derivative derivative degree
	 * @return std::vector<double> size of VEC_NUM
	 */
	std::vector<double> val(const std::vector<double> & x, const AllBasis<AlptBasis> & all_bas, const std::vector<int> & derivative) const;

	/**
	 * @brief return values of collection of all Lagrange basis (with coefficients ucoe_intp) in this element at any point x
	 * 
	 * @param x 
	 * @param all_bas_Lag
	 * @return std::vector<double> size of VEC_NUM
	 */	
	std::vector<double> val_Lag(const std::vector<double> & x, const AllBasis<LagrBasis> & all_bas_Lag) const;

	/**
	 * @brief return values of collection of all Hermite basis (with coefficients ucoe_intp) in this element at any point x
	 * 
	 * @param x 
	 * @param all_bas_Her 
	 * @return std::vector<double> size of VEC_NUM
	 */
	std::vector<double> val_Her(const std::vector<double> & x, const AllBasis<HermBasis> & all_bas_Her) const;

	/**
	 * @brief return value of flux function of collection of all Lagrange basis (with coefficients fucoe_intp) in this element at any point x
	 * 
	 * @param x 
	 * @param ii index of unknown variable
	 * @param dd index of dimension
	 * @param all_bas_Lag 
	 * @return double 
	 */
	double val_Lag(const std::vector<double> & x, const int ii, const int dd, const AllBasis<LagrBasis> & all_bas_Lag) const;

	/**
	 * @brief return value of flux function of collection of all Hermite basis (with coefficients fucoe_intp) in this element at any point x
	 * 
	 * @param x 
	 * @param ii index of unknown variable
	 * @param dd index of dimension
	 * @param all_bas_Her
	 * @return double 
	 */
	double val_Her(const std::vector<double> & x, const int ii, const int dd, const AllBasis<HermBasis> & all_bas_Her) const;
	
	// ------------------------------------------------------------------------------------------------------------------------

	static int PMAX_alpt;	// max polynomial degree for Alpert's basis functions
	static int PMAX_intp;	// max polynomial degree for interpolation basis functions

	// order of basis in 1D
	static int index_to_order_basis(const int n, const int j, const int p, const int PMAX);	

private:

	// return true if two intervals in 1D intersect
	static bool is_interval_intersect(const std::vector<double> & interval_u, const std::vector<double> & interval_v);

	// return true if two intervals in 1D intersect or adjacent
	static bool is_interval_intersect_adjacent(const std::vector<double> & interval_u, const std::vector<double> & interval_v);

	// return true if exist at least one point (in a set of points) is inside an interval
	static bool is_pt_in_interval(const std::vector<double> & pt, const std::vector<double> & interval);

	// return true if a given point in a interval in 1D
	static bool is_pt_in_interval(const double pt, const std::vector<double> & interval);

	// return true if current element and given element has the same index in 1D in dim-th dimension
	bool is_same_elem_1D(const Element & elem, const int dim) const;

	// order of element in 1D
	static int index_to_order_elem(const int level, const int suppt);

	// order of element in multi dimension
	static std::vector<int> index_to_order_elem(const std::vector<int> & level, const std::vector<int> & suppt);
};

