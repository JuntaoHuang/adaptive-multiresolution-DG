#pragma once
#include "DGSolution.h"


// DGAapt is derived from DGSolution
class DGAdapt :
	public DGSolution
{
public:
	DGAdapt(const bool sparse_, const int level_init_, const int NMAX_, AllBasis<AlptBasis> & all_bas_, AllBasis<LagrBasis> & all_bas_Lag_, AllBasis<HermBasis> & all_bas_Her_, Hash & hash_, const double eps_, const double eta_, const bool is_find_ptr_alpt_, const bool is_find_ptr_intp_, const bool is_find_ptr_general_ = false);
	~DGAdapt() {};

	/// adaptive initialization for seperable initial value
	/// a function of separable form f(x1,x2,...,xdim) = g_1(x1) * g_2(x2) * ... * g_dim(xdim)
	/// input is a functional: (x, d) -> g_d(x)
	void init_separable_scalar(std::function<double(double, int)> scalar_func);
	void init_separable_system(std::vector<std::function<double(double, int)>> vector_func);
	
	/// adaptive initialization for summation of seperable initial value
	/// given a initial function, which is summation of separable functions
	/// size of sum_func is num of separable functions
	void init_separable_scalar_sum(std::vector<std::function<double(double, int)>> sum_scalar_func);
	void init_separable_system_sum(std::vector<std::vector<std::function<double(double, int)>>> sum_vector_func);

	// refine based on reference solution generated by Euler forward
	void refine();

	// coarsen
	void coarsen();

	// update elements with artificial viscosity
	void update_viscosity_element(const double shock_kappa);

	static std::vector<int> indicator_var_adapt;

protected:
	const double eps;
	const double eta;

	std::unordered_map<int, Element*> leaf;		// leaf (element which do not have its all children in DGSolution)
	std::unordered_map<int, Element*> leaf_zero_child;	// leaf (element which have zero children in DGSolution)

	// set new_add variable in all elements to be false
	void set_all_new_add_false();

	// after adding or deleting, check no holes in solution
	void check_hole();

	// update leaf
	// if num of existing children is not the same as num of total children, then add it to leaf element
	void update_leaf();


	// add a given element into DG solution
	void add_elem(Element & elem);
	
	// delete a given element into DG solution
	void del_elem(Element & elem);

	// return collections of index of parent/children of given index
	std::set<std::array<std::vector<int>,2>> index_all_par( const std::vector<int> & lev_n, const std::vector<int> & sup_j);
	std::set<std::array<std::vector<int>,2>> index_all_chd( const std::vector<int> & lev_n, const std::vector<int> & sup_j);

private:
	
	const bool is_find_ptr_alpt;
	const bool is_find_ptr_intp;
	const bool is_find_ptr_general;

	// check that total num of children and parents in all elements are equal
	// this function will only be used for debug
	bool check_total_num_chd_par_equal() const;

	// initial number and pointers of parents and children
	void init_chd_par();

	// recursively refine for seperable initial value
	void refine_init_separable_scalar(const VecMultiD<double> & coeff);
	void refine_init_separable_system(const std::vector<VecMultiD<double>> & coeff);

	// recursively refine for summation of seperable initial value 
	void refine_init_separable_scalar_sum(const std::vector<VecMultiD<double>> & coeff);
	void refine_init_separable_system_sum(const std::vector<std::vector<VecMultiD<double>>> & coeff);

	// coarsen based on leaf with 0 child
	void coarsen_no_leaf();

	// update leaf with 0 child
	// if num of existing children is not the same as num of total children, then add it to leaf element
	void update_leaf_zero_child();

	void check_hole_init_separable_scalar(const VecMultiD<double> & coeff);
	void check_hole_init_separable_system(const std::vector<VecMultiD<double>> & coeff);

	void check_hole_init_separable_scalar_sum(const std::vector<VecMultiD<double>> & coeff);
	void check_hole_init_separable_system_sum(const std::vector<std::vector<VecMultiD<double>>> & coeff);

	// return L2 norm of functions in an element
 	double indicator_norm(const Element & elem) const;

	// return number of all parents (no matter if already in dg) of element with given mesh level n
	int num_all_par(const std::vector<int> & lev_n) const;

	// return number of all children (no matter if already in dg) of element with given mesh level n
	int num_all_chd(const std::vector<int> & lev_n) const;
};

