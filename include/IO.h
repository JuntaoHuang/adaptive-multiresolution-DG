#pragma once
#include "DGSolution.h"

class IO
{
public:
    IO(DGSolution & dgsolution);
    ~IO() {};

    // output numerical solution
    // format: x1, x2, ..., xdim, u
    // first loop over x1, then inside loop over x2, ..., last loop over xdim
    void output_num(const std::string & file_name) const;

    // output numerical solution for schrodinger equation
    // format: x1, x2, ..., xdim, |u|
    // first loop over x1, then inside loop over x2, ..., last loop over xdim
    void output_num_schrodinger(const std::string & file_name) const;

    // output numerical solution in 3D cut in 2D
    void output_num_cut_2D(const std::string & file_name, const double cut_x, const int cut_dim) const;

    // output numerical solution in 3D cut in 1D
    void output_num_3D_cut_1D(const std::string & file_name, const std::vector<double> & cut_x, const std::vector<int> & cut_dim, const int vec_index) const;

    // output numerical and exact solution in 2D
    // format: x1, x2, ..., xdim, u_num, u_exa
    // first loop over x1, then inside loop over x2, ..., last loop over xdim
    void output_num_exa(const std::string & file_name, std::function<double(std::vector<double>)> exact_solution) const;

    // output numerical and exact solution, error
    // format: x1, x2, ..., xdim, u_num, u_exa
    // format: x1, x2, ..., xdim, u_error
    // first loop over x1, then inside loop over x2, ..., last loop over xdim
    void output_num_exa_err_system(const std::string & file_name, const std::string & file_name_err, 
                                   std::vector< std::function<double(std::vector<double>)> > exact_solution) const;

    // output center of element into file
    // format: x_0, x_1, ..., x_(dim-1)
    // each line denote center of support of the element
    void output_element_center(const std::string & file_name) const;

    // write support interval of all the active elements into file
    // format: each line denote a interval in multi dimension
    // x0_l, x0_r, x1_l, x1_r, ..., x(dim-1)_l, x(dim-1)_r
    void output_element_support(const std::string & file_name) const;

    // write support interval and also its mesh level of all the active elements into file
    void output_element_level_support(const std::string & file_name) const;

    // write shock support interval into file 
    // format: each line denote a interval in multi dimension
    // x0_l, x0_r, x1_l, x1_r, ..., x(dim-1)_l, x(dim-1)_r
    void output_shock_support(const std::string & file_name) const;

    void write_error_time(const double tn, const std::vector<double> & err, const int init, const std::string file_name) const;

    /**
     * @brief write error (between numerical solution and exact solution) into file
     */
    void write_error(const int NMAX, const std::vector<double> & err, const std::string file_name) const;

    // err is a vector of size: VEC_NUM * 3, denote L1, L2, Linf error for each unknown variable
    void write_error_system(const int NMAX, const std::vector<std::vector<double> > & err, std::string file_name) const;

    // err is a vector of size: VEC_NUM * 3, denote L1, L2, Linf error for each unknown variable
    void write_error_eps_dof(const double eps, const int dof, const std::vector<std::vector<double>> & err, const std::string file_name) const;
    // overload for scalar case
    void write_error_eps_dof(const double eps, const int dof, const std::vector<double> & err, const std::string file_name) const;

    // output flux function of collection of all Lagrange basis (with coefficients fucoe_intp)
    // this function is only used for debug
    void output_flux_lagrange(const std::string & file_name, const int unknown_var_index, const int dim, const int NMAX) const;

    // check solution is symmetric along a give dimension
    // return largest difference between symmetric solution
    double check_symmetry(const int dim, const int num_pt);

	void output_num_2D_controls(const std::string & file_name) const;
private:
    DGSolution* const dgsolution_ptr;

    const int max_mesh;
    const int num_grid_point;
    const double dx;

    std::vector<double> x_1D;

    // output numerical solution in 1D
    // format: x, u
    void output_num_1D(const std::string & file_name) const;

    // output numerical solution in 1D for schrodinger eq.
    // format: x, |u|
    void output_num_1D_schrodinger(const std::string & file_name) const;

    // output numerical solution in 2D
    // format: x, y, u
    // first loop over x, then inside loop over y
    void output_num_2D(const std::string & file_name) const;

    // output numerical solution in 2D for schrodinger eq.
    // format: x, y, |u|
    // first loop over x, then inside loop over y
    void output_num_2D_schrodinger(const std::string & file_name) const;


    // output numerical solution in 3D
    // format: x, y, z, u
    // first loop over x, then loop over y and then z
    void output_num_3D(const std::string & file_name) const;

    // output numerical and exact solution in 1D
    // format: x, u_num, u_exa
    void output_num_exa_1D(const std::string & file_name, std::function<double(std::vector<double>)> exact_solution) const;

    // output numerical and exact solution, error in 1D
    // format: x, u_num, u_exa
    // format: x, u_error
    void output_num_exa_err_system_1D(const std::string & file_name, const std::string & file_name_err, 
                                      std::vector< std::function<double(std::vector<double>)> > exact_solution) const;

    // output numerical and exact solution in 2D
    // format: x, y, u_num, u_exa
    // first loop over x, then inside loop over y
    void output_num_exa_2D(const std::string & file_name, std::function<double(std::vector<double>)> exact_solution) const;
};