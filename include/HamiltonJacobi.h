#pragma once
#include "BilinearForm.h"


class HamiltonJacobiLDG 
{
public:
	HamiltonJacobiLDG(std::vector<HyperbolicAlpt> & linearvec, const int dim) :
		dgoperator_ptr(&linearvec), col(linearvec[0].col), row(linearvec[0].row), DIM(dim) {}

	~HamiltonJacobiLDG() {};

	void HJ_apply_operator_to_vector(const Eigen::VectorXd &phicoe);

	const std::vector<HyperbolicAlpt>* dgoperator_ptr; 


	Eigen::VectorXd get_phi_eigenvec() const;
	void copy_eigenvec_to_phi(const Eigen::VectorXd &phicoe);

	//Eigen::VectorXd phicoe;
private:
	int const col;
	int const row;
	int const DIM;


};
