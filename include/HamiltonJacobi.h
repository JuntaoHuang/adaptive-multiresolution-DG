#pragma once
#include "BilinearForm.h"

template<typename HJBilinear>
class HamiltonJacobiLDG 
{
public:
	HamiltonJacobiLDG() : dgoperator_ptr(nullptr), col(0), row(0), DIM(0) {}
	HamiltonJacobiLDG(std::vector<HJBilinear> & linearvec, const int dim) :
		dgoperator_ptr(&linearvec), col(linearvec[0].col), row(linearvec[0].row), DIM(dim) {}
	//HamiltonJacobiLDG(std::vector<HJOutflowAlpt> & linearvec, const int dim) :
	//	dgoperator_ptr(&linearvec), col(linearvec[0].col), row(linearvec[0].row), DIM(dim) {}
	~HamiltonJacobiLDG() {};

	void HJ_apply_operator_to_vector(const Eigen::VectorXd &phicoe);

	const std::vector<HJBilinear>* dgoperator_ptr;


	Eigen::VectorXd get_phi_eigenvec() const;
	void copy_eigenvec_to_phi(const Eigen::VectorXd &phicoe);

	//Eigen::VectorXd phicoe;
private:
	int const col;
	int const row;
	int const DIM;

};




//class HamiltonJacobiLDGOutFlow:
//	public HamiltonJacobiLDG
//{
//public:
//	HamiltonJacobiLDGOutFlow(std::vector<HJOutflowAlpt> & linearvec, const int dim) :
//		dgoperator_ptr(&linearvec), col(linearvec[0].col), row(linearvec[0].row), DIM(dim) {}
//	~HamiltonJacobiLDGOutFlow() {};
//
//	//void HJ_apply_operator_to_vector(const Eigen::VectorXd &phicoe);
//
//
//
//	//Eigen::VectorXd get_phi_eigenvec() const;
//	//void copy_eigenvec_to_phi(const Eigen::VectorXd &phicoe);
//
//	//Eigen::VectorXd phicoe;
//private:
//	int const col;
//	int const row;
//	int const DIM;
//	const std::vector<HJOutflowAlpt>* dgoperator_ptr;
//
//};


template <class HJBilinear>
Eigen::VectorXd HamiltonJacobiLDG<HJBilinear>::get_phi_eigenvec() const
{

	Eigen::VectorXd phicoe;
	phicoe.resize(col);
	int order_alpt_basis_in_phi = 0;
	for (auto & iter : dgoperator_ptr->at(0).dgsolution_ptr->dg)
	{
		for (size_t num_basis = 0; num_basis < iter.second.size_alpt(); num_basis++)
		{
			const std::vector<int> & order_local_basis = iter.second.order_local_alpt[num_basis];
			phicoe(order_alpt_basis_in_phi) = iter.second.ucoe_alpt[0].at(order_local_basis);
			order_alpt_basis_in_phi++;
		}
	}
	return phicoe;

}

template <class HJBilinear>
void HamiltonJacobiLDG<HJBilinear>::copy_eigenvec_to_phi(const Eigen::VectorXd &phicoe)
{

	int order_alpt_basis_in_phi = 0;
	for (auto & iter : dgoperator_ptr->at(0).dgsolution_ptr->dg)
	{
		for (size_t num_basis = 0; num_basis < iter.second.size_alpt(); num_basis++)
		{
			const std::vector<int> & order_local_basis = iter.second.order_local_alpt[num_basis];
			iter.second.ucoe_alpt[0].at(order_local_basis) = phicoe(order_alpt_basis_in_phi);
			order_alpt_basis_in_phi++;
		}
	}


}

template <class HJBilinear>
void HamiltonJacobiLDG<HJBilinear>::HJ_apply_operator_to_vector(const Eigen::VectorXd &phicoe) {

	for (size_t ind_var = 1; ind_var < 2 * DIM + 1; ++ind_var)
	{
		Eigen::VectorXd gradcoe;
		gradcoe.resize(col);


		gradcoe = dgoperator_ptr->at(ind_var - 1).mat * phicoe; // only 2 * DIM variables

		int order_alpt_basis_in_grad = 0;
		for (auto & iter : dgoperator_ptr->at(0).dgsolution_ptr->dg)
		{
			for (size_t num_basis = 0; num_basis < iter.second.size_alpt(); num_basis++)
			{
				const std::vector<int> & order_local_basis = iter.second.order_local_alpt[num_basis];
				iter.second.ucoe_alpt[ind_var].at(order_local_basis) = gradcoe(order_alpt_basis_in_grad++);

			}
		}


	}



}
