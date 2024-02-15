#include "ODESolver.h"
#include "HamiltonJacobi.h"

ODESolver::ODESolver(BilinearForm & bilinearform):
	dgoperator_ptr(&bilinearform), dgsolution_ptr(dgoperator_ptr->dgsolution_ptr),
	dof(dgsolution_ptr->get_dof())
{
	ucoe.resize(dof);
	rhs.resize(dof);

	ucoe.setZero();
	rhs.setZero();	
}

ODESolver::ODESolver(DGSolution & dg):
	dgoperator_ptr(nullptr), dgsolution_ptr(& dg), dof(dgsolution_ptr->get_dof())
{
	ucoe.resize(dof);
	rhs.resize(dof);	

	ucoe.setZero();
	rhs.setZero();
}

void ODESolver::ucoe_to_eigenvec()
{
	int order_alpt_basis_in_dgmap = 0;
	for (auto & iter : dgsolution_ptr->dg)
	{
		for (size_t num_vec = 0; num_vec < dgsolution_ptr->VEC_NUM; num_vec++)
		{
			for (size_t num_basis = 0; num_basis < iter.second.size_alpt(); num_basis++)
			{
				const std::vector<int> & order_local_basis = iter.second.order_local_alpt[num_basis];
				ucoe(order_alpt_basis_in_dgmap) = iter.second.ucoe_alpt[num_vec].at(order_local_basis);
				order_alpt_basis_in_dgmap++;
			}
		}
	}
}




void ODESolver::eigenvec_to_ucoe() const
{
	int order_alpt_basis_in_dgmap = 0;
	for (auto & iter : dgsolution_ptr->dg)
	{
		for (size_t num_vec = 0; num_vec < dgsolution_ptr->VEC_NUM; num_vec++)
		{
			for (size_t num_basis = 0; num_basis < iter.second.size_alpt(); num_basis++)
			{
				const std::vector<int> & order_local_basis = iter.second.order_local_alpt[num_basis];
				iter.second.ucoe_alpt[num_vec].at(order_local_basis) = ucoe(order_alpt_basis_in_dgmap);
				order_alpt_basis_in_dgmap++;
			}
		}
	}	
}

void ODESolver::rhs_to_eigenvec()
{
	int order_alpt_basis_in_dgmap = 0;
	for (auto & iter : dgsolution_ptr->dg)
	{
		for (size_t num_vec = 0; num_vec < dgsolution_ptr->VEC_NUM; num_vec++)
		{
			for (size_t num_basis = 0; num_basis < iter.second.size_alpt(); num_basis++)
			{
				const std::vector<int> & order_local_basis = iter.second.order_local_alpt[num_basis];
				rhs(order_alpt_basis_in_dgmap) = iter.second.rhs[num_vec].at(order_local_basis);
				order_alpt_basis_in_dgmap++;
			}
		}
	}
}

void ODESolver::rhs_to_eigenvec(std::string prob)
{
	assert(prob == "HJ");
	
	int order_alpt_basis_in_dgmap = 0;
	for (auto & iter : dgsolution_ptr->dg)
	{
		for (size_t num_basis = 0; num_basis < iter.second.size_alpt(); num_basis++)
		{
			const std::vector<int> & order_local_basis = iter.second.order_local_alpt[num_basis];
			rhs(order_alpt_basis_in_dgmap) = iter.second.rhs[0].at(order_local_basis);
			order_alpt_basis_in_dgmap++;
		}
	}
}

void ODESolver::eigenvec_to_rhs() const
{
	int order_alpt_basis_in_dgmap = 0;
	for (auto & iter : dgsolution_ptr->dg)
	{
		for (size_t num_vec = 0; num_vec < dgsolution_ptr->VEC_NUM; num_vec++)
		{
			for (size_t num_basis = 0; num_basis < iter.second.size_alpt(); num_basis++)
			{
				const std::vector<int> & order_local_basis = iter.second.order_local_alpt[num_basis];
				iter.second.rhs[num_vec].at(order_local_basis) = rhs(order_alpt_basis_in_dgmap);
				order_alpt_basis_in_dgmap++;
			}
		}
	}	
}

void ODESolver::add_rhs_to_eigenvec()
{
	int order_alpt_basis_in_dgmap = 0;
	for (auto & iter : dgsolution_ptr->dg)
	{
		for (size_t num_vec = 0; num_vec < dgsolution_ptr->VEC_NUM; num_vec++)
		{
			for (size_t num_basis = 0; num_basis < iter.second.size_alpt(); num_basis++)
			{
				const std::vector<int> & order_local_basis = iter.second.order_local_alpt[num_basis];
				rhs(order_alpt_basis_in_dgmap) += iter.second.rhs[num_vec].at(order_local_basis);
				order_alpt_basis_in_dgmap++;
			}
		}
	}
}

void ODESolver::add_eigenvec_to_rhs() const
{
	int order_alpt_basis_in_dgmap = 0;
	for (auto & iter : dgsolution_ptr->dg)
	{
		for (size_t num_vec = 0; num_vec < dgsolution_ptr->VEC_NUM; num_vec++)
		{
			for (size_t num_basis = 0; num_basis < iter.second.size_alpt(); num_basis++)
			{
				const std::vector<int> & order_local_basis = iter.second.order_local_alpt[num_basis];
				iter.second.rhs[num_vec].at(order_local_basis) += rhs(order_alpt_basis_in_dgmap);
				order_alpt_basis_in_dgmap++;
			}
		}
	}
}

void ODESolver::fucoe_to_eigenvec(const int dim)
{
	// size of fucoe is the same as num of interpolation basis
	const int col = dgsolution_ptr->size_basis_intp() * dgsolution_ptr->VEC_NUM;
	fucoe.resize(col);

	int order_intp_basis_in_dgmap = 0;
	for (auto & iter : dgsolution_ptr->dg)
	{
		for (size_t num_vec = 0; num_vec < dgsolution_ptr->VEC_NUM; num_vec++)
		{
			for (size_t num_basis = 0; num_basis < iter.second.size_intp(); num_basis++)
			{
				const std::vector<int> & order_local_basis = iter.second.order_local_intp[num_basis];
				fucoe(order_intp_basis_in_dgmap) = iter.second.fucoe_intp[num_vec][dim].at(order_local_basis);
				order_intp_basis_in_dgmap++;
			}
		}
	}	
}

 void ODESolver::fucoe_to_eigenvec_HJ()
{
	// size of fucoe is the same as num of interpolation basis
	const int col = dgsolution_ptr->size_basis_intp();
	fucoe.resize(col);

	int order_intp_basis_in_dgmap = 0;
	for (auto & iter : dgsolution_ptr->dg)
	{
		for (size_t num_basis = 0; num_basis < iter.second.size_intp(); num_basis++)
		{
			const std::vector<int> & order_local_basis = iter.second.order_local_intp[num_basis];
			fucoe(order_intp_basis_in_dgmap) = iter.second.fucoe_intp[0][0].at(order_local_basis);
			order_intp_basis_in_dgmap++;
		}

	}
 }

 void ExplicitRK::ucoe_tn_to_eigenvec()
 {
	 int order_alpt_basis_in_dgmap = 0;
	 for (auto & iter : dgsolution_ptr->dg)
	 {
		 for (size_t num_vec = 0; num_vec < dgsolution_ptr->VEC_NUM; num_vec++)
		 {
			 for (size_t num_basis = 0; num_basis < iter.second.size_alpt(); num_basis++)
			 {
				 const std::vector<int> & order_local_basis = iter.second.order_local_alpt[num_basis];
				 ucoe_tn(order_alpt_basis_in_dgmap) = iter.second.ucoe_alpt[num_vec].at(order_local_basis);
				 order_alpt_basis_in_dgmap++;
			 }
		 }
	 }
 }

 void ForwardEuler::step_rk()
 {
	 dgoperator_ptr->multi(ucoe, rhs);
	 ucoe = ucoe_tn + dt * rhs;
 }

 void ForwardEuler::step_stage(const int stage)
 {
	 assert(stage == 0);
	 ucoe = ucoe_tn + dt * rhs;
 }

 void RK2SSP::step_rk()
 {
	 dgoperator_ptr->multi(ucoe, rhs);
	 u1 = ucoe + dt * rhs;

	 dgoperator_ptr->multi(u1, rhs);
	 ucoe = 0.5 * ucoe + 0.5 * (u1 + dt * rhs);
 }

 void RK2SSP::step_stage(const int stage)
 {
	 assert((stage >= 0) && (stage <= 1));

	 if (stage == 0)
	 {
		 ucoe = ucoe_tn + dt * rhs;
	 }
	 else if (stage == 1)
	 {
		 ucoe = 0.5 * ucoe_tn + 0.5 * (ucoe + dt * rhs);
	 }
 }

void RK2Midpoint::step_rk()
{
	dgoperator_ptr->multi(ucoe, rhs);
	u1 = ucoe_tn + 0.5 * dt * rhs;

	dgoperator_ptr->multi(u1, rhs);
	ucoe = ucoe_tn + dt * rhs;
}

void RK2Midpoint::step_stage(const int stage)
{
	assert((stage >= 0) && (stage <= 1));

	if (stage == 0)
	{
		ucoe = ucoe_tn + 0.5 * dt * rhs;
	}
	else if (stage == 1)
	{
		ucoe = ucoe_tn + dt * rhs;
	}
}

 void RK3SSP::step_rk()
 {
	 dgoperator_ptr->multi(ucoe, rhs);
	 u1 = ucoe + dt * rhs;

	 dgoperator_ptr->multi(u1, rhs);
	 u2 = 3. / 4. * ucoe + 1. / 4. * (u1 + dt * rhs);

	 dgoperator_ptr->multi(u2, rhs);
	 ucoe = 1. / 3. * ucoe + 2. / 3. * (u2 + dt * rhs);
 }

void RK3SSP::step_rk_source(const Eigen::VectorXd & source)
{   
    dgoperator_ptr->multi(ucoe, rhs);
    u1 = ucoe + dt * (rhs + source);

    dgoperator_ptr->multi(u1, rhs);
    u2 = 3./4. * ucoe + 1./4. * (u1 + dt * (rhs + source));

    dgoperator_ptr->multi(u2, rhs);
    ucoe = 1./3. * ucoe + 2./3. * (u2 + dt * (rhs + source));
}

 void RK3SSP::step_stage(const int stage)
 {
	 assert((stage >= 0) && (stage <= 2));

	 if (stage == 0)
	 {
		 ucoe = ucoe_tn + dt * rhs;
	 }
	 else if (stage == 1)
	 {
		 ucoe = 3. / 4. * ucoe_tn + 1. / 4. * (ucoe + dt * rhs);
	 }
	 else if (stage == 2)
	 {
		 ucoe = 1. / 3. * ucoe_tn + 2. / 3. * (ucoe + dt * rhs);
	 }
 }


void RK3HeunLinear::step_rk()
{

}

void RK3HeunLinear::step_rk_source(const Eigen::VectorXd & source)
{
	
}

void RK3HeunLinear::step_stage(const int stage)
{
	assert((stage >= 0) && (stage <= 2));

	if (stage == 0)
	{
		ucoe = ucoe_tn + 1. / 3. * dt * rhs;
	}
	else if (stage == 1)
	{
		ucoe = ucoe_tn + 1. / 2. * dt * rhs;
	}
	else if (stage == 2)
	{
		ucoe = ucoe_tn + dt * rhs;
	}
}


 void RK4::step_rk()
 {
	 //   0  |
	 //  1/2 | 1/2
	 //  1/2 |  0   1/2
	 //   1  |  0    0    1
	 // -----+-------------------
	 //      | 1/6  1/3  1/3  1/6

	 std::cout << "to be completed in RK4::step" << std::endl;
	 exit(1);
 }

void RKODE2nd::init()
{
	ucoe_to_eigenvec(); 
	ucoe_tn = ucoe;
	ucoe_ut_to_eigenvec();
	vcoe_tn = vcoe;
}

void RKODE2nd::final() 
{ 
	eigenvec_to_ucoe();
	eigenvec_to_ucoe_ut();
	rhs.setZero();
};

void RKODE2nd::ucoe_ut_to_eigenvec()
{
	int order_alpt_basis_in_dgmap = 0;
	for (auto & iter : dgsolution_ptr->dg)
	{
		for (size_t num_vec = 0; num_vec < dgsolution_ptr->VEC_NUM; num_vec++)
		{
			for (size_t num_basis = 0; num_basis < iter.second.size_alpt(); num_basis++)
			{
				const std::vector<int> & order_local_basis = iter.second.order_local_alpt[num_basis];
				vcoe(order_alpt_basis_in_dgmap) = iter.second.ucoe_ut[num_vec].at(order_local_basis);
				order_alpt_basis_in_dgmap++;
			}
		}
	}
}

void RKODE2nd::eigenvec_to_ucoe_ut()
{
	int order_alpt_basis_in_dgmap = 0;
	for (auto & iter : dgsolution_ptr->dg)
	{
		for (size_t num_vec = 0; num_vec < dgsolution_ptr->VEC_NUM; num_vec++)
		{
			for (size_t num_basis = 0; num_basis < iter.second.size_alpt(); num_basis++)
			{
				const std::vector<int> & order_local_basis = iter.second.order_local_alpt[num_basis];
				iter.second.ucoe_ut[num_vec].at(order_local_basis) = vcoe(order_alpt_basis_in_dgmap);
				order_alpt_basis_in_dgmap++;
			}
		}
	}
}

void EulerODE2nd::step_rk()
{
	dgoperator_ptr->multi(ucoe, rhs);
	ucoe = ucoe + dt * vcoe;
	vcoe = vcoe + dt * rhs;
}

void EulerODE2nd::step_rk_source(std::function<Eigen::VectorXd(double)> source_func, double time_tn)
{
	dgoperator_ptr->multi(ucoe, rhs);
	ucoe = ucoe + dt * vcoe;
	vcoe = vcoe + dt * (rhs + source_func(time_tn));
}

void EulerODE2nd::step_stage(const int stage)
{
	assert(stage == 0);
	
	ucoe = ucoe + dt * vcoe;
	vcoe = vcoe + dt * rhs;
}

void RK2ODE2nd::step_rk()
{
	dgoperator_ptr->multi(ucoe, rhs);
	Eigen::VectorXd u1 = ucoe + dt * vcoe;
	Eigen::VectorXd v1 = vcoe + dt * rhs;

	dgoperator_ptr->multi(u1, rhs);
	ucoe = 1. / 2. * ucoe + 1. / 2. * (u1 + dt * v1);
	vcoe = 1. / 2. * vcoe + 1. / 2. * (v1 + dt * rhs);
}

void RK2ODE2nd::step_rk_source(std::function<Eigen::VectorXd(double)> source_func, double time_tn)
{
	dgoperator_ptr->multi(ucoe, rhs);
	Eigen::VectorXd u1 = ucoe + dt * vcoe;
	Eigen::VectorXd v1 = vcoe + dt * (rhs + source_func(time_tn));

	dgoperator_ptr->multi(u1, rhs);
	ucoe = 1. / 2. * ucoe + 1. / 2. * (u1 + dt * v1);
	vcoe = 1. / 2. * vcoe + 1. / 2. * (v1 + dt * (rhs + source_func(time_tn+dt)));
}

void RK2ODE2nd::step_stage(const int stage)
{
	assert((stage >= 0) && (stage <= 1));

	if (stage == 0)
	{
		ucoe = ucoe_tn + dt * vcoe;
		vcoe = vcoe_tn + dt * rhs;
	}
	else if (stage == 1)
	{
		ucoe = 1. / 2. * ucoe_tn + 1. / 2. * (ucoe + dt * vcoe);
		vcoe = 1. / 2. * vcoe_tn + 1. / 2. * (vcoe + dt * rhs);
	}
}

void RK2ODE2nd::step_stage_source(const int stage, std::function<Eigen::VectorXd(double)> source_func, double time_tn)
{
	assert((stage >= 0) && (stage <= 1));

	if (stage == 0)
	{
		ucoe = ucoe_tn + dt * vcoe;
		vcoe = vcoe_tn + dt * (rhs + source_func(time_tn));
	}
	else if (stage == 1)
	{
		ucoe = 1. / 2. * ucoe_tn + 1. / 2. * (ucoe + dt * vcoe);
		vcoe = 1. / 2. * vcoe_tn + 1. / 2. * (vcoe + dt * (rhs + source_func(time_tn+dt)));
	}
}

void RK3ODE2nd::step_rk()
{
	dgoperator_ptr->multi(ucoe, rhs);
	Eigen::VectorXd u1 = ucoe + dt * vcoe;
	Eigen::VectorXd v1 = vcoe + dt * rhs;

	dgoperator_ptr->multi(u1, rhs);
	Eigen::VectorXd u2 = 3. / 4. * ucoe + 1. / 4. * (u1 + dt * v1);
	Eigen::VectorXd v2 = 3. / 4. * vcoe + 1. / 4. * (v1 + dt * rhs);

	dgoperator_ptr->multi(u2, rhs);
	ucoe = 1. / 3. * ucoe + 2. / 3. * (u2 + dt * v2);
	vcoe = 1. / 3. * vcoe + 2. / 3. * (v2 + dt * rhs);
}

void RK3ODE2nd::step_rk_source(std::function<Eigen::VectorXd(double)> source_func, double time_tn)
{
	dgoperator_ptr->multi(ucoe, rhs);
	Eigen::VectorXd u1 = ucoe + dt * vcoe;
	Eigen::VectorXd v1 = vcoe + dt * (rhs + source_func(time_tn));

	dgoperator_ptr->multi(u1, rhs);
	Eigen::VectorXd u2 = 3. / 4. * ucoe + 1. / 4. * (u1 + dt * v1);
	Eigen::VectorXd v2 = 3. / 4. * vcoe + 1. / 4. * (v1 + dt * (rhs + source_func(time_tn + dt)));

	dgoperator_ptr->multi(u2, rhs);
	ucoe = 1. / 3. * ucoe + 2. / 3. * (u2 + dt * v2);
	vcoe = 1. / 3. * vcoe + 2. / 3. * (v2 + dt * (rhs + source_func(time_tn + dt/2.)));
}

void RK3ODE2nd::step_stage(const int stage)
{
	assert((stage >= 0) && (stage <= 2));

	if (stage == 0)
	{
		ucoe = ucoe_tn + dt * vcoe;
		vcoe = vcoe_tn + dt * rhs;
	}
	else if (stage == 1)
	{
		ucoe = 3. / 4. * ucoe_tn + 1. / 4. * (ucoe + dt * vcoe);
		vcoe = 3. / 4. * vcoe_tn + 1. / 4. * (vcoe + dt * rhs);
	}
	else if (stage == 2)
	{
		ucoe = 1. / 3. * ucoe_tn + 2. / 3. * (ucoe + dt * vcoe);
		vcoe = 1. / 3. * vcoe_tn + 2. / 3. * (vcoe + dt * rhs);
	}
}

void RK3ODE2nd::step_stage_source(const int stage, std::function<Eigen::VectorXd(double)> source_func, double time_tn)
{
	assert((stage >= 0) && (stage <= 2));

	if (stage == 0)
	{
		ucoe = ucoe_tn + dt * vcoe;
		vcoe = vcoe_tn + dt * (rhs + source_func(time_tn));
	}
	else if (stage == 1)
	{
		ucoe = 3. / 4. * ucoe_tn + 1. / 4. * (ucoe + dt * vcoe);
		vcoe = 3. / 4. * vcoe_tn + 1. / 4. * (vcoe + dt * (rhs + source_func(time_tn + dt)));
	}
	else if (stage == 2)
	{
		ucoe = 1. / 3. * ucoe_tn + 2. / 3. * (ucoe + dt * vcoe);
		vcoe = 1. / 3. * vcoe_tn + 2. / 3. * (vcoe + dt * (rhs + source_func(time_tn + dt/2.)));
	}
}

void RK4ODE2nd::step_rk()
{
	Eigen::VectorXd k1u = dt * vcoe;
	Eigen::VectorXd k1v = dt * (dgoperator_ptr->mat * ucoe);

	Eigen::VectorXd k2u = dt * (vcoe + k1v/2.);
	Eigen::VectorXd k2v = dt * (dgoperator_ptr->mat * (ucoe + k1u/2.));

	Eigen::VectorXd k3u = dt * (vcoe + k2v/2.);
	Eigen::VectorXd k3v = dt * (dgoperator_ptr->mat * (ucoe + k2u/2.));

	Eigen::VectorXd k4u = dt * (vcoe + k3v);
	Eigen::VectorXd k4v = dt * (dgoperator_ptr->mat * (ucoe + k3u));

	ucoe = ucoe + 1./6. * (k1u + 2*k2u + 2*k3u + k4u);
	vcoe = vcoe + 1./6. * (k1v + 2*k2v + 2*k3v + k4v);
}

void RK4ODE2nd::step_rk_source(std::function<Eigen::VectorXd(double)> source_func, double time_tn)
{
	Eigen::VectorXd k1u = dt * vcoe;
	Eigen::VectorXd k1v = dt * (dgoperator_ptr->mat * ucoe + source_func(time_tn));

	Eigen::VectorXd k2u = dt * (vcoe + k1v/2.);
	Eigen::VectorXd k2v = dt * (dgoperator_ptr->mat * (ucoe + k1u/2.) + source_func(time_tn + dt/2.));

	Eigen::VectorXd k3u = dt * (vcoe + k2v/2.);
	Eigen::VectorXd k3v = dt * (dgoperator_ptr->mat * (ucoe + k2u/2.) + source_func(time_tn + dt/2.));

	Eigen::VectorXd k4u = dt * (vcoe + k3v);
	Eigen::VectorXd k4v = dt * (dgoperator_ptr->mat * (ucoe + k3u) + source_func(time_tn + dt));

	ucoe = ucoe + 1./6. * (k1u + 2*k2u + 2*k3u + k4u);
	vcoe = vcoe + 1./6. * (k1v + 2*k2v + 2*k3v + k4v);
}

void RK4ODE2nd::step_stage(const int stage)
{
	assert((stage >= 0) && (stage <= 3));

	if (stage == 0)
	{
		k1u = dt * vcoe;
		k1v = dt * rhs;

		ucoe = ucoe_tn + k1u/2.;
		vcoe = vcoe_tn + k1v/2.;
	}
	else if (stage == 1)
	{
		k2u = dt * vcoe;
		k2v = dt * rhs;

		ucoe = ucoe_tn + k2u/2.;
		vcoe = vcoe_tn + k2v/2.;
	}
	else if (stage == 2)
	{
		k3u = dt * vcoe;
		k3v = dt * rhs;

		ucoe = ucoe_tn + k3u;
		vcoe = vcoe_tn + k3v;
	}
	else if (stage == 3)
	{
		k4u = dt * vcoe;
		k4v = dt * rhs;

		ucoe = ucoe_tn + 1./6. * (k1u + 2*k2u + 2*k3u + k4u);
		vcoe = vcoe_tn + 1./6. * (k1v + 2*k2v + 2*k3v + k4v);
	}
}

void RK4ODE2nd::step_stage_source(const int stage, std::function<Eigen::VectorXd(double)> source_func, double time_tn)
{
	assert((stage >= 0) && (stage <= 3));

	if (stage == 0)
	{
		k1u = dt * vcoe;
		k1v = dt * (rhs + source_func(time_tn));

		ucoe = ucoe_tn + k1u/2.;
		vcoe = vcoe_tn + k1v/2.;
	}
	else if (stage == 1)
	{
		k2u = dt * vcoe;
		k2v = dt * (rhs + source_func(time_tn+dt/2.));

		ucoe = ucoe_tn + k2u/2.;
		vcoe = vcoe_tn + k2v/2.;
	}
	else if (stage == 2)
	{
		k3u = dt * vcoe;
		k3v = dt * (rhs + source_func(time_tn+dt/2.));

		ucoe = ucoe_tn + k3u;
		vcoe = vcoe_tn + k3v;
	}
	else if (stage == 3)
	{
		k4u = dt * vcoe;
		k4v = dt * (rhs + source_func(time_tn+dt));

		ucoe = ucoe_tn + 1./6. * (k1u + 2*k2u + 2*k3u + k4u);
		vcoe = vcoe_tn + 1./6. * (k1v + 2*k2v + 2*k3v + k4v);
	}
}

void RK4ODE2nd::resize_ku_kv()
{
	k1u.resize(dof);
	k2u.resize(dof);
	k3u.resize(dof);
	k4u.resize(dof);

	k1v.resize(dof);
	k2v.resize(dof);
	k3v.resize(dof);
	k4v.resize(dof);
}
 // ----------------------------------------------
 // 
 // below is explicit multistep method
 // 
 // ----------------------------------------------
 void ExplicitMultiStep::ucoe_to_eigenvec_t_m1()
 {
	 int order_alpt_basis_in_dgmap = 0;
	 for (auto & iter : dgsolution_ptr->dg)
	 {
		 for (size_t num_vec = 0; num_vec < dgsolution_ptr->VEC_NUM; num_vec++)
		 {
			 for (size_t num_basis = 0; num_basis < iter.second.size_alpt(); num_basis++)
			 {
				 const std::vector<int> & order_local_basis = iter.second.order_local_alpt[num_basis];
				 ucoe_t_m1(order_alpt_basis_in_dgmap) = iter.second.ucoe_alpt_t_m1[num_vec].at(order_local_basis);
				 order_alpt_basis_in_dgmap++;
			 }
		 }
	 }
 }

 void ExplicitMultiStep::eigenvec_to_ucoe_t_m1() const
 {
	 int order_alpt_basis_in_dgmap = 0;
	 for (auto & iter : dgsolution_ptr->dg)
	 {
		 for (size_t num_vec = 0; num_vec < dgsolution_ptr->VEC_NUM; num_vec++)
		 {
			 for (size_t num_basis = 0; num_basis < iter.second.size_alpt(); num_basis++)
			 {
				 const std::vector<int> & order_local_basis = iter.second.order_local_alpt[num_basis];
				 iter.second.ucoe_alpt_t_m1[num_vec].at(order_local_basis) = ucoe_t_m1(order_alpt_basis_in_dgmap);
				 order_alpt_basis_in_dgmap++;
			 }
		 }
	 }
 }

 void ExplicitMultiStep::resize_variable()
 {
	 ucoe_t_m1.resize(dof);
 }

 void Newmard2nd::step_ms()
 {
	 // solution in time step (n+1)
	 Eigen::VectorXd ucoe_next = 2. * ucoe - ucoe_t_m1 + dt * dt * dgoperator_ptr->mat * ucoe;

	 ucoe_t_m1 = ucoe;
	 ucoe = ucoe_next;
 }

 void Newmard2nd::step_ms(const Eigen::VectorXd & vec_b)
 {
	 // solution in time step (n+1)
	 Eigen::VectorXd ucoe_next = 2. * ucoe - ucoe_t_m1 + dt * dt * (dgoperator_ptr->mat * ucoe + vec_b);

	 ucoe_t_m1 = ucoe;
	 ucoe = ucoe_next;
 }

 void Newmard2nd::step_ms_rhs()
 {
	 Eigen::VectorXd ucoe_next = 2. * ucoe - ucoe_t_m1 + dt * dt * rhs;

	 ucoe_t_m1 = ucoe;
	 ucoe = ucoe_next;
 }

 void Newmard2nd::step_ms_rhs(const Eigen::VectorXd & vec_b)
 {
	 Eigen::VectorXd ucoe_next = 2. * ucoe - ucoe_t_m1 + dt * dt * (rhs + vec_b);

	 ucoe_t_m1 = ucoe;
	 ucoe = ucoe_next;
 }

 void Newmard4th::step_ms()
 {
	 // A * u
	 Eigen::VectorXd Mv = dgoperator_ptr->mat * ucoe;

	 // A * A * u
	 Eigen::VectorXd M2v = dgoperator_ptr->mat * Mv;

	 // solution in time step (n+1)
	 Eigen::VectorXd ucoe_next = 2. * ucoe - ucoe_t_m1 + dt * dt * (Mv + (dt*dt / 12.) * M2v);

	 ucoe_t_m1 = ucoe;
	 ucoe = ucoe_next;
 }

 void Newmard4th::step_ms(const Eigen::VectorXd & vec_b, const Eigen::VectorXd & vec_b_2nd_derivative)
 {
	 // A * u
	 Eigen::VectorXd Mv = dgoperator_ptr->mat * ucoe;

	 // A * A * u
	 Eigen::VectorXd M2v = dgoperator_ptr->mat * Mv;

	 Eigen::VectorXd ucoe_next = 2. * ucoe - ucoe_t_m1 + dt * dt * (Mv + (dt*dt / 12.) * M2v + vec_b + dt * dt / 12. * (dgoperator_ptr->mat * vec_b + vec_b_2nd_derivative));

	 ucoe_t_m1 = ucoe;
	 ucoe = ucoe_next;
 }

 void Newmard4th::step_ms_rhs()
 {
	 Eigen::VectorXd ucoe_next = 2. * ucoe - ucoe_t_m1 + dt * dt * (rhs_Au + (dt*dt / 12.) * rhs_A2u);

	 ucoe_t_m1 = ucoe;
	 ucoe = ucoe_next;
 }


 void Newmard4th::step_ms_rhs(const Eigen::VectorXd & vec_b, const Eigen::VectorXd & vec_b_2nd_derivative)
 {
	 Eigen::VectorXd ucoe_next = 2. * ucoe - ucoe_t_m1 + dt * dt * (rhs_Au + (dt*dt / 12.) * rhs_A2u + vec_b + dt * dt / 12. * (rhs_Ab + vec_b_2nd_derivative));

	 ucoe_t_m1 = ucoe;
	 ucoe = ucoe_next;
 }

 void Newmard4th::swap_vec(Eigen::VectorXd & vec_a, Eigen::VectorXd & vec_b)
 {
	 assert(vec_a.size() == vec_b.size());

	 Eigen::VectorXd tmp = vec_a;
	 vec_a = vec_b;
	 vec_b = tmp;
 }

 void IMEX::rhs_explicit_to_eigenvec()
 {
	 int order_alpt_basis_in_dgmap = 0;
	 for (auto & iter : dgsolution_ptr->dg)
	 {
		 for (size_t num_vec = 0; num_vec < dgsolution_ptr->VEC_NUM; num_vec++)
		 {
			 for (size_t num_basis = 0; num_basis < iter.second.size_alpt(); num_basis++)
			 {
				 const std::vector<int> & order_local_basis = iter.second.order_local_alpt[num_basis];
				 rhs_explicit(order_alpt_basis_in_dgmap) = iter.second.rhs[num_vec].at(order_local_basis);
				 order_alpt_basis_in_dgmap++;
			 }
		 }
	 }
 }

 void IMEX::eigenvec_to_rhs_explicit() const
 {
	 int order_alpt_basis_in_dgmap = 0;
	 for (auto & iter : dgsolution_ptr->dg)
	 {
		 for (size_t num_vec = 0; num_vec < dgsolution_ptr->VEC_NUM; num_vec++)
		 {
			 for (size_t num_basis = 0; num_basis < iter.second.size_alpt(); num_basis++)
			 {
				 const std::vector<int> & order_local_basis = iter.second.order_local_alpt[num_basis];
				 iter.second.rhs[num_vec].at(order_local_basis) = rhs_explicit(order_alpt_basis_in_dgmap);
				 order_alpt_basis_in_dgmap++;
			 }
		 }
	 }
 }

IMEXEuler::IMEXEuler(BilinearForm & bilinearForm_explicit, BilinearForm & bilinearForm_implicit, const double dt, const std::string linear_solver):
	IMEX(bilinearForm_explicit, bilinearForm_implicit, dt), num_stage(1), linear_solver_type(linear_solver), implicit_linear_matx(dof, dof)
{
	init_linear_solver();
}

IMEXEuler::IMEXEuler(BilinearForm & bilinearForm_implicit, const double dt, const std::string linear_solver):
	IMEX(bilinearForm_implicit, dt), num_stage(1), linear_solver_type(linear_solver), implicit_linear_matx(dof, dof)
{
	init_linear_solver();
}

void IMEXEuler::init_linear_solver()
{
	if (linear_solver_type == "sparselu") { init_sparseLU_solver(); }
}

void IMEXEuler::init_sparseLU_solver()
{
	const int dof = dgoperator_ptr_implicit->row;

	// indentity matrix
	Eigen::MatrixXd I(dof, dof);
	I.setIdentity();

	implicit_linear_matx = I - _dt * dgoperator_ptr_implicit->mat;

	sparselu.analyzePattern(implicit_linear_matx);
	sparselu.factorize(implicit_linear_matx);
}

void IMEXEuler::step_stage(const int stage)
{	
	assert(stage==0);

	if (linear_solver_type == "sparselu") { ucoe = sparselu.solve(ucoe_tn + _dt * rhs); }
}

const double IMEX43::alpha = 0.24169426078821;
const double IMEX43::beta = 0.06042356519705;
const double IMEX43::eta = 0.12915286960590;

IMEX43::IMEX43(BilinearForm & bilinearForm_explicit, BilinearForm & bilinearForm_implicit, const double dt, const std::string linear_solver):
	IMEX(bilinearForm_explicit, bilinearForm_implicit, dt), num_stage(5), linear_solver_type(linear_solver), implicit_linear_matx(dof, dof)
{
	init_linear_solver();
	resize_variable();
}

IMEX43::IMEX43(BilinearForm & bilinearForm_implicit, const double dt, const std::string linear_solver):
	IMEX(bilinearForm_implicit, dt), num_stage(5), linear_solver_type(linear_solver), implicit_linear_matx(dof, dof)
{
	init_linear_solver();
	resize_variable();
}

void IMEX43::init_linear_solver()
{
	if (linear_solver_type == "cg") { init_cg_solver(); }
	else if (linear_solver_type == "bicg") { init_bicg_solver(); }
	else if (linear_solver_type == "sparselu") { init_sparseLU_solver(); }
}

void IMEX43::init_cg_solver()
{
	const int dof = dgoperator_ptr_implicit->row;

	// indentity matrix
	Eigen::SparseMatrix<double> I(dof, dof);
	I.setIdentity();

	cg.setTolerance(1e-12);
	cg.setMaxIterations(1e3);
	cg.compute(I - _dt * alpha * dgoperator_ptr_implicit->mat);
	
	implicit_linear_matx = I - _dt * alpha * dgoperator_ptr_implicit->mat;

	if( cg.info() != Eigen::Success) { std::cout << "fail in IMEX43::init_cg_solver()" << std::endl; exit(1); }
}

void IMEX43::init_bicg_solver()
{
	const int dof = dgoperator_ptr_implicit->row;

	// indentity matrix
	Eigen::SparseMatrix<double> I(dof, dof);
	I.setIdentity();

	bicg.setTolerance(1e-12);
	bicg.setMaxIterations(1e4);
	// bicg.preconditioner().setDroptol(1e-4);
	// bicg.preconditioner().setFillfactor(20);
	bicg.compute(I - _dt * alpha * dgoperator_ptr_implicit->mat);

	implicit_linear_matx = I - _dt * alpha * dgoperator_ptr_implicit->mat;

	if( bicg.info() != Eigen::Success) { std::cout << "fail in IMEX43::init_bicg_solver()" << std::endl; exit(1); }
}

void IMEX43::init_sparseLU_solver()
{
	const int dof = dgoperator_ptr_implicit->row;

	// indentity matrix
	Eigen::SparseMatrix<double> I(dof, dof);
	I.setIdentity();

	implicit_linear_matx = I - _dt * alpha * dgoperator_ptr_implicit->mat;
	sparselu.compute(implicit_linear_matx);

	if( sparselu.info() != Eigen::Success) { std::cout << "fail in IMEX43::init_sparseLU_solver()" << std::endl; exit(1); }
}

 void IMEX43::resize_variable()
 {
	 const int dof = dgoperator_ptr_implicit->row;

	 u1.resize(dof);
	 u2.resize(dof);
	 u3.resize(dof);
	 u4.resize(dof);

	 Fu2.resize(dof);
	 Fu3.resize(dof);
	 Fu4.resize(dof);

	 Gu1.resize(dof);
	 Gu2.resize(dof);
	 Gu3.resize(dof);
	 Gu4.resize(dof);
 }

 void IMEX43::step_rk()
 {
	 // 1st stage
	 u1 = cg.solve(ucoe_tn);
	 Gu1 = dgoperator_ptr_implicit->mat * u1;

	 // 2nd stage
	 u2 = cg.solve(ucoe_tn + _dt * (-alpha) * Gu1);
	 Gu2 = dgoperator_ptr_implicit->mat * u2;

	 // 3nd stage
	 Fu2 = dgoperator_ptr_explicit->mat * u2;

	 u3 = cg.solve(ucoe_tn + _dt * Fu2 + _dt * (1 - alpha) * Gu2);
	 Gu3 = dgoperator_ptr_implicit->mat * u3;

	 // 4th stage
	 Fu3 = dgoperator_ptr_explicit->mat * u3;

	 u4 = cg.solve(ucoe_tn + _dt * 0.25 * Fu2 + _dt * 0.25 * Fu3 + _dt * beta * Gu1 + _dt * eta * Gu2 + _dt * (0.5 - beta - eta - alpha) * Gu3);
	 Gu4 = dgoperator_ptr_implicit->mat * u4;

	 // 5th (final) stage
	 Fu4 = dgoperator_ptr_explicit->mat * u4;

	 ucoe = ucoe_tn + _dt * (1. / 6. * Fu2 + 1. / 6. * Fu3 + 2. / 3. * Fu4) + _dt * (1. / 6. * Gu2 + 1. / 6. * Gu3 + 2. / 3. * Gu4);
 }

void IMEX43::step_stage(const int stage)
{
	assert((stage >= 0) && (stage < num_stage));

	if (stage == 0)
	{
		if (linear_solver_type == "cg") { u1 = cg.solve(ucoe_tn); }
		else if (linear_solver_type == "bicg")  {  u1 = bicg.solve(ucoe_tn); }
		else if (linear_solver_type == "sparselu") { u1 = sparselu.solve(ucoe_tn); }

		Gu1 = dgoperator_ptr_implicit->mat * u1;

		ucoe = u1;
	}

	else if (stage == 1)
	{
		if (linear_solver_type == "cg") { u2 = cg.solve(ucoe_tn + _dt * (-alpha) * Gu1); }
		else if (linear_solver_type == "bicg")  { u2 = bicg.solve(ucoe_tn + _dt * (-alpha) * Gu1); }
		else if (linear_solver_type == "sparselu") { u2 = sparselu.solve(ucoe_tn + _dt * (-alpha) * Gu1); }

		Gu2 = dgoperator_ptr_implicit->mat * u2;

		ucoe = u2;
	}

	// before this stage, rhs should be updated based on explicit operator on u2
	else if (stage == 2)
	{
		Fu2 = rhs;

		if (linear_solver_type == "cg") { u3 = cg.solve(ucoe_tn + _dt * Fu2 + _dt * (1 - alpha) * Gu2); }
		else if (linear_solver_type == "bicg")  { u3 = bicg.solve(ucoe_tn + _dt * Fu2 + _dt * (1 - alpha) * Gu2); }
		else if (linear_solver_type == "sparselu") { u3 = sparselu.solve(ucoe_tn + _dt * Fu2 + _dt * (1 - alpha) * Gu2); }

		Gu3 = dgoperator_ptr_implicit->mat * u3;

		ucoe = u3;
	}

	// before this stage, rhs should be updated based on explicit operator on u3
	else if (stage == 3)
	{
		Fu3 = rhs;

		if (linear_solver_type == "cg") { u4 = cg.solve(ucoe_tn + _dt * 0.25 * Fu2 + _dt * 0.25 * Fu3 + _dt * beta * Gu1 + _dt * eta * Gu2 + _dt * (0.5 - beta - eta - alpha) * Gu3); }
		else if (linear_solver_type == "bicg")  { u4 = bicg.solve(ucoe_tn + _dt * 0.25 * Fu2 + _dt * 0.25 * Fu3 + _dt * beta * Gu1 + _dt * eta * Gu2 + _dt * (0.5 - beta - eta - alpha) * Gu3); }
		else if (linear_solver_type == "sparselu") { u4 = sparselu.solve(ucoe_tn + _dt * 0.25 * Fu2 + _dt * 0.25 * Fu3 + _dt * beta * Gu1 + _dt * eta * Gu2 + _dt * (0.5 - beta - eta - alpha) * Gu3); }

		Gu4 = dgoperator_ptr_implicit->mat * u4;

		ucoe = u4;
	}

	// before this stage, rhs should be updated based on explicit operator on u4
	else if (stage == 4)
	{
		Fu4 = rhs;

		ucoe = ucoe_tn + _dt * (1. / 6. * Fu2 + 1. / 6. * Fu3 + 2. / 3. * Fu4) + _dt * (1. / 6. * Gu2 + 1. / 6. * Gu3 + 2. / 3. * Gu4);
	}

	// check if linear solver really works
	if (linear_solver_type == "cg")
	{
		if( cg.info() != Eigen::Success) { std::cout << "fail in IMEX43::step_stage() in stage " << stage << std::endl; exit(1); }
	}
	else if (linear_solver_type == "bicg")
	{
		if( bicg.info() != Eigen::Success) { std::cout << "fail in IMEX43::step_stage() in stage " << stage << std::endl; exit(1); }
	}
	else if (linear_solver_type == "sparselu")
	{
		if( sparselu.info() != Eigen::Success) { std::cout << "fail in IMEX43::step_stage() in stage " << stage << std::endl; exit(1); }
	}
}