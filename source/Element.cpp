#include "Element.h"
#include "subs.h"

int Element::PMAX_alpt;	// max polynomial degree for Alpert's basis functions
int Element::PMAX_intp;	// max polynomial degree for interpolation basis functions
int Element::DIM;			// dimension
int Element::VEC_NUM;		// num of unknown variables in PDEs
std::vector<std::vector<bool>> Element::is_intp; //specify which components need interpolation 

Element::Element(const std::vector<int> & level_, const std::vector<int> & suppt_, AllBasis<AlptBasis> & all_bas, Hash & hash):
	level(level_), 
	suppt(suppt_),
	hash_key(0),
	order_local_alpt(),
	order_local_intp(),
	order_global_alpt(),
	order_global_intp(),
	ucoe_alpt(),
	ucoe_alpt_predict(),
	ucoe_tn(),
	rhs(),
	num_total_chd(0), 
	num_total_par(0),
	new_add(true)
{
	assert(level.size()==DIM && suppt.size()==DIM);

	hash_key = hash.hash_key(level, suppt);

	fucoe_intp.resize(VEC_NUM);
	fucoe_intp_inter.resize(VEC_NUM);
	fp_intp.resize(VEC_NUM);


	for (int i = 0; i < VEC_NUM; i++)
	{
		for (int d = 0; d < DIM; d++)
		{
			VecMultiD<double> a = is_intp[i][d]? VecMultiD<double>(DIM, PMAX_intp+1) : VecMultiD<double>();

			fucoe_intp[i].push_back(a);
			fp_intp[i].push_back(a);

			VecMultiD<std::vector<double> > b = is_intp[i][d] ? VecMultiD<std::vector<double>> (DIM, PMAX_intp+1) : VecMultiD<std::vector<double>>();
			if (is_intp[i][d])
			{
				auto index_iterator = b.get_index_iterator();

				for (auto it : index_iterator)
				{
					(b.at(it)).resize(DIM + 1);
				}
			}
			
			fucoe_intp_inter[i].push_back(b);
		}		
	}


	for (int i = 0; i < VEC_NUM; i++)
	{
		VecMultiD<double> a(DIM, PMAX_intp+1);

		ucoe_intp.push_back(a);

		up_intp.push_back(a);
	}


	VecMultiD<std::vector<double> > b(DIM, PMAX_intp+1);
	auto index_iterator = b.get_index_iterator();
	for (auto it : index_iterator)
	{
		( b.at(it) ).resize(DIM + 1); 
	}

	std::vector<VecMultiD<std::vector<double> > > c(VEC_NUM, b);

	ucoe_intp_inter = c;


	VecMultiD<double> scalar_init(DIM, PMAX_alpt+1);
	std::vector<VecMultiD<double>> vec_init(VEC_NUM, scalar_init);
	ucoe_alpt = vec_init;
	ucoe_alpt_t_m1 = vec_init;
	ucoe_alpt_predict = vec_init;
	ucoe_alpt_predict_t_m1 = vec_init;
	ucoe_tn = vec_init;
	ucoe_ut = vec_init;
	ucoe_ut_predict = vec_init;
	rhs = vec_init;

	ucoe_trans_from = ucoe_alpt;
	ucoe_trans_to = ucoe_alpt;
	source = ucoe_alpt;

	{
		VecMultiD<int> order_int(DIM, PMAX_alpt+1);
		std::vector<VecMultiD<int>> order_vec_int(VEC_NUM, order_int);
		order_alpt_basis_in_dg = order_vec_int;
	}

	{
		VecMultiD<int> order_int(DIM, PMAX_intp+1);
		std::vector<VecMultiD<int>> order_vec_int(VEC_NUM, order_int);
		order_intp_basis_in_dg = order_vec_int;
	}

	order_local_alpt = ucoe_alpt[0].get_index_iterator();	
	for (auto const & order_local : order_local_alpt)
	{
		std::vector<int> order_global(DIM);
		for (size_t d = 0; d < DIM; d++)
		{
			order_global[d] = index_to_order_basis(level[d], suppt[d], order_local[d], PMAX_alpt);
		}		
		order_global_alpt.push_back(order_global);
	}

	order_local_intp = fucoe_intp[0][0].get_index_iterator();
	for (auto const & order_local : order_local_intp)
	{
		std::vector<int> order_global(DIM);
		for (size_t d = 0; d < DIM; d++)
		{
			order_global[d] = index_to_order_basis(level[d], suppt[d], order_local[d], PMAX_intp);
		}		
		order_global_intp.push_back(order_global);
	}		

	order_elem = index_to_order_elem(level,suppt);

	for (int d = 0; d < DIM; d++)
	{
		xl.push_back(all_bas.at(level[d], suppt[d], 0).supp_interv[0]);
		xr.push_back(all_bas.at(level[d], suppt[d], 0).supp_interv[1]);
		dis_point.push_back(all_bas.at(level[d], suppt[d], 0).dis_point);
		supp_interv.push_back(all_bas.at(level[d], suppt[d], 0).supp_interv);
	}
}

int Element::size_alpt() const
{
	return pow_int(PMAX_alpt+1, DIM);
}

int Element::size_intp() const
{
	return pow_int(PMAX_intp+1, DIM);
}

std::vector<double> Element::val(const std::vector<double> & x, const AllBasis<AlptBasis> & all_bas, const std::vector<int> & derivative) const
{
	std::vector<double> value(VEC_NUM, 0.);
	for (size_t d = 0; d < DIM; d++)
	{
		if (x[d]<xl[d] || x[d]>xr[d])	return value;
	}

	// loop over all basis in this element
	for (size_t num_vec = 0; num_vec < VEC_NUM; num_vec++)
	{
		for (size_t i = 0; i < size_alpt(); i++)
		{	
			double val_bas = ucoe_alpt[num_vec].at(order_local_alpt[i]);
			const auto & order = order_global_alpt[i];
			for (size_t d = 0; d < DIM; d++)
			{
				val_bas *= all_bas.at(order[d]).val(x[d], derivative[d], 0);
			}
			value[num_vec] += val_bas;
		}
	}
	
	return value;
}

std::vector<double> Element::val_Lag(const std::vector<double> & x, const AllBasis<LagrBasis> & all_bas_Lag) const
{
	std::vector<double> value(VEC_NUM, 0.);
	for (size_t d = 0; d < DIM; d++)
	{
		if (x[d]<xl[d] || x[d]>xr[d])	return value;
	}

	// loop over all basis in this element
	for (size_t num_vec = 0; num_vec < VEC_NUM; num_vec++)
	{
		for (size_t i = 0; i < size_intp(); i++)
		{	
			double val_bas = ucoe_intp[num_vec].at(order_local_intp[i]);
			const std::vector<int> & order = order_global_intp[i];
			for (size_t d = 0; d < DIM; d++)
			{
				val_bas *= all_bas_Lag.at(order[d]).val(x[d], 0, 0);				
			}
			value[num_vec] += val_bas;
		}
	}
	
	return value;
}

std::vector<double> Element::val_Her(const std::vector<double> & x, const AllBasis<HermBasis> & all_bas_Her) const
{
	std::vector<double> value(VEC_NUM, 0.);
	for (size_t d = 0; d < DIM; d++)
	{
		if (x[d]<xl[d] || x[d]>xr[d])	return value;
	}

	// loop over all basis in this element
	for (size_t num_vec = 0; num_vec < VEC_NUM; num_vec++)
	{
		for (size_t i = 0; i < size_intp(); i++)
		{	
			double val_bas = ucoe_intp[num_vec].at(order_local_intp[i]);
			const std::vector<int> & order = order_global_intp[i];
			for (size_t d = 0; d < DIM; d++)
			{
				val_bas *= all_bas_Her.at(order[d]).val(x[d], 0, 0);				
			}
			value[num_vec] += val_bas;
		}
	}
	
	return value;
}

double Element::val_Lag(const std::vector<double> & x, const int ii, const int dd, const AllBasis<LagrBasis> & all_bas_Lag) const
{
	double val = 0.;
	for (size_t d = 0; d < DIM; d++)
	{
		if (x[d]<xl[d] || x[d]>xr[d])
		{
			return val;
		}
	}

	// loop over all basis in this element
	for (size_t i = 0; i < size_intp(); i++)
	{	

		double val_bas = fucoe_intp[ii][dd].at(order_local_intp[i]);

		std::vector<int> order = order_global_intp[i];

		for (size_t d = 0; d < DIM; d++)
		{
			val_bas *= all_bas_Lag.at(order[d]).val(x[d], 0, 0);
		}		

		val += val_bas;
	}
	return val;
}

// return value of flux function at position x with Hermite interpolation

double Element::val_Her(const std::vector<double> & x, const int ii, const int dd, const AllBasis<HermBasis> & all_bas_Her) const
{
	double val = 0.;
	for (size_t d = 0; d < DIM; d++)
	{
		if (x[d]<xl[d] || x[d]>xr[d])
		{
			return val;
		}
	}

	// loop over all basis in this element
	for (size_t i = 0; i < size_intp(); i++)
	{	
		double val_bas = fucoe_intp[ii][dd].at(order_local_intp[i]);
		std::vector<int> order = order_global_intp[i];
		for (size_t d = 0; d < DIM; d++)
		{
			val_bas *= all_bas_Her.at(order[d]).val(x[d], 0, 0);
		}		

		val += val_bas;
	}
	return val;
}

int Element::index_to_order_basis(const int n, const int j, const int p, const int PMAX)
{
	assert(p <= PMAX);

	return (n == 0) ? p : ((pow_int(2, n - 1) + (j - 1) / 2)*(PMAX + 1) + p);
}

bool Element::is_vol_alpt(const Element & elem, const int dim) const
{
	for (size_t d = 0; d < DIM; d++)
	{
		// in the input dimension, support interval in 1D should have intersection
		if (d==dim)
		{
			if (!(is_interval_intersect(supp_interv[d], elem.supp_interv[d]))) { return false; }
		}
		// in other dimensions, index should be exactly the same because of orthogonality
		else
		{
			if ( level[d]!=elem.level[d] || suppt[d]!=elem.suppt[d]) { return false; }
		}		
	}
	return true;
}

bool Element::is_flx_alpt(const Element & elem, const int dim) const
{
	for (size_t d = 0; d < DIM; d++)
	{
		// in the input dimension, discontinuous point of test element (at least one) should be inside support interval
		if (d==dim)
		{
			if (!(is_interval_intersect_adjacent(supp_interv[d], elem.supp_interv[d]))) { return false; }
		}
		// in other dimensions, index should be exactly the same because of orthogonality
		else
		{
			if ( level[d]!=elem.level[d] || suppt[d]!=elem.suppt[d]) { return false; }
		}		
	}
	return true;
}

bool Element::is_vol_intp(const Element & elem) const
{
	for (size_t d = 0; d < DIM; d++)
	{
		if (!(is_interval_intersect(supp_interv[d], elem.supp_interv[d]))) { return false; }
	}
	return true;
}

bool Element::is_flx_intp(const Element & elem, const int dim) const
{
	for (size_t d = 0; d < DIM; d++)
	{
		// in the input dimension, discontinuous point of test element (at least one) should be inside support interval
		if (d==dim)
		{
			if (!(is_interval_intersect_adjacent(supp_interv[d], elem.supp_interv[d]))) { return false; }
		}
		// in other dimensions, support interval in 1D should have intersection
		else
		{
			if (!(is_interval_intersect(supp_interv[d], elem.supp_interv[d]))) { return false; }
		}		
	}
	return true;
}

bool Element::is_same_elem_1D(const Element & elem, const int dim) const
{
	return ( (level[dim]==elem.level[dim]) && (suppt[dim]==elem.suppt[dim]) );
}

// return true if two intervals intersect or adjacent
bool Element::is_interval_intersect_adjacent(const std::vector<double> & interval_u, const std::vector<double> & interval_v)
{
	assert(interval_u.size() == 2 && interval_v.size() == 2);

	const bool no_intersect = ( (interval_u[0] >= interval_v[1]) || (interval_u[1] <= interval_v[0]) );
	const bool adjacent = ( (std::abs(interval_u[0]-interval_v[1])<Const::ROUND_OFF) || (std::abs(interval_u[1]-interval_v[0])<Const::ROUND_OFF) );
	const bool periodic_adjacent = ( ( (std::abs(interval_u[0])<Const::ROUND_OFF) && (std::abs(interval_v[1]-1.0)<Const::ROUND_OFF) )
								|| ( (std::abs(interval_v[0])<Const::ROUND_OFF) && (std::abs(interval_u[1]-1.0)<Const::ROUND_OFF) ) );
	
	if (!no_intersect || adjacent || periodic_adjacent) { return true; }
	return false;
}

bool Element::is_pt_in_interval(const std::vector<double> & pt, const std::vector<double> & interval)
{
	// if there exist at least one point (in a set of points) is inside an interval
	// then return true
	for (auto const & p : pt)
	{
		if ( is_pt_in_interval(p, interval)	) return true;
	}
	return false;
}

bool Element::is_pt_in_interval(const double pt, const std::vector<double> & interval)
{		
	// periodic boundary condition
	// if (point is at x=0 && right end of interval is at x=1)
	// or (point is at x=1 && left end of interval is at x=0)
	// then this point is in the interval
	if ( (std::abs(pt)<Const::ROUND_OFF && std::abs(interval[1]-1.)<Const::ROUND_OFF) ||
		(std::abs(pt-1.0)<Const::ROUND_OFF && std::abs(interval[0])<Const::ROUND_OFF) )
	{
		return true;
	}
	
	// if (point < left end of interval)
	// or (point > right end of interval)
	// then this point is not in the interval
	if ( (pt<interval[0]-Const::ROUND_OFF) ||  (pt>interval[1]+Const::ROUND_OFF) )	return false;
	return true;
}

// return true if two intervals intersect
bool Element::is_interval_intersect(const std::vector<double> & interval_u, const std::vector<double> & interval_v)
{
	assert(interval_u.size() == 2 && interval_v.size() == 2);

	if ( (interval_u[0] >= interval_v[1]) || (interval_u[1] <= interval_v[0]) ) { return false; }
	return true;
}

int Element::index_to_order_elem(const int n, const int j)
{
	return (n == 0) ? 0 : ((pow_int(2, n - 1) + (j - 1) / 2));
}

std::vector<int> Element::index_to_order_elem(const std::vector<int> & level, const std::vector<int> & suppt)
{
	std::vector<int> index;
	for (size_t d = 0; d < level.size(); d++)
	{
		index.push_back(index_to_order_elem(level[d],suppt[d]));
	}
	return index;
}