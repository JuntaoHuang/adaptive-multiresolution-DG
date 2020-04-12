#pragma once

#include "libs.h"
#include "subs.h"
#include "Quad.h"

// T can be AlptBasis, LagrBasis, HermBasis
// collection of all basis functions in 1D
template <class T>
class AllBasis
{
public:
	AllBasis(const int max_level_);
	~AllBasis() {};
	
	// get basis function with index (level, suppt, dgree)
	const T & at(const int level, const int suppt, const int dgree) const;

	// get basis function with index denoting order
	const T & at(const int order) const;

	int size() const;	// number of all basis functions in 1D

	/**
	 * @brief project function in 1D to collection of basis functions or its derivatives
	 * 
	 * @param func 
	 * @param derivative 
	 * @return std::vector<double> coefficients of all basis functions
	 * @note	this function only make sense for Alpert basis
	 */
	std::vector<double> projection(std::function<double(double)> func, const int derivative = 0) const;

	// return value of this collection of basis function at point x with given coefficient
	double val(const std::vector<double> & ucoe, const double x) const;

	// get L1, L2 and Linf error between a function and this collection of basis function
	// return a vector of size 3: {err_L1, err_L2, err_Linf}
	std::vector<double> get_error(const std::vector<double> & ucoe, std::function<double(double)> func, const int gauss_point) const;

	const int PMAX;
	const int NMAX;

private:
	int bas_size;
	
	// store all the basis ind 1d info in a vector;
	std::vector<T> allbasis;

	int index_to_order(const int n, const int j, const int p) const;	
};

template<class T>
AllBasis<T>::AllBasis(const int max_level_): 
	PMAX(T::PMAX), NMAX(max_level_), bas_size(0)
{
	for (int n = 0; n < NMAX + 1; n++)
	{
		for (int j = 0; j < pow(2, n - 1); j++)
		{
			// mapping from {0, ..., 2^(n-1)-1} to {1, 3, ..., 2^n-1}
			const int suppt = 2 * j + 1;

			for (int p = 0; p < PMAX + 1; p++)
			{
				T bas(n, suppt, p);
				allbasis.push_back(bas);
				bas_size ++;
			}
		}
	}	
}

template<class T>
int AllBasis<T>::index_to_order(const int n, const int j, const int p) const
{
	return (n == 0) ? p : ((pow_int(2, n - 1) + (j - 1) / 2)*(PMAX + 1) + p);
}

template<class T>
const T & AllBasis<T>::at(const int level, const int suppt, const int dgree) const
{
	const int ind = index_to_order(level, suppt, dgree);
	return allbasis[ind];
}

template<class T>
const T & AllBasis<T>::at(const int order) const
{
	return allbasis[order];
}

template<class T>
int AllBasis<T>::size() const
{
	return bas_size;
}

template<class T>
std::vector<double> AllBasis<T>::projection(std::function<double(double)> func, const int derivative) const
{	
	std::vector<double> ucoe(bas_size);
	for (size_t i = 0; i < bas_size; i++)
	{
		ucoe[i] = allbasis[i].product_function(func, derivative);
	}	
	return ucoe;
}

template<class T>
double AllBasis<T>::val(const std::vector<double> & ucoe, const double x) const
{
	double value = 0.;
	for (size_t i = 0; i < bas_size; i++)
	{
		value += ucoe[i] * allbasis[i].val(x);
	}
	return value;
}

template<class T>
std::vector<double> AllBasis<T>::get_error(const std::vector<double> & ucoe, std::function<double(double)> func, const int gauss_point) const
{
	auto err_fun = [&](std::vector<double> x) { return std::abs( val(ucoe, x[0]) - func(x[0]) ); };
	
	const int dim = 1;
	Quad quad(dim);
	return quad.norm_multiD(err_fun, NMAX, gauss_point);
}