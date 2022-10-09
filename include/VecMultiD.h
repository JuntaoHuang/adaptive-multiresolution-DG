#pragma once

#include "libs.h"
#include "subs.h"

template <class T>
class VecMultiD
{
public:
	// default constructor: initialized with a vector of size zero
	VecMultiD(): _dim(0), _vec_size(), _total_size(0), _vec(){};

	// constuctor: vector with uniform size in each dimension
	VecMultiD(const int dim_, const int vec_size_);

	// constuctor: vector with non-uniform size in each dimension
	VecMultiD(const int dim_, const std::vector<int> & vec_size_);

	// copy constructor
	VecMultiD(const VecMultiD & vec_copy) :
		_dim(vec_copy._dim), _vec_size(vec_copy._vec_size), _total_size(vec_copy._total_size),
		_vec(vec_copy._vec), _vec_accu_size(vec_copy._vec_accu_size){};

	// assignment = operator
	VecMultiD & operator=(const VecMultiD & vec_assign);

	// set vector to be constant
	VecMultiD & operator=(const T & const_assign);

	// add += operator
	VecMultiD & operator+=(const VecMultiD & vec_add);

	VecMultiD & operator+=(const T & const_add);

	// add + operator
	VecMultiD operator+(const VecMultiD & vec_add) const;

	// substract -= operator
	VecMultiD & operator-=(const VecMultiD & vec_subs);

	// substract - operator
	VecMultiD operator-(const VecMultiD & vec_subs) const;

	// multiply by coefficient *= operator
	VecMultiD & operator*=(const T coefficient);

	// multiply by coefficient * operator
	VecMultiD operator*(const T coefficient) const;

	~VecMultiD() {};

	// store all index for VecMultiD; temporarily remove it for saving memory
	//std::vector<std::vector<int>> index_iterator;
	// instead a function can be defined to get index_iterator;
	std::vector<std::vector<int>> get_index_iterator() const;

	// non-constant version
	T & at(const std::vector<int> & index);		// access element for multidimensional vector
	T & at(const int row, const int col);		// access element just for 2D vector
	T & at(const int row, const int col, const int hgt);	// access element just for 3D vector
	T & at(const int d1, const int d2, const int d3, const int d4);		// access element for 4D vector
	T & at(const int order);

	// constant version
	const T & at(const std::vector<int> & index) const;		// access element for multidimensional vector
	const T & at(const int row, const int col) const;		// access element just for 2D vector
	const T & at(const int row, const int col, const int hgt) const;	// access element just for 3D vector
	const T & at(const int order) const;

	int size() const;	// return total size of this multi dimensional vector		
	std::vector<int> vec_size() const { return _vec_size; };	// return size in each dimension of this multi dimensional vector

	// transfer to vector or matrix in Eigen library
	Eigen::VectorXd vec_to_eigenvec();
	Eigen::MatrixXd mat_to_eigenmat();

	// reshape but keep total size and values of all elements unchanged
	// e.g. VecMultiD a with vec_size (3,4), we can make it size to be (2,6) by using a.reshape(2,6)
	// but we cannot make it size to be (1,6) since total size should not be changed in this function
	void reshape(const std::vector<int> & new_vec_size);

	// resize 
	void resize(const std::vector<int> & new_vec_size);

	// resize without update index_iterator
	void resize_no_index_iterator(const std::vector<int> & new_vec_size);
	
	void set_zero();	// set all elements to be zero

	int num_non_zero() const;	// return number of non-zero entries

	double sparsity() const;	// return sparsity (defined as nnz/size)

	double norm(const double p = 2.0) const;	// return Lp norm of this vector

	// print all the elements for vector (1D) and matrix (2D)
	void print() const;

	// convert (or reshape) std::vector<T> to VecMultiD<T>
	void stdvector_to_VecMultiD(const std::vector<T> & vec, std::vector<int> size_VecMultiD);

private:
	int _dim;
	std::vector<int> _vec_size;	//	_vec_size in each dimension	
	int _total_size;

	std::vector<T> _vec;	

	// return total size of a vec (muliplication of size in all dimensions)
	// i.e. vec[0]*vec[1]*...*vec[d-1]
	int multiply_size(const std::vector<int> & vec) const;

public:
	std::vector<int> _vec_accu_size; // accumulated size of vector
	
	int multiD_to_1D(const std::vector<int> & index) const { return std::inner_product(index.cbegin(), index.cend(), _vec_accu_size.cbegin(), 0);}

};

// constructor: vector with uniform size in each dimension
template<class T>
VecMultiD<T>::VecMultiD(const int dim_, const int vec_size_) :
	_dim(dim_), _vec_size(_dim, vec_size_), _total_size(1), _vec(), _vec_accu_size(_dim,1)
{
	for (size_t d = 0; d < _dim; d++)
	{
		_total_size *= vec_size_;
	}
	for (size_t i = 0; i < _total_size; i++)
	{
		if constexpr(std::is_same_v<T, int>)
		{
			_vec.push_back(0);
		}
		else if constexpr (std::is_same_v<T, double>)
		{
			_vec.push_back(0.);
		}
		else
		{
			T vec_elem;
			_vec.push_back(vec_elem);
		}
	}


	if (_dim == 1) return;
	for(size_t d=_dim-2; d<_vec_accu_size.size(); d--)
	{
		_vec_accu_size[d] = _vec_accu_size[d + 1] * _vec_size[d + 1];
	
	}
	//IterativeNestedLoop(index_iterator, _dim, _vec_size);
}

// constuctor: vector with different size in each dimension
template<class T>
VecMultiD<T>::VecMultiD(const int dim_, const std::vector<int> & vec_size_) :
	_dim(dim_), _vec_size(vec_size_), _total_size(1), _vec(), _vec_accu_size(_dim, 1)
{
	assert(_dim==_vec_size.size());
	_total_size = multiply_size(_vec_size);

	for (size_t i = 0; i < _total_size; i++)
	{
		if constexpr (std::is_same_v<T, int>)
		{
			_vec.push_back(0);
		}
		else if constexpr (std::is_same_v<T, double>)
		{
			_vec.push_back(0.);
		}
		else
		{
			T vec_elem;
			_vec.push_back(vec_elem);
		}
	}

	if (_dim == 1) return;
	for (size_t d = _dim - 2; d < _vec_accu_size.size(); d--)
	{
		_vec_accu_size[d] = _vec_accu_size[d + 1] * _vec_size[d + 1];

	}
	//IterativeNestedLoop(index_iterator, _dim, _vec_size);
}

// assignment = operator 
template<class T>
VecMultiD<T> & VecMultiD<T>::operator=(const VecMultiD & vec_assign)
{
	assert(_vec_size == vec_assign._vec_size);
	if (this != &vec_assign)
	{
		_vec = vec_assign._vec;
	}
	return *this;
}

template<class T>
VecMultiD<T> & VecMultiD<T>::operator=(const T & const_assign)
{
	for (size_t i = 0; i < _total_size; i++)
	{
		_vec[i] = const_assign;
	}
	return *this;
}

// add += operator
template<class T>
VecMultiD<T> & VecMultiD<T>::operator+=(const VecMultiD & vec_add)
{
	assert(_vec_size == vec_add._vec_size);
	for (size_t i = 0; i < _total_size; i++)
	{
		_vec[i] += vec_add._vec[i];
	}
	return *this;
}

template<class T>
VecMultiD<T> & VecMultiD<T>::operator+=(const T & const_add)
{
	for (size_t i = 0; i < _total_size; i++)
	{
		_vec[i] += const_add;
	}
	return *this;
}

// add + operator
template<class T>
VecMultiD<T> VecMultiD<T>::operator+(const VecMultiD & vec_add) const
{
	VecMultiD<T> vec_tmp(*this);
	for (size_t i = 0; i < _total_size; i++)
	{
		vec_tmp._vec[i] += vec_add._vec[i];
	}
	return vec_tmp;
}

// substract -= operator
template<class T>
VecMultiD<T> & VecMultiD<T>::operator-=(const VecMultiD & vec_subs)
{
	assert(_vec_size == vec_subs._vec_size);
	for (size_t i = 0; i < _total_size; i++)
	{
		_vec[i] -= vec_subs._vec[i];
	}
	return *this;
}

// substract - operator
template<class T>
VecMultiD<T> VecMultiD<T>::operator-(const VecMultiD & vec_subs) const
{
	VecMultiD<T> vec_tmp(*this);
	for (size_t i = 0; i < _total_size; i++)
	{
		vec_tmp._vec[i] -= vec_subs._vec[i];
	}
	return vec_tmp;
}

// multiply by coefficient *= operator
template<class T>
VecMultiD<T> & VecMultiD<T>::operator*=(const T coefficient)
{
	for (size_t i = 0; i < _total_size; i++)
	{
		_vec[i] *= coefficient;
	}
	return *this;
}

// multiply by coefficient * operator
template<class T>
VecMultiD<T> VecMultiD<T>::operator*(const T coefficient) const
{
	VecMultiD<T> vec_tmp(*this);
	for (size_t i = 0; i < _total_size; i++)
	{
		vec_tmp._vec[i] *= coefficient;
	}
	return vec_tmp;
}

template<class T>
int VecMultiD<T>::size() const
{
	return _total_size;
}

// non-constant version
template<class T>
T & VecMultiD<T>::at(const std::vector<int>& index)
{
	int order = multiD_to_1D(index);
	//assert(order < _total_size);
	return _vec[order];
}

template<class T>
T & VecMultiD<T>::at(const int row, const int col)
{
	int order = row * _vec_size[1] + col;
	assert(order < _total_size);
	return _vec[order];
}

template<class T>
T & VecMultiD<T>::at(const int row, const int col, const int hgt)
{
	int order = (row * _vec_size[1] + col) * _vec_size[2] + hgt;
	assert(order < _total_size);
	return _vec[order];
}

template<class T>
T & VecMultiD<T>::at(const int d1, const int d2, const int d3, const int d4)
{
	int order = ((d1 * _vec_size[1] + d2) * _vec_size[2] + d3) * _vec_size[3] + d4;
	assert(order < _total_size);
	return _vec[order];
}

template<class T>
T & VecMultiD<T>::at(const int order)
{
	assert(order < _total_size);

	return _vec[order];
}

// constant version
template<class T>
const T & VecMultiD<T>::at(const std::vector<int>& index) const
{
	int order = multiD_to_1D(index);
	assert(order < _total_size);
	return _vec[order];
}

template<class T>
const T & VecMultiD<T>::at(const int row, const int col) const
{
	int order = row * _vec_size[1] + col;
	assert(order < _total_size);
	return _vec[order];
}

template<class T>
const T & VecMultiD<T>::at(const int row, const int col, const int hgt) const
{
	int order = (row * _vec_size[1] + col) * _vec_size[2] + hgt;
	assert(order < _total_size);
	return _vec[order];
}

template<class T>
const T & VecMultiD<T>::at(const int order) const
{
	assert(order < _total_size);
	return _vec[order];
}

//template<class T>
//int VecMultiD<T>::multiD_to_1D(const std::vector<int>& index) const
//{
//	// vec(i_0, i_1, ..., i_{_dim-1})
//	// vec_size in each dimension (s_0, s_1, ..., s_{_dim-1})
//	// order = i_{_dim-1} 
//	//		+ i_{_dim-2} * s_{_dim-1}
//	//		+ i_{_dim-3} * s_{_dim-1} * s_{_dim-2}
//	//		+ ...
//	//		+ i_0 * s_{_dim-1} * s_{_dim-2} * ... * s_1
//	int order = 0;
//	int prod = 1;
//	for (size_t d = _dim; d > 0; d--)
//	{
//		order += index[d - 1] * prod;
//		prod *= _vec_size[d - 1];
//	}
//	return order;
//}

//template<class T>
//int VecMultiD<T>::multiD_to_1D(const std::vector<int>& index) const
//{
//	// vec(i_0, i_1, ..., i_{_dim-1})
//	// vec_size in each dimension (s_0, s_1, ..., s_{_dim-1})
//	// order = i_{_dim-1} 
//	//		+ i_{_dim-2} * s_{_dim-1}
//	//		+ i_{_dim-3} * s_{_dim-1} * s_{_dim-2}
//	//		+ ...
//	//		+ i_0 * s_{_dim-1} * s_{_dim-2} * ... * s_1
//	int order = 0;
//	//int prod = 1;
//	for (size_t d = 0; d < _dim; d++)
//	{
//		order += index[d] * _vec_accu_size[d];
//		//prod *= _vec_size[d - 1];
//	}
//	return order;
//}

template<class T>
Eigen::VectorXd VecMultiD<T>::vec_to_eigenvec()
{
	assert(_dim==1);

	Eigen::VectorXd eigenvec(_total_size);
	for (int i = 0; i < _total_size; i++)
	{
		eigenvec(i) = _vec[i];

		if (std::abs(eigenvec(i)) <= Const::ROUND_OFF) { eigenvec(i) = 0.; }
	}
	return eigenvec;
}

template<class T>
Eigen::MatrixXd VecMultiD<T>::mat_to_eigenmat()
{
	assert(_dim==2);

	Eigen::MatrixXd eigenmat(_vec_size[0], _vec_size[1]);
	for (int row = 0; row < _vec_size[0]; row++)
	{
		for (int col = 0; col < _vec_size[1]; col++)
		{
			eigenmat(row, col) = this->at(row, col);

			if (std::abs(eigenmat(row, col)) <= Const::ROUND_OFF) { eigenmat(row, col) = 0.; }
		}
	}
	return eigenmat;
}

template<class T>
void VecMultiD<T>::reshape(const std::vector<int> & new_vec_size)
{
	// calculate total size of new vector
	// make sure that it is the same with original
	int new_tot_size = multiply_size(new_vec_size);
	assert(new_tot_size == _total_size);

	_dim = new_vec_size.size();
	_vec_size = new_vec_size;

	_vec_accu_size.resize(new_vec_size.size());
	_vec_accu_size.assign(_vec_accu_size.size(), 1);
	if (_dim == 1) return;
	for (size_t d = _dim - 2; d < _vec_accu_size.size(); d--)
	{
		_vec_accu_size[d] = _vec_accu_size[d + 1] * _vec_size[d + 1];

	}

	// update index iterator
	//IterativeNestedLoop(index_iterator, _dim, _vec_size);
}

template<class T>
void VecMultiD<T>::resize(const std::vector<int> & new_vec_size)
{
	int new_tot_size = multiply_size(new_vec_size);

	_vec.resize(new_tot_size);
	_dim = new_vec_size.size();
	_vec_size = new_vec_size;
	_total_size = new_tot_size;



	_vec_accu_size.resize(new_vec_size.size());
	_vec_accu_size.assign(_vec_accu_size.size(), 1);
	if (_dim == 1) return;
	for (size_t d = _dim - 2; d < _vec_accu_size.size(); d--)
	{
		_vec_accu_size[d] = _vec_accu_size[d + 1] * _vec_size[d + 1];

	}

	// update index iterator
	//IterativeNestedLoop(index_iterator, _dim, _vec_size);
}

template<class T>
void VecMultiD<T>::resize_no_index_iterator(const std::vector<int> & new_vec_size)
{
	int new_tot_size = multiply_size(new_vec_size);

	_vec.resize(new_tot_size);
	_dim = new_vec_size.size();
	_vec_size = new_vec_size;
	_total_size = new_tot_size;

	_vec_accu_size.resize(new_vec_size.size());
	_vec_accu_size.assign(_vec_accu_size.size(), 1);
	if (_dim == 1) return;
	for (size_t d = _dim - 2; d < _vec_accu_size.size(); d--)
	{
		_vec_accu_size[d] = _vec_accu_size[d + 1] * _vec_size[d + 1];

	}
}

template<class T>
int VecMultiD<T>::multiply_size(const std::vector<int> & vec) const
{	
	if (vec.size()==0) { return 0; }
	
	int vec_multiply_size = 1;
	for (auto const & v : vec) 
	{ 
		vec_multiply_size *= v; 
	}
	return vec_multiply_size;
}

template<class T>
void VecMultiD<T>::set_zero()
{
	for (auto & v : _vec) { v = 0; }
}

template<class T>
inline void VecMultiD<T>::print() const
{
	if (_dim==1)
	{
		for (auto const & v : _vec)
		{
			std::cout << v << "  ";
		}		
		std::cout << std::endl;
	}
	else if (_dim==2)
	{
		for (size_t i = 0; i < _vec_size[0]; i++)
		{
			for (size_t j = 0; j < _vec_size[1]; j++)
			{
				std::cout << this->at(i, j) << "  ";
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;
	}
	else if (_dim==3)
	{
		for (size_t i = 0; i < _vec_size[0]; i++)
		{
			for (size_t j = 0; j < _vec_size[1]; j++)
			{
				for (size_t k = 0; k < _vec_size[2]; k++)
				{
					std::cout << this->at(i, j, k) << "  ";
				}				
			}
		}
		std::cout << std::endl;
	}	
	else
	{
		std::cout << "only print information for dimension = 1, 2, 3 in VecMultiD<T>::print()" << std::endl;
		exit(1);
	}
}

template<class T>
int VecMultiD<T>::num_non_zero() const
{
	int nnz = 0;
	for (auto const & v : _vec)
	{
		if (std::abs(v) > Const::ROUND_OFF) { nnz ++; }
	}
	return nnz;
}

template<class T>
double VecMultiD<T>::sparsity() const
{
	return num_non_zero()/(_total_size + 0.0);
}

template<class T>
double VecMultiD<T>::norm(const double p) const
{
	assert(p>0.);

	double norm_lp = 0.;
	for(auto const & v : _vec)
	{
		norm_lp += std::pow(v, p);
	}
	return std::pow(norm_lp, 1./p);
}

template<class T>
std::vector<std::vector<int>> VecMultiD<T>::get_index_iterator() const
{
	std::vector<std::vector<int>> index_iterator;
	IterativeNestedLoop(index_iterator, _dim, _vec_size);
	return index_iterator;

}

template<class T>
void VecMultiD<T>::stdvector_to_VecMultiD(const std::vector<T> & vec, std::vector<int> size_VecMultiD)
{		
	_vec_size = size_VecMultiD;

	_dim = _vec_size.size();

	_total_size = 1;
	for (size_t d = 0; d < _dim; d++)
	{
		_total_size *= _vec_size[d];
	}

	assert(_total_size == vec.size());

	_vec = vec;
}