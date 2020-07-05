#pragma once

#include "libs.h"

void IterativeNestedLoop(std::vector<std::vector<int>> & array, const int dim, const std::vector<int> & max);

void IterativeNestedLoop(std::vector<std::vector<int>> & array, const int dim, const int max);

void GenerateIndexMultiD(std::set<std::array<std::vector<int>,2>> & index_multiD, const std::vector<std::vector<std::array<int,2>>> & index_1D);

int pow_int(int base, int exp);

// print an integer on scree, just for debug
void print(int i);

// Fill the zipped vector with pairs consisting of the
// corresponding elements of a and b. 
// It is required that two vectors have equal length
template <typename A, typename B>
void zip(const std::vector<A> &a, const std::vector<B> &b, std::vector<std::pair<A, B>> &zipped) 
{
	for (size_t i = 0; i < a.size(); ++i)
	{
		zipped.push_back(std::make_pair(a[i], b[i]));
	}
}

// Write the first and second element of the pairs in 
// the given zipped vector into a and b. 
// It is required that the vectors have equal length
template <typename A, typename B>
void unzip(const std::vector<std::pair<A, B>> &zipped, std::vector<A> &a, std::vector<B> &b) {

	for (size_t i = 0; i < a.size(); i++)
	{
		a[i] = zipped[i].first;
		b[i] = zipped[i].second;
	}
}


// sort vector a by values of vector b (smaller one ranks before bigger one)
// these two vectors should be of the same length
template <typename A, typename B>
void sort_first_by_second_order(std::vector<A> &a, std::vector<B> &b) {

	// Zip the vectors together
	std::vector<std::pair<A, B>> zipped;
	zip(a, b, zipped);

	// Sort the vector of pairs
	std::sort(std::begin(zipped), std::end(zipped),
		[&](const auto& a, const auto& b)
	{
		return a.second < b.second;
	});

	// Write the sorted pairs back to the original vectors
	unzip(zipped, a, b);
}

template <typename T>
std::vector<T> linspace(T start_point, T end_point, size_t num)
{	
	assert(num >= 2);

    T h = (end_point - start_point) / static_cast<T>(num-1);
    std::vector<T> xs(num);
    typename std::vector<T>::iterator x;
    T val;
    for (x = xs.begin(), val = start_point; x != xs.end(); ++x, val += h) { *x = val; }
    return xs;
}

template <typename T> std::string to_str(const T & t)
{
  std::ostringstream os;
  os << t;
  return os.str ();
}