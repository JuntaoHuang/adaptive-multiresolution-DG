
#include "subs.h"


// input:  dim, dynamic number of nested loops
// 		max, vec of size dim, max number in each dim loop
// output: array, each element is a vector of size dim, every number in each element loops from 0 to max-1
// example:
// dim = 2, max = {2,3}
// array = {{0,0},{0,1},{0,2},{1,0},{1,1},{1,2}}
// adapted from: https://stackoverflow.com/questions/18732974/c-dynamic-number-of-nested-for-loops-without-recursion
void IterativeNestedLoop(std::vector<std::vector<int> > & arr, const int dim, const std::vector<int> & max) {

	if(!arr.empty()) arr.clear();

	int arr_size = 1;
	for (auto const & i: max) arr_size *= i;
	arr.reserve(arr_size);

	std::vector<int> slots(dim, 0);

	int index = dim-1;
	while (true)
	{
		// TODO: Your inner loop code goes here. You can inspect the values in slots
		arr.push_back(slots);

		// Increment
		slots[dim-1]++;

		// Carry
		while (slots[index] == max[index])
		{
			// Overflow, we're done
			if (index == 0)
			{
				return;
			}

			slots[index--] = 0;
			slots[index]++;
		}

		index = dim-1;
	}
}

// overload, so that max can be an integer
void IterativeNestedLoop(std::vector<std::vector<int> > & arr, const int dim, const int max) {

	std::vector<int> max_arr(dim, max);

	IterativeNestedLoop(arr, dim, max_arr);
}

// copy from https://stackoverflow.com/questions/101439/the-most-efficient-way-to-implement-an-integer-based-power-function-powint-int
int pow_int(int base, int exp)
{
	assert(exp>=0);
	
    int result = 1;
    for (;;)
    {
        if (exp & 1)
            result *= base;
        exp >>= 1;
        if (!exp)
            break;
        base *= base;
    }
    return result;
}

// input: 
// 		index_1D; vector<vector<array<int,2>>>, size of dim
// 		for each dimension, it is a vec of array<int, 2>
// output:
// 		index_multiD: tensor product of vector in each dimension
// 
// for example:
// 		index_1D:
// 		dim 1: {2,4}, {3,5}
// 		dim 2: {8,10}, {20, 23}, {30,32}

// 		index_multiD:
// 		{(2 8), (4 10)} 
// 		{(3 8), (5 10)}
// 		{(2 20), (4 23)}
// 		{(3 20), (5 23)}
// 		{(2 30), (4 32)}
// 		{(3 30), (5 32)}
// 
void GenerateIndexMultiD(std::set<std::array<std::vector<int>,2>> & index_multiD, const std::vector<std::vector<std::array<int,2>>> & index_1D)
{
	if (!index_multiD.empty()) index_multiD.clear();

	const int dim = index_1D.size();
	std::vector<std::vector<int> > arr;
	std::vector<int> nmax(dim);
	for (size_t d = 0; d < dim; d++)
	{
		nmax[d] = index_1D[d].size();
	}
	
	IterativeNestedLoop(arr, dim, nmax);

	for (auto const & idx : arr)
	{
		std::vector<int> n(dim);
		std::vector<int> j(dim);
		for (size_t d = 0; d < dim; d++)
		{			
			n[d] = index_1D[d][idx[d]][0];
			j[d] = index_1D[d][idx[d]][1];
		}
		std::array<std::vector<int>,2> nj_pair = {n, j};
		index_multiD.insert(nj_pair);
	}
}

void print(int i)
{
	std::cout << i << std::endl;
}