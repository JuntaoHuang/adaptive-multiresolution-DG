#include "Hash.h"
#include "subs.h"

std::vector<std::vector<double>> Hash::bicoef_t;
std::map<int,int> Hash::mypow3;

bool Hash::check_no_duplicate(const int NMAX, const int DIM) 
{
	std::set<int> all_hash_key;

	std::vector<std::vector<int>> narray;
	const std::vector<int> narray_max(DIM, NMAX + 1);
	IterativeNestedLoop(narray, DIM, narray_max);

	// loop over all mesh level array of size dim
	int count = 0;
	for (auto const & lev_n : narray)
	{
		// generate index j: 0, 1, ..., 2^(n-1)-1, for n >= 2. Otherwise j=0 for n=0,1
		std::vector<int> jarray_max;
		for (auto const & n : lev_n)
		{
			int jmax = 1;
			if (n != 0) jmax = pow_int(2, n - 1);

			jarray_max.push_back(jmax);
		}

		std::vector<std::vector<int>> jarray;
		IterativeNestedLoop(jarray, DIM, jarray_max);

		for (auto const & sup_j : jarray)
		{
			// transform to odd index
			std::vector<int> odd_j(sup_j.size());
			for (size_t i = 0; i < sup_j.size(); i++) { odd_j[i] = 2 * sup_j[i] + 1; }
			
			auto it = all_hash_key.insert(hash_key(lev_n, odd_j));
			if (it.second==false)
			{
				std::cout << "there exist duplicated hash key" << std::endl;
				return false;
			}
		}
	}
	
	return true;
}

int Hash::hash_key(const std::array<std::vector<int>,2> & lj)
{
	return hash_key(lj[0], lj[1]);
}

int Hash::hash_key(const std::vector<int> & l, const std::vector<int> & j)
{
    const int d = l.size();
	//assert(d<=4 && "dimension is too large for this hash key function");
	//assert(d<4 && "there might be negetive hash key for dim=4");

	 int ind1, ind2, ind3;
	 std::map<int, int>::iterator it;
	 ind1 = 0;

	 for(int t=0; t<d; ++t){
		 it = mypow3.find(l[t]);
		 if(it!=mypow3.end()){
			 ind1 = ind1 * it->second + (j[t]-1)/2;
		 }
		 else{
			 int temp = myPow(2,l[t]);
			 ind1 = ind1 * temp + (j[t]-1)/2;
			 mypow3.insert(std::make_pair(l[t], temp));
		 }
	 }

	 int sum = l[0];
	 ind2 = 0;

	 for(int t=1; t<d; ++t){
		 ind2 -= bicoef_t[t+sum][t];
		 sum += l[t];
		 ind2 += bicoef_t[t+sum][t];
	 }

	 it = mypow3.find(sum);
	 if(it!=mypow3.end()){
		 ind2 *= it->second;
	 }
	 else{
		 int temp = myPow(2,sum);
		 ind2 *= temp;
		 mypow3.insert(std::make_pair(sum, temp));
	 }

	 ind3 = 0;

	 for(int s=0; s<sum; ++s){
		 it = mypow3.find(s);
		 if(it!=mypow3.end()){
			 ind3 += bicoef_t[d-1+s][d-1] * it->second;
		 }
		 else{
			 int temp = myPow(2,s);
			 ind3 += bicoef_t[d-1+s][d-1] * temp;
			 mypow3.insert(std::make_pair(s, temp));
		 }
	 }

	 return ind1+ind2+ind3;



}



double Hash::Factorial(double nValue)
   {
       double result = nValue;
       double result_next;
       double pc = nValue;

	   if(nValue <2 ) return 1.0;

       do
       {
           result_next = result*(pc-1);
           result = result_next;
           pc--;
       }while(pc>2);
       nValue = result;
       return nValue;
   }

double Hash::EvalBiCoef(double nValue, double nValue2)
   {
       double result;
       if(nValue2 == 1)return nValue;
       result = (Factorial(nValue))/(Factorial(nValue2)*Factorial((nValue - nValue2)));
       nValue2 = result;
       return nValue2;
   }

void Hash::bi_coef(){

	bicoef_t.resize(Nmax);
	for(int i=0; i<Nmax; ++i){
		bicoef_t[i].resize(i+1);
		for(int j=0; j<=i; ++j){
			if(j<=i/2)
				bicoef_t[i][j] = EvalBiCoef(i,j);
			else
				bicoef_t[i][j] = bicoef_t[i][i-j];
		}
	}
}

int Hash::myPow(int x, int p)
{
    if (p == 0) return 1;
    if (p == 1) return x;

    int tmp = myPow(x, p/2);
    if (p%2 == 0) return tmp * tmp;
    else return x * tmp * tmp;
}
