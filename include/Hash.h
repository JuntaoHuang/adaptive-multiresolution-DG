#pragma once
#include "libs.h"

// copy from sg_index.h and sg_index.cpp by Wei Guo
// it is valid for vector of dim <=4
// there exist segmentation fault for dim>=5
class Hash
{
public:
    Hash() { bi_coef(); };
    ~Hash() {};

    // input two vectors of the same dimension, return a unique integer (hash key)
    int hash_key(const std::vector<int> & l, const std::vector<int> & j);

    // overload, just combine l and j together
    int hash_key(const std::array<std::vector<int>,2> & lj);

    // check that there exist no duplicated hash key for all elements of full grid DG
    // only use for debug
    // this function will take a long long time especially for high dimension
    bool check_no_duplicate(const int NMAX, const int DIM);

private:

    const static int Nmax = 100;
    
    static std::vector<std::vector<double>> bicoef_t;
    
    static std::map<int,int> mypow3;

    double Factorial(double nValue);

    double EvalBiCoef(double nValue, double nValue2);

    void bi_coef();

    int myPow(int x, int p);
};