#include <gtest/gtest.h> 

#include "subs.h"

TEST(TestSubs, TestIterativeNestedLoop)
{
    const int dim = 2;    
    const std::vector<int> max{2,3};
    std::vector<std::vector<int> > arr;

    IterativeNestedLoop(arr, dim, max);

    int iter = 0;
    std::vector<int> correct_index;
    for (auto const & vec : arr)
    {
        if (iter==0) { correct_index = std::vector<int>{0, 0}; }
        if (iter==1) { correct_index = std::vector<int>{0, 1}; }
        if (iter==2) { correct_index = std::vector<int>{0, 2}; }
        if (iter==3) { correct_index = std::vector<int>{1, 0}; }
        if (iter==4) { correct_index = std::vector<int>{1, 1}; }
        if (iter==5) { correct_index = std::vector<int>{1, 2}; }

        EXPECT_EQ(vec, correct_index);
        iter++;        
    }    
}

TEST(TestSubs, TestPowInt)
{
    int base = 3;    
    EXPECT_EQ(pow_int(3, 0), 1);
    EXPECT_EQ(pow_int(3, 1), 3);
    EXPECT_EQ(pow_int(3, 2), 9);
    EXPECT_EQ(pow_int(3, 3), 27);
    
    base = -2;
    EXPECT_EQ(pow_int(-2, 0), 1);
    EXPECT_EQ(pow_int(-2, 1), -2);
    EXPECT_EQ(pow_int(-2, 2), 4);
    EXPECT_EQ(pow_int(-2, 3), -8);
}