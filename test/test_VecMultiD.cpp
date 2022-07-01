#include <gtest/gtest.h>

#include "VecMultiD.h"

const double TOL = 1e-14;

TEST(VecMultiDTest, initial_2D)
{
    const int dim = 2;
    const int len = 4;
    VecMultiD<double> v(dim, len);
    EXPECT_EQ(v.size(), pow_int(len, dim)) << "vector size not correct";
    for (int i = 0; i < len; i++)
    {
        for (int j = 0; j < len; j++)
        {
            std::vector<int> index{ i, j };
            EXPECT_NEAR(v.at(index), 0., TOL) << "vector initialization not zero";
            EXPECT_NEAR(v.at(i,j), 0., TOL) << "vector initialization not zero";
        }
    }
}

TEST(VecMultiDTest, input_output_1D)
{
    const int dim = 1;
    const int sz = 4;

    VecMultiD<double> vec(dim, sz);
    vec.at(0) = 1.1;
    vec.at(1) = 2.1;
    vec.at(2) = 3.1;
    vec.at(3) = 4.1;

    for (size_t i = 0; i < sz; i++)
    {
        EXPECT_NEAR(vec.at(i), i+1.1, TOL);
    }
}

TEST(VecMultiDTest, input_output_2D)
{
    const int dim = 2;
    const int len = 4;
    VecMultiD<double> v(dim, len);

    // test input
    for (int i = 0; i < len; i++)
    {
        for (int j = 0; j < len; j++)
        {
            v.at(i, j) = i * 2 + j;
        }
    }

    // test output
    for (int i = 0; i < len; i++)
    {
        for (int j = 0; j < len; j++)
        {
            EXPECT_NEAR(v.at(i,j), i*2+j, TOL);
        }
    }
}

TEST(VecMultiDTest, input_output_3D)
{
    const int dim = 3;
    const int len = 4;
    VecMultiD<double> v(dim, len);

    for (int i = 0; i < len; i++)
    {
        for (int j = 0; j < len; j++)
        {
            for (int k = 0; k < len; k++)
            {
                v.at(i, j, k) = i * 2.2 + j + k/100.;
            }
        }
    }

    for (int i = 0; i < len; i++)
    {
        for (int j = 0; j < len; j++)
        {
            for (int k = 0; k < len; k++)
            {
                EXPECT_NEAR(v.at(i,j,k), i * 2.2 + j + k/100., TOL);
            }
        }
    }
}

TEST(VecMultiDTest, operator_test)
{
	const int dim = 2;
    const int len = 4;
	VecMultiD<double> u(dim, len), v(dim, len);
	for (int i = 0; i < len; i++)
	{
		for (int j = 0; j < len; j++)
		{				
			u.at(i,j) = 2.34 * i + j;				
		}
	}

	// copy constructor
	VecMultiD<double> w(u);
	for (int i = 0; i < len; i++)
	{
		for (int j = 0; j < len; j++)
		{			
            EXPECT_NEAR(u.at(i,j), w.at(i,j), TOL );
		}
	}				

	// assignment = operator
	v = u;
	for (int i = 0; i < len; i++)
	{
		for (int j = 0; j < len; j++)
		{				
            EXPECT_NEAR(u.at(i,j), v.at(i,j), TOL );
		}
	}

	// add += operator
	v += u;

	// substract += operator
	w -= v;

	// multiply by coefficient *= operator
	u *= 2.;

	for (int i = 0; i < len; i++)
	{
		for (int j = 0; j < len; j++)
		{				
            EXPECT_NEAR(v.at(i,j), 2*(2.34 * i + j), TOL);
			EXPECT_NEAR(w.at(i,j), -(2.34 * i + j), TOL);
            EXPECT_NEAR(u.at(i,j), 2*(2.34 * i + j), TOL);
		}
	}
    
    // set to be constant
    v = 1.;
	for (int i = 0; i < len; i++)
	{
		for (int j = 0; j < len; j++)
		{				
            EXPECT_NEAR(v.at(i,j), 1., TOL);
		}
	}    
}

TEST(VecMultiDTest, iterator_test)
{
    const std::vector<int> size{2,3};
    const int dim = 2;
    VecMultiD<double> u(dim, size);
    int iter = 0;
    std::vector<int> correct_index;
    for (auto const & index : u.get_index_iterator())
    {   
        if (iter==0) { correct_index = std::vector<int>{0, 0}; }
        if (iter==1) { correct_index = std::vector<int>{0, 1}; }
        if (iter==2) { correct_index = std::vector<int>{0, 2}; }
        if (iter==3) { correct_index = std::vector<int>{1, 0}; }
        if (iter==4) { correct_index = std::vector<int>{1, 1}; }
        if (iter==5) { correct_index = std::vector<int>{1, 2}; }

        EXPECT_EQ(index, correct_index);
        iter++;
    }
}

TEST(VecMultiDTest, reshape_test)
{   
    const int size_x = 2;
    const int size_y = 3;

    const std::vector<int> size{size_x, size_y};
    const int dim = 2;
    VecMultiD<double> u(dim, size);
    
    // u
    for (size_t i = 0; i < size_x; i++)
    {
        for (size_t j = 0; j < size_y; j++)
        {
            u.at(i,j) = i * size_y + j;
        }
    }
    
    // v = reshape of u
    VecMultiD<double> v(u);
    std::vector<int> new_size{size_y, size_x};
    v.reshape(new_size);

    for (size_t i = 0; i < size_y; i++)
    {
        for (size_t j = 0; j < size_x; j++)
        {
            EXPECT_NEAR(v.at(i,j), i * size_x + j, TOL);
        }
    }
}

TEST(VecMultiDTest, resize_test)
{
    const int size_x = 2;
    const int size_y = 3;

    const std::vector<int> size{size_x, size_y};
    const int dim = 2;
    VecMultiD<double> u(dim, size);

    const int new_dim = 3;
    const int size_z = 4;
    u.resize(std::vector<int>{size_x, size_y, size_z});

    EXPECT_EQ(u.size(), size_x * size_y * size_z);
}

TEST(VecMultiDTest, stdvec_to_vec_test)
{
    std::vector<double> u_vec{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};

    const std::vector<int> size{2, 3};
    VecMultiD<double> u;    
    u.stdvector_to_VecMultiD(u_vec, size);

    EXPECT_NEAR(u.at(0,0), 1.0, TOL);
    EXPECT_NEAR(u.at(0,1), 2.0, TOL);
    EXPECT_NEAR(u.at(0,2), 3.0, TOL);
    EXPECT_NEAR(u.at(1,0), 4.0, TOL);
    EXPECT_NEAR(u.at(1,1), 5.0, TOL);
    EXPECT_NEAR(u.at(1,2), 6.0, TOL);
}