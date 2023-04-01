#include <gtest/gtest.h>
#include "Quad.h"
#include "AlptBasis.h"

const double TOL = 1e-14;

// test orthogonality of Alpert basis functions
TEST(AlptBasisTest, orthogonal)
{
    AlptBasis::PMAX = 2;

    const int level_basis_u = 1;
    const int suppt_basis_u = 1;
    const int dgree_basis_u = 2;

    AlptBasis basis_u(level_basis_u, suppt_basis_u, dgree_basis_u);

    const int level_basis_v = 2;
    const int suppt_basis_v = 3;
    const int dgree_basis_v = 1;
    AlptBasis basis_v(level_basis_v, suppt_basis_v, dgree_basis_v);    

    // different basis function should be orthogonal
    double val = basis_u.product_volume(basis_v, 0, 0, 10);    
    EXPECT_NEAR(val, 0., TOL);    

    val = basis_v.product_volume(basis_u, 0, 0, 10);
    EXPECT_NEAR(val, 0., TOL);

    // norm of basis function should be 1
    val = basis_u.product_volume(basis_u, 0, 0, 10);
    EXPECT_NEAR(val, 1., TOL);

    val = basis_v.product_volume(basis_v, 0, 0, 10);
    EXPECT_NEAR(val, 1., TOL);
}

