#include <gtest/gtest.h> 

#include "Quad.h"


const double TOL_1D = 5e-15;
const double TOL_MultiD = 1e-14;
const std::vector<int> Quad_num_test_list{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };



//* Test Function for 1D Integration (GL_1D function of class Quad)
TEST(QuadTest, PowerFunction_1D_Integration) {
	
	Quad testQuad(1);			    /// Quad variable for test, 1D
	double stp = 1.1, endp = -0.2;	/// start point and endpoint of integration: absolute value of them should not be too large; stp could be larger than endp
	
	for (int j : Quad_num_test_list) {
		for (int i = 0; i <= 2 * j - 1; i++) {

			std::function<double(double)> func_poly; 
			if (i != 0)
				func_poly = [&](double x) {return pow(x, i); };	/// polynomial x^i
			else
				func_poly = [](double x) {return 1.;};
			
			double int_correct = (pow(endp, i + 1) - pow(stp, i + 1)) / (1.0 + i);                /// correct integration of x^i from stp to endp
			double int_res = testQuad.GL_1D(func_poly, stp, endp, j);                             /// numerical integration using testQuad of Quad class
		
			EXPECT_TRUE(fabs(int_correct - int_res) < fabs(stp - endp)*TOL_1D) \
				<< "Integration of x^" << i << " on [" << stp << "," << endp << "] fails, with " << j << " many points:" \
				<< "Error is " << fabs(int_res - int_correct) << "\n";
			/// EXPECT_TRUE function to test whether the error of numerical integration is tolerable
		
		}
	}

}




//* Power Function in high dimension: return product w.r.t k from 0 to test_dim-1, of test_var[k]^test_pow[k]
double test_multidim_power(std::vector<double> test_var, std::vector<int> test_pow, int test_dim) {
	//ASSERT_EQ should not be used in a function that return something rather than void
	//		see https://stackoverflow.com/questions/34696483/assert-throw-error-void-value-not-ignored-as-it-ought-to-be
	//		or https://stackoverflow.com/questions/8896281/why-am-i-getting-void-value-not-ignored-as-it-ought-to-be/17915388
	//		for details

	double return_val = 1.;
	int actual_dim = std::min(test_dim, (int)(std::min(test_var.size(), test_pow.size())));
	if (actual_dim < test_dim)
		std::cout << "Either size of variable vector or size of power vector is smaller than the supposed dimension!!!\n" \
		<< "Variable size " << test_var.size() << ", power vector size " << test_pow.size() << ", dimension " << test_dim << "\n";
	for (int i = 0; i < actual_dim; i++)
		return_val *= pow(test_var[i], test_pow[i]);
	return return_val;
}



//* Test Function for Multi-Dimension Integration (GL_multiD function of class Quad)
TEST(QuadTest, PowerFunction_MultiDim_Integration) {

	for (int test_dim = 1; test_dim <= 3; test_dim++) {

		Quad test_Quad(test_dim);			  /// Quad variable for test, with DIM = test_dim
		std::vector<double> stp, endp;		  /// Start and End point for integration on each direction of multi-dimension
		for (int k = 1; k <= test_dim;k++) {
			stp.push_back(-0.3-k*0.02);
			endp.push_back(0.8+k*0.02);
		}

		for (int j : Quad_num_test_list) {		  /// Number of points on each direction to be tested
		if (test_dim==3 && j>=5) { continue; }
			for (long i = 0; i < pow(2 * j, test_dim) - 0.5; i++) {		
				/// index i, iterate over (2j)^dim, each of which represents a sequence of tested power; see following defintion of test_pow

				std::vector<int> test_pow;
				long ind = i;
				for (int k = 1; k <= test_dim; k++) {
					test_pow.push_back((int)(ind % (2 * j)));
					ind = ind / (2 * j);
				}

				std::function<double(std::vector<double>)> func_poly_multiDim \
					= std::bind(test_multidim_power, std::placeholders::_1, test_pow, test_dim);	/// create the power function for test, based on test_pow as powers
				
				double int_correct = 1.0;		/// accurate integration, to be calculated
				double vol = 1.0;				/// volume of integration area
				for (int k = 0; k < test_dim; k++) {
					int_correct *= (pow(endp[k], test_pow[k] + 1) - pow(stp[k], test_pow[k] + 1)) / (1. + test_pow[k]);
					vol *= (endp[k] - stp[k]);
				}

				double int_res = test_Quad.GL_multiD(func_poly_multiDim, stp, endp, j);    /// numerical integration, using j points on each direction

				EXPECT_TRUE(fabs(int_correct - int_res) < vol*TOL_MultiD) \
					<< "Integration Error (index " << i << ") is " << fabs(int_correct - int_res) << "\n";		/// EXPECT_TRUE function to test whether the error of numerical integration is tolerable

			}

			std::cout << "Finished: dimension " << test_dim << "; number of points " << j << "\n";

		}

	}

}





// Test of Multi-dimension case, other functions - ((x[0]+...+x[k])/k)^(2j-1) when j is number of points
TEST(QuadTest, OtherPolynomial_MultiDim_Integration) {

	std::function<double(std::vector<double>, int power)> func_poly = [](std::vector<double> x, int power) {
		return pow(std::accumulate(x.begin(), x.end(), 0.)/x.size(), power);
	};

	for (int test_dim = 1; test_dim <= 4; test_dim++) {

		double xabs = 1.; //1.2/test_dim;

		Quad test_Quad(test_dim);			  /// Quad variable for test, with DIM = test_dim
		std::vector<double> stp, endp;		  /// Start and End point for integration on each direction of multi-dimension
		for (int k = 1; k <= test_dim;k++) {
			stp.push_back(-xabs);
			endp.push_back(xabs);
		}

		for (int j : Quad_num_test_list) {				/// Number of points on each direction to be tested
			std::function<double(std::vector<double>)> func_poly_multiDim \
				= std::bind(func_poly, std::placeholders::_1, 2 * j - 1);		
			double int_res = test_Quad.GL_multiD(func_poly_multiDim, stp, endp, j);
			EXPECT_TRUE(fabs(int_res) < TOL_MultiD) << "Integration is " << int_res << " when j = " << j << " and dimension = " << test_dim << "\n";
			/// Power of the polynomial is odd number 2j-1, so we could find the accurate integral is 0
		}

		std::cout << "Finished: Dimension " << test_dim << "\n";

	}

}