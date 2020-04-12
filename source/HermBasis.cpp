#include "HermBasis.h"

int HermBasis::PMAX;
int HermBasis::msh_case;

std::vector<double> HermBasis::intp_msh0;
std::vector<double> HermBasis::intp_msh1;

HermBasis::HermBasis(const int level_, const int suppt_, const int dgree_) :
    Basis(level_, suppt_, dgree_)
{
    for (size_t i = 0; i < 3; i++)
	{
		lft[i] = val(dis_point[i], 0, -1);
		rgt[i] = val(dis_point[i], 0, 1);
		jmp[i] = rgt[i] - lft[i];
	}

    if(level == 0)
	{
		intep_pt = intp_msh0[dgree];
	}
	else
	{
		double h = supp_interv[1] - supp_interv[0];

		intep_pt = supp_interv[0] + h * intp_msh1[dgree];
	}

}

/////////////////////////////////////////////////////
// dgree = 0: point 1, function value
// dgree = 1: point 2, function value
// dgree = 2: point 1, first derivative
// dgree = 3: point 2, first derivative
// dgree = 4: point 1, second derivative, if available
// dgree = 5: point 2, second derivative, if available

void HermBasis::set_interp_msh01()
{

    if (PMAX == 3)
    {
        intp_msh0 = std::vector<double>{ -1.0 + Const::ROUND_OFF, 1.0 - Const::ROUND_OFF, -1.0 + Const::ROUND_OFF, 1.0 - Const::ROUND_OFF };
        intp_msh1 = std::vector<double>{ 0.0 - Const::ROUND_OFF, 0.0 + Const::ROUND_OFF, 0.0 - Const::ROUND_OFF, 0.0 + Const::ROUND_OFF };
    }
    else if (PMAX == 5)
    {
        intp_msh0 = std::vector<double>{ -1.0 + Const::ROUND_OFF, 1.0 - Const::ROUND_OFF, -1.0 + Const::ROUND_OFF, 1.0 - Const::ROUND_OFF, -1.0 + Const::ROUND_OFF, 1.0 - Const::ROUND_OFF };
        intp_msh1 = std::vector<double>{ 0.0 - Const::ROUND_OFF, 0.0 + Const::ROUND_OFF, 0.0 - Const::ROUND_OFF, 0.0 + Const::ROUND_OFF, 0.0 - Const::ROUND_OFF, 0.0 + Const::ROUND_OFF };
    }
    else 
    {

		std::cout << "parameter out of range in function set_intp_msh01 of HermBasis" << std::endl;
		exit(1);
	}

    // linear map from [-1, 1] to [0, 1]
	auto f = [](double x) { return (x + 1) / 2.; };
	for (int i = 0; i < PMAX + 1; ++i) {

		intp_msh0[i] = f(intp_msh0[i]);
		intp_msh1[i] = f(intp_msh1[i]);
	}
    
}


double HermBasis::val(const double x, const int derivative, const int sgn) const
{
    double xlim = x;
    if (sgn == -1)
    {
        if (std::abs(x) <= Const::ROUND_OFF) return 0.;
        xlim -= Const::ROUND_OFF;
    }
    else if (sgn == 1)
    {
        if (std::abs(x - 1) <= Const::ROUND_OFF) return 0.;
        xlim += Const::ROUND_OFF;
    }

    if (derivative == 0)
    {
        return val0(xlim, level, suppt, dgree);
    }
    else if (derivative == 1)
    {
        return val1(xlim, level, suppt, dgree);
    }
    else if (derivative == 2)
    {
        return val2(xlim, level, suppt, dgree);
    }
	else if (derivative == 3)
    {
        return val3(xlim, level, suppt, dgree);
    }
    else
    {
        std::cout << "derivative degree is too large in definition of val in HermBasis.cpp" << std::endl;
		exit(1);
    }

}


// from degree to get the index of point and derivative 

void HermBasis::deg_pt_deri_1d(const int pp, int & p, int & l)
{
	// p: index for point
	// l: index for function or derivative

    // transform between pp and p & l
    if (pp == 0)
    {
        p = 0;
        l = 0;
    }
    else if (pp == 1)
    {
        p = 1;
        l = 0;
    }
    else if (pp == 2)
    {
        p = 0;
        l = 1;
    }
    else if (pp == 3)
    {
        p = 1;
        l = 1;
    }
    else if (pp == 4)
    {
        p = 0;
        l = 2;
    }
    else if (pp == 5)
    {
        p = 1;
        l = 2;
    }
    else
    {
        std::cout << "pp out of range in function deg_pt_deri_1d in HermBasis.cpp" << std::endl;
        exit(1); 
    }

}


double HermBasis::val0(const double x, const int n, const int j, const int p)
{
	if (n <= 1)
	{
        if (PMAX == 3)
        {
            return phi_P1_L1(x, n, p);
        }
        else if (PMAX == 5)
        {
            return phi_P1_L2(x, n, p);
        }
        else
        {
            std::cout << "PMAX out of range in function val0 of HermBasis" << std::endl;
            exit(1); 
        } 
		
	}
	else
	{
		int odd_j = (j - 1)/2;
		double xtrans = pow(2, n - 1)*x - odd_j;

		int q, l; // l:index for function or derivative

		deg_pt_deri_1d(p, q, l);

        if (PMAX == 3)
        {
            return phi_P1_L1(xtrans, 1, p) * std::pow(2, -(n-1) * l);
        }
        else if (PMAX == 5)
        {
            return phi_P1_L2(xtrans, 1, p) * std::pow(2, -(n-1) * l);
        }
        else
        {
            std::cout << "PMAX out of range in function val0 of HermBasis" << std::endl;
            exit(1); 
        }
        
	}
}

double HermBasis::val1(const double x, const int n, const int j, const int p)
{
	if (n <= 1)
	{
        if (PMAX == 3)
        {
            return phi_P1_L1_D1(x, n, p);
        }
        else if (PMAX == 5)
        {
            return phi_P1_L2_D1(x, n, p);
        }
        else
        {
            std::cout << "PMAX out of range in function val1 of HermBasis" << std::endl;
            exit(1); 
        } 
		
	}
	else
	{

		int q, l;  // l:index for function or derivative

		deg_pt_deri_1d(p, q, l);

		int ll = 1; // since here compute 1st-order derivative

		////////////////

		int odd_j = (j - 1)/2;
		double xtrans = pow(2, n - 1)*x - odd_j;

        if (PMAX == 3)
        {
            return phi_P1_L1_D1(xtrans, 1, p) * std::pow(2, -(n-1) * (l - ll));
        }
        else if (PMAX == 5)
        {
            return phi_P1_L2_D1(xtrans, 1, p) * std::pow(2, -(n-1) * (l - ll));
        }
        else
        {
            std::cout << "PMAX out of range in function val1 of HermBasis" << std::endl;
            exit(1); 
        }
        
	}
}

double HermBasis::val2(const double x, const int n, const int j, const int p)
{
	if (n <= 1)
	{
		if (PMAX == 3)
		{
			return phi_P1_L1_D2(x, n, p);
		}
        else if (PMAX == 5)
        {
            return phi_P1_L2_D2(x, n, p);
        }
        else
        {
            std::cout << "function val2 of HermBasis can not run for: " << PMAX << std::endl;
            exit(1); 
        } 
		
	}
	else
	{
		int q, l; // l:index for function or derivative

		deg_pt_deri_1d(p, q, l);

		int ll = 2; // since here compute 2nd-order derivative

		////////////////////

		int odd_j = (j - 1)/2;
		double xtrans = pow(2, n - 1)*x - odd_j;

		if (PMAX == 3)
		{
			return phi_P1_L1_D2(xtrans, 1, p) * std::pow(2, -(n-1) * (l - ll));
		}
        else if (PMAX == 5)
        {
            return phi_P1_L2_D2(xtrans, 1, p) * std::pow(2, -(n-1) * (l - ll));
        }
        else
        {
            std::cout << "function val2 of HermBasis can not run for: " << PMAX << std::endl;
            exit(1); 
        }
        
	}
}


double HermBasis::val3(const double x, const int n, const int j, const int p)
{
	if (n <= 1)
	{
		if (PMAX == 3)
		{
			return phi_P1_L1_D3(x, n, p);
		}
        else
        {
            std::cout << "function val3 of HermBasis can not run for: " << PMAX << std::endl;
            exit(1); 
        } 
		
	}
	else
	{
		int q, l; // l:index for function or derivative

		deg_pt_deri_1d(p, q, l);

		int ll = 3; // since here compute 3rd-order derivative

		////////////////////

		int odd_j = (j - 1)/2;
		double xtrans = pow(2, n - 1)*x - odd_j;

		if (PMAX == 3)
		{
			return phi_P1_L1_D3(xtrans, 1, p) * std::pow(2, -(n-1) * (l - ll));
		}
        else
        {
            std::cout << "function val3 of HermBasis can not run for: " << PMAX << std::endl;
            exit(1); 
        }
        
	}
}

/////////////////////////////////

// P=1: 2 interpolation points
// L=1: every point has value and 1st-order derivative
double HermBasis::phi_P1_L1(const double x, int msh, const int pp)
{
    if (x<0 || x>1) return 0.;

    int p = 0; // index for point
    int l = 0; // index for function or derivative

	deg_pt_deri_1d(pp, p, l);

    /////////////////////

    if (msh==0){  // level 0
		
		if (p==0){ // left point
			
			if (l==0){ // function
				
				return (-1+x)*(-1+x)*(1+2*x);
			}	
			else if (l==1){ // 1st-order derivative
				
				return (1-x)*(1-x)*x;
			}	
		}			
		else if (p==1){ // right point
			
			if (l==0){

				return -x*x*(-3+2*x);
			}	
			else if (l==1){
				
				return (-1+x)*x*x;
			}	
		}

	}
	else if (msh==1){ // level 1

		if (p==0){
			
			if (l==0){

				if (x<=0.5){
				 	
					return -4*x*x*(4*x-3);
				}
				else if (x>0.5){

					return 0.;
				}
			}	
			else if (l==1){
				
				if (x<=0.5){
				 	
					return 2*x*x*(2*x-1);
				}
				else if (x>0.5){

					return 0.;
				}					
			}	
		}			
		else if (p==1){
			
			if (l==0){
				
				if (x<0.5){
				 	
					return 0.;
				}
				else if (x>=0.5){

					return 4*(x-1)*(x-1)*(4*x-1);
				}
			}	
			else if (l==1){
				
				if (x<0.5){
				 	
					return 0.;
				}
				else if (x>=0.5){

					return 2*(x-1)*(x-1)*(2*x-1);
				}					
			}	
		}

	}

    //////////////////////
    
    std::cout << "parameter out of range in function phi_P1_L1 in HermBasis.cpp " << std::endl;
	exit(1);
    
}


// P=1: 2 interpolation points
// L=1: every point has value and 1st-order derivative
// D=1: 1st-order derivative of basis function
double HermBasis::phi_P1_L1_D1(const double x, int msh, const int pp)
{

    if (x<0 || x>1) return 0.;

    int p = 0; // index for point
    int l = 0; // index for function or derivative

	deg_pt_deri_1d(pp, p, l);

    /////////////////////

    if (msh==0){ // level 0 
		
		if (p==0){ // left point
			
			if (l==0){ // function
				
				return 6*x*(-1+x);
			}	
			else if (l==1){ // 1st-order derivative
				
				return (x-1)*(3*x-1);
			}	
		}			
		else if (p==1){ // right point
			
			if (l==0){

				return -6*x*(x-1);
			}	
			else if (l==1){
				
				return x*(3*x-2);
			}	
		}

	}
	else if (msh==1){ // level 1 

		if (p==0){
			
			if (l==0){

				if (x<=0.5){
				 	
					return 24*x*(1-2*x);
				}
				else if (x>0.5){

					return 0.;
				}
			}	
			else if (l==1){
				
				if (x<=0.5){
				 	
					return 4*x*(3*x-1);
				}
				else if (x>0.5){

					return 0.;
				}					
			}	
		}			
		else if (p==1){
			
			if (l==0){
				
				if (x<0.5){
				 	
					return 0.;
				}
				else if (x>=0.5){

					return 24*(x-1)*(2*x-1);
				}
			}	
			else if (l==1){
				
				if (x<0.5){
				 	
					return 0.;
				}
				else if (x>=0.5){

					return 4*(x-1)*(3*x-2);
				}					
			}	
		}

	}

    ////////////////////

    std::cout << "parameter out of range in function phi_P1_L1_D1" << std::endl;
	exit(1);

}

// P=1: 2 interpolation points
// L=1: every point has value and 1st-order derivative
// D=2: 2nd-order derivative of basis function
double HermBasis::phi_P1_L1_D2(const double x, int msh, const int pp)
{

    if (x<0 || x>1) return 0.;

    int p = 0; // index for point
    int l = 0; // index for function or derivative

	deg_pt_deri_1d(pp, p, l);

    /////////////////////

    if (msh==0){ // level 0 
		
		if (p==0){ // left point
			
			if (l==0){ // function
				
				return 12 * x - 6;
			}	
			else if (l==1){ // 1st-order derivative
				
				return 6 * x - 4;
			}	
		}			
		else if (p==1){ // right point
			
			if (l==0){

				return -12 * x + 6;
			}	
			else if (l==1){
				
				return 6 * x - 2;
			}	
		}

	}
	else if (msh==1){ // level 1 

		if (p==0){
			
			if (l==0){

				if (x<=0.5){
				 	
					return -96 * x + 24;
				}
				else if (x>0.5){

					return 0.;
				}
			}	
			else if (l==1){
				
				if (x<=0.5){
				 	
					return 24 * x - 4;
				}
				else if (x>0.5){

					return 0.;
				}					
			}	
		}			
		else if (p==1){
			
			if (l==0){
				
				if (x<0.5){
				 	
					return 0.;
				}
				else if (x>=0.5){

					return 96 * x - 72;
				}
			}	
			else if (l==1){
				
				if (x<0.5){
				 	
					return 0.;
				}
				else if (x>=0.5){

					return 24 * x - 20;
				}					
			}	
		}

	}

    ////////////////////

    std::cout << "parameter out of range in function phi_P1_L1_D2 in  HermBasis.cpp" << std::endl;
	exit(1);

}

// P=1: 2 interpolation points
// L=1: every point has value and 1st-order derivative
// D=3: 3rd-order derivative of basis function
double HermBasis::phi_P1_L1_D3(const double x, int msh, const int pp)
{

    if (x<0 || x>1) return 0.;

    int p = 0; // index for point
    int l = 0; // index for function or derivative

	deg_pt_deri_1d(pp, p, l);

    /////////////////////

    if (msh==0){ // level 0 
		
		if (p==0){ // left point
			
			if (l==0){ // function
				
				return 12.;
			}	
			else if (l==1){ // 1st-order derivative
				
				return 6.;
			}	
		}			
		else if (p==1){ // right point
			
			if (l==0){

				return -12.;
			}	
			else if (l==1){
				
				return 6.;
			}	
		}

	}
	else if (msh==1){ // level 1 

		if (p==0){
			
			if (l==0){

				if (x<=0.5){
				 	
					return -96.;
				}
				else if (x>0.5){

					return 0.;
				}
			}	
			else if (l==1){
				
				if (x<=0.5){
				 	
					return 24.;
				}
				else if (x>0.5){

					return 0.;
				}					
			}	
		}			
		else if (p==1){
			
			if (l==0){
				
				if (x<0.5){
				 	
					return 0.;
				}
				else if (x>=0.5){

					return 96.;
				}
			}	
			else if (l==1){
				
				if (x<0.5){
				 	
					return 0.;
				}
				else if (x>=0.5){

					return 24.;
				}					
			}	
		}

	}

    ////////////////////

    std::cout << "parameter out of range in function phi_P1_L1_D3 in HermBasis.cpp" << std::endl;
	exit(1);

}



// P=1: 2 interpolation points
// L=2: every point has value, 1st-order derivative and 2nd-order derivative
double HermBasis::phi_P1_L2(const double x, int msh, const int pp)
{
    if (x<0 || x>1) return 0.;

    int p = 0; // index for point
    int l = 0; // index for function or derivative

	deg_pt_deri_1d(pp, p, l);

    /////////////////////

    if (msh==0){ // level 0
		
		if (p==0){ // left point
			
			if (l==0){ // function 
				
				return -pow(x-1,3.)*(6*x*x+3*x+1);
			}	
			else if (l==1){ // 1st-order derivative
				
				return -x*pow(x-1,3.)*(3*x+1);
			}	
			else if (l==2){  // 2nd-order derivative
				
				return -0.5*x*x*pow(x-1,3.);
			}				
		}			
		else if (p==1){  // right point
			
			if (l==0){ 

				return pow(x,3.)*(6*x*x-15*x+10);
			}	
			else if (l==1){  
				
				return -pow(x,3.)*(x-1)*(3*x-4);
			}	
			else if (l==2){
				
				return 0.5*pow(x,3.)*pow(x-1,2.);
			}				
		}

	}
	else if (msh==1){ // level 1

		if (p==0){ 
			
			if (l==0){

				if (x<=0.5){
				 	
					return 16*pow(x,3.)*(12*x*x-15*x+5);
				}
				else if (x>0.5){

					return 0.;
				}
			}	
			else if (l==1){
				
				if (x<=0.5){
				 	
					return -8*pow(x,3.)*(2*x-1)*(3*x-2);
				}
				else if (x>0.5){

					return 0.;
				}					
			}	
			else if (l==2){
				
				if (x<=0.5){
				 	
					return pow(x,3.)*pow(2*x-1,2.);
				}
				else if (x>0.5){

					return 0.;
				}					
			}	

		}			
		else if (p==1){
			
			if (l==0){
				
				if (x<0.5){
				 	
					return 0.;
				}
				else if (x>=0.5){

					return -16*pow(x-1,3.)*(12*x*x-9*x+2);
				}
			}	
			else if (l==1){
				
				if (x<0.5){
				 	
					return 0.;
				}
				else if (x>=0.5){

					return -8*pow(x-1,3.)*(2*x-1)*(3*x-1);
				}					
			}	
			else if (l==2){
				
				if (x<0.5){
				 	
					return 0.;
				}
				else if (x>=0.5){

					return -pow(x-1,3.)*pow(2*x-1,2.);
				}					
			}	

		}

	}

    //////////////////////
    
    std::cout << "parameter out of range in function phi_P1_L2" << std::endl;
	exit(1);
    
}

// P=1: 2 interpolation points
// L=2: every point has value, 1st-order derivative and 2nd_order derivative
// D=1: 1st-order derivative of basis function
double HermBasis::phi_P1_L2_D1(const double x, int msh, const int pp)
{

    if (x<0 || x>1) return 0.;

    int p = 0; // index for point
    int l = 0; // index for function or derivative

	deg_pt_deri_1d(pp, p, l);

    /////////////////////

    if (msh==0){ // level 0
		
		if (p==0){ // left point
			
			if (l==0){ // function
				
				return -30*pow(x,2.)*pow(x-1,2.);
			}	
			else if (l==1){ // 1st-order derivative
				
				return -pow(x-1,2.)*(3*x-1)*(5*x+1);
			}	
			else if (l==2){ // 2nd-order derivative
				
				return -0.5*x*(5*x-2)*pow(x-1,2.);
			}			

		}			
		else if (p==1){ // right point
			
			if (l==0){

				return 30*pow(x,2.)*pow(x-1,2.);
			}	
			else if (l==1){
				
				return -pow(x,2.)*(3*x-2)*(5*x-6);
			}	
			else if (l==2){
				
				return 0.5*pow(x,2.)*(x-1)*(5*x-3);
			}				
		}

	}
	else if (msh==1){ // level 1

		if (p==0){
			
			if (l==0){

				if (x<=0.5){
				 	
					return 240*pow(x,2.)*pow(2*x-1,2.);
				}
				else if (x>0.5){

					return 0.;
				}
			}	
			else if (l==1){
				
				if (x<=0.5){
				 	
					return -16*pow(x,2.)*(3*x-1)*(5*x-3);
				}
				else if (x>0.5){

					return 0.;
				}					
			}	
			else if (l==2){
				
				if (x<=0.5){
				 	
					return pow(x,2.)*(2*x-1)*(10*x-3);
				}
				else if (x>0.5){

					return 0.;
				}					
			}				
		}			
		else if (p==1){
			
			if (l==0){
				
				if (x<0.5){
				 	
					return 0.;
				}
				else if (x>=0.5){

					return -240*pow(x-1,2.)*pow(2*x-1,2.);
				}
			}	
			else if (l==1){
				
				if (x<0.5){
				 	
					return 0.;
				}
				else if (x>=0.5){

					return -16*pow(x-1,2.)*(3*x-2)*(5*x-2);
				}					
			}	
			else if (l==2){
				
				if (x<0.5){
				 	
					return 0.;
				}
				else if (x>=0.5){

					return -pow(x-1,2.)*(2*x-1)*(10*x-7);
				}					
			}				
		}

	}


    //////////////
    std::cout << "parameter out of range in function phi_P1_L2_D1" << std::endl;
    exit(1); 

}


// P=1: 2 interpolation points
// L=2: every point has value, 1st-order derivative and 2nd_order derivative
// D=2: 2nd-order derivative of basis function
double HermBasis::phi_P1_L2_D2(const double x, int msh, const int pp)
{

    if (x<0 || x>1) return 0.;

    int p = 0; // index for point
    int l = 0; // index for function or derivative

	deg_pt_deri_1d(pp, p, l);

    /////////////////////

    if (msh==0){ // level 0
		
		if (p==0){ // left point
			
			if (l==0){ // function
				
				return -60*x*(x-1)*(2*x-1);
			}	
			else if (l==1){ // 1st-order derivative
				
				return -12*x*(x-1)*(5*x-3);
			}	
			else if (l==2){ // 2nd-order derivative
				
				return -(x-1)*(10*x*x-8*x+1);
			}			

		}			
		else if (p==1){ // right point
			
			if (l==0){

				return 60*x*(x-1)*(2*x-1);
			}	
			else if (l==1){
				
				return -12*x*(x-1)*(5*x-2);
			}	
			else if (l==2){
				
				return x*(10*x*x-12*x+3);
			}				
		}

	}
	else if (msh==1){ // level 1

		if (p==0){
			
			if (l==0){

				if (x<=0.5){
				 	
					return 480*x*(2*x-1)*(4*x-1);
				}
				else if (x>0.5){

					return 0.;
				}
			}	
			else if (l==1){
				
				if (x<=0.5){
				 	
					return -96*x*(2*x-1)*(5*x-1);
				}
				else if (x>0.5){

					return 0.;
				}					
			}	
			else if (l==2){
				
				if (x<=0.5){
				 	
					return 2*x*(40*x*x-24*x+3);
				}
				else if (x>0.5){

					return 0.;
				}					
			}				
		}			
		else if (p==1){
			
			if (l==0){
				
				if (x<0.5){
				 	
					return 0.;
				}
				else if (x>=0.5){

					return -480*(x-1)*(2*x-1)*(4*x-3);
				}
			}	
			else if (l==1){
				
				if (x<0.5){
				 	
					return 0.;
				}
				else if (x>=0.5){

					return -96*(x-1)*(2*x-1)*(5*x-4);
				}					
			}	
			else if (l==2){
				
				if (x<0.5){
				 	
					return 0.;
				}
				else if (x>=0.5){

					return -2*(x-1)*(40*x*x-56*x+19);
				}					
			}				
		}

	}

    //////////////
    std::cout << "parameter out of range in function phi_P1_L2_D2" << std::endl;
    exit(1); 

}