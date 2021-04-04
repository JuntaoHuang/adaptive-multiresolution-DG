#pragma once
#include "AllBasis.h"
#include "VecMultiD.h"


// OpertorMatrix1D calculates and stores the operations between the 1D basis;
// The basis can be of different types. 
template <class U, class V>
class OperatorMatrix1D
{
public:
	// boundary_type can be period, zero, inside
	// see details in Basis::product_edge_dis_v()
	OperatorMatrix1D(const AllBasis<U> & allbasis_U, const AllBasis<V> & allbasis_V, const std::string & boundary_type);
	~OperatorMatrix1D() {};

	const int row, col;
	const std::vector<int> size;	// equivalent to std::vector<int>{row, col}

	const std::string boundary;

	// for all u in allbasis_U and all v in allbasis_V
	// 1. mass matrix, store inner product of u and v
	VecMultiD<double> u_v;

	// 2. stiffness matrix, store inner product of u and v_x
	VecMultiD<double> u_vx;

	VecMultiD<double> ux_v;

	// 3. flux left limit matrix, store u^- * [v]
	VecMultiD<double> ulft_vjp;

	// u^- * v^-
	VecMultiD<double> ulft_vlft;
	
	// u^- * v^+
	VecMultiD<double> ulft_vrgt;

	// u^+ * v^-
	VecMultiD<double> urgt_vlft;
	
	// u^+ * v^+
	VecMultiD<double> urgt_vrgt;

	// u^+ * [v]
	VecMultiD<double> urgt_vjp;

	// [u] * v+
	VecMultiD<double> ujp_vrgt;

	// [u] * vx-
	VecMultiD<double> ujp_vxlft;

	// [u] * vx+
	VecMultiD<double> ujp_vxrgt;

	// u_x * v_x
	VecMultiD<double> ux_vx;
	
	// [u] * [v]
	VecMultiD<double> ujp_vjp;

	// uave * [v]
	VecMultiD<double> uave_vjp;

	// ux_ave * [v]
	VecMultiD<double> uxave_vjp;

	// [u] * vx_ave
	VecMultiD<double> ujp_vxave;

	// u_x * v at left boundary: x = 0
	VecMultiD<double> ux_v_bdrleft;

	// u_x * v at right boundary: x = 1
	VecMultiD<double> ux_v_bdrright;

	// u * v_x at left boundary: x = 0
	VecMultiD<double> u_vx_bdrleft;

	// u * v_x at right boundary: x = 1
	VecMultiD<double> u_vx_bdrright;

	// u * v at left boundary: x = 0
	VecMultiD<double> u_v_bdrleft;

	// u * v at right boundary: x = 1
	VecMultiD<double> u_v_bdrright;

	// kdv: stiffness matrix, store inner product of u and v_xxx
	VecMultiD<double> u_vxxx;

	// zk: inner product of u and v_xx
	VecMultiD<double> u_vxx;

	VecMultiD<double> uxx_v;

	// kdv: flux right limit matrix, store u_xx^+ * [v]
	VecMultiD<double> uxxrgt_vjp;

	// zk: u_x^+ * [v]
	VecMultiD<double> uxrgt_vjp;

	// u_x^- * [v]
	VecMultiD<double> uxlft_vjp;	

	// u^- * [v_x]
	VecMultiD<double> ulft_vxjp;

	// u^+ * [v_x]
	VecMultiD<double> urgt_vxjp;	

	// [u_x] * [v_x]
	VecMultiD<double> uxjp_vxjp;

	// u_x^+ * [v_x]
	VecMultiD<double> uxrgt_vxjp;

	// u^- * [v_xx]
	VecMultiD<double> ulft_vxxjp;	
};

template<class U, class V>
OperatorMatrix1D<U,V>::OperatorMatrix1D(const AllBasis<U> & allbasis_U, const AllBasis<V> & allbasis_V, const std::string & boundary_type):
	row(allbasis_U.size()), 
	col(allbasis_V.size()),
	size(std::vector<int>{row,col}),
	boundary(boundary_type),
	u_v(2, size),
	u_vx(2, size),
	ux_v(2, size),
	ulft_vjp(2, size),
	urgt_vjp(2, size),
	ujp_vrgt(2, size),
	ulft_vlft(2, size),
	ulft_vrgt(2, size),
	urgt_vlft(2, size),
	urgt_vrgt(2, size),
	ux_vx(2, size),
	ujp_vjp(2, size),
	uave_vjp(2, size),
	uxave_vjp(2, size),
	ujp_vxave(2, size),
	ux_v_bdrleft(2, size),
	ux_v_bdrright(2, size),
	u_vx_bdrleft(2, size),
	u_vx_bdrright(2, size),
	u_v_bdrleft(2, size),
	u_v_bdrright(2, size),
	ujp_vxlft(2, size),
	ujp_vxrgt(2, size),
	// kdv eqn needs
	u_vxxx(2, size),
	u_vxx(2, size),
	uxx_v(2, size),
	uxxrgt_vjp(2, size),
	uxrgt_vjp(2, size),
	uxlft_vjp(2, size),
	ulft_vxjp(2, size),
	urgt_vxjp(2, size),
	uxjp_vxjp(2, size),
	uxrgt_vxjp(2, size),
	ulft_vxxjp(2, size)	
{
	for (size_t row_idx = 0; row_idx < row; row_idx++)
	{
		for (size_t col_idx = 0; col_idx < col; col_idx++)
		{
			u_v.at(row_idx, col_idx) = allbasis_U.at(row_idx).product_volume(allbasis_V.at(col_idx));

			if constexpr(std::is_same_v<U, LagrBasis> && std::is_same_v<V, LagrBasis>) 
			{
				continue;
			}

			// if constexpr(std::is_same_v<U, HermBasis> && std::is_same_v<V, HermBasis>) {continue;}

			u_vx.at(row_idx, col_idx) = allbasis_U.at(row_idx).product_volume(allbasis_V.at(col_idx), 0, 1);
			ux_v.at(col_idx, row_idx) = u_vx.at(row_idx, col_idx);

			{
				const double ur_vr = allbasis_U.at(row_idx).product_edge_dis_v(allbasis_V.at(col_idx), 1, 1, 0, 0, boundary_type);
				const double ur_vl = allbasis_U.at(row_idx).product_edge_dis_v(allbasis_V.at(col_idx), 1, -1, 0, 0, boundary_type);
				const double ul_vr = allbasis_U.at(row_idx).product_edge_dis_v(allbasis_V.at(col_idx), -1, 1, 0, 0, boundary_type);
				const double ul_vl = allbasis_U.at(row_idx).product_edge_dis_v(allbasis_V.at(col_idx), -1, -1, 0, 0, boundary_type);

				ulft_vlft.at(row_idx, col_idx) = ul_vl;
				ulft_vrgt.at(row_idx, col_idx) = ul_vr;
				urgt_vlft.at(row_idx, col_idx) = ur_vl;
				urgt_vrgt.at(row_idx, col_idx) = ur_vr;

				ulft_vjp.at(row_idx, col_idx) = ul_vr - ul_vl;
				urgt_vjp.at(row_idx, col_idx) = ur_vr - ur_vl;
			}

			ujp_vjp.at(row_idx, col_idx) = urgt_vjp.at(row_idx, col_idx) - ulft_vjp.at(row_idx, col_idx);
			uave_vjp.at(row_idx, col_idx) = (urgt_vjp.at(row_idx, col_idx) + ulft_vjp.at(row_idx, col_idx))/2.;

			ulft_vxjp.at(row_idx, col_idx) = allbasis_U.at(row_idx).product_edge_dis_v(allbasis_V.at(col_idx), -1, 1, 0, 1, boundary_type)
											- allbasis_U.at(row_idx).product_edge_dis_v(allbasis_V.at(col_idx), -1, -1, 0, 1, boundary_type);

			urgt_vxjp.at(row_idx, col_idx) = allbasis_U.at(row_idx).product_edge_dis_v(allbasis_V.at(col_idx), 1, 1, 0, 1, boundary_type)
											- allbasis_U.at(row_idx).product_edge_dis_v(allbasis_V.at(col_idx), 1, -1, 0, 1, boundary_type);

			const double ur_vxr = allbasis_U.at(row_idx).product_edge_dis_u(allbasis_V.at(col_idx), 1, 1, 0, 1, boundary_type);
			const double ul_vxr = allbasis_U.at(row_idx).product_edge_dis_u(allbasis_V.at(col_idx), -1, 1, 0, 1, boundary_type);
			const double ur_vxl = allbasis_U.at(row_idx).product_edge_dis_u(allbasis_V.at(col_idx), 1, -1, 0, 1, boundary_type);
			const double ul_vxl = allbasis_U.at(row_idx).product_edge_dis_u(allbasis_V.at(col_idx), -1, -1, 0, 1, boundary_type);
			ujp_vxlft.at(row_idx, col_idx) = ur_vxl - ul_vxl;
			ujp_vxrgt.at(row_idx, col_idx) = ur_vxr - ul_vxr;

			{
				const double ur_vr = allbasis_U.at(row_idx).product_edge_dis_u(allbasis_V.at(col_idx), 1, 1, 0, 0, boundary_type);
				const double ul_vr = allbasis_U.at(row_idx).product_edge_dis_u(allbasis_V.at(col_idx), -1, 1, 0, 0, boundary_type);
				ujp_vrgt.at(row_idx, col_idx) = ur_vr - ul_vr;
			}

			// only calculate when u and v are both Alpert basis
			if constexpr(std::is_same_v<U, AlptBasis> && std::is_same_v<V, AlptBasis>)
			{
				ux_vx.at(row_idx, col_idx) = allbasis_U.at(row_idx).product_volume(allbasis_V.at(col_idx), 1, 1);
				const double uxl_vr = allbasis_U.at(row_idx).product_edge_dis_v(allbasis_V.at(col_idx), -1, 1, 1, 0, boundary_type);
				const double uxl_vl = allbasis_U.at(row_idx).product_edge_dis_v(allbasis_V.at(col_idx), -1, -1, 1, 0, boundary_type);
				const double uxr_vr = allbasis_U.at(row_idx).product_edge_dis_v(allbasis_V.at(col_idx), 1, 1, 1, 0, boundary_type);
				const double uxr_vl = allbasis_U.at(row_idx).product_edge_dis_v(allbasis_V.at(col_idx), 1, -1, 1, 0, boundary_type);
				uxrgt_vjp.at(row_idx, col_idx) = uxr_vr - uxr_vl;
				uxlft_vjp.at(row_idx, col_idx) = uxl_vr - uxl_vl;
				uxave_vjp.at(row_idx, col_idx) = (uxlft_vjp.at(row_idx, col_idx) + uxrgt_vjp.at(row_idx, col_idx))/2.;
				ujp_vxave.at(col_idx, row_idx) = uxave_vjp.at(row_idx, col_idx);

				// boundary terms
				ux_v_bdrleft.at(row_idx, col_idx) = allbasis_U.at(row_idx).product_boundary(allbasis_V.at(col_idx), 1, 0, -1);
				ux_v_bdrright.at(row_idx, col_idx) = allbasis_U.at(row_idx).product_boundary(allbasis_V.at(col_idx), 1, 0, 1);

				u_vx_bdrleft.at(col_idx, row_idx) = ux_v_bdrleft.at(row_idx, col_idx);
				u_vx_bdrright.at(col_idx, row_idx) = ux_v_bdrright.at(row_idx, col_idx);

				// for kdv equation
				u_vxxx.at(row_idx, col_idx) = allbasis_U.at(row_idx).product_volume(allbasis_V.at(col_idx), 0, 3);
				u_vxx.at(row_idx, col_idx) = allbasis_U.at(row_idx).product_volume(allbasis_V.at(col_idx), 0, 2);
				uxx_v.at(row_idx, col_idx) = u_vxx.at(col_idx, row_idx);

				const double uxxr_vr = allbasis_U.at(row_idx).product_edge_dis_v(allbasis_V.at(col_idx), 1, 1, 2, 0, boundary_type);
				const double uxxr_vl = allbasis_U.at(row_idx).product_edge_dis_v(allbasis_V.at(col_idx), 1, -1, 2, 0, boundary_type);
				uxxrgt_vjp.at(row_idx, col_idx) = uxxr_vr - uxxr_vl;

				const double uxl_vxl = allbasis_U.at(row_idx).product_edge_dis_v(allbasis_V.at(col_idx), -1, -1, 1, 1, boundary_type);
				const double uxr_vxl = allbasis_U.at(row_idx).product_edge_dis_v(allbasis_V.at(col_idx), 1, -1, 1, 1, boundary_type);
				const double uxl_vxr = allbasis_U.at(row_idx).product_edge_dis_v(allbasis_V.at(col_idx), -1, 1, 1, 1, boundary_type);
				const double uxr_vxr = allbasis_U.at(row_idx).product_edge_dis_v(allbasis_V.at(col_idx), 1, 1, 1, 1, boundary_type);
				uxjp_vxjp.at(row_idx, col_idx) = uxr_vxr+uxl_vxl-uxr_vxl-uxl_vxr;
				uxrgt_vxjp.at(row_idx, col_idx) = uxr_vxr - uxr_vxl;

				const double ul_vxxr = allbasis_U.at(row_idx).product_edge_dis_v(allbasis_V.at(col_idx), -1, 1, 0, 2, boundary_type);
				const double ul_vxxl = allbasis_U.at(row_idx).product_edge_dis_v(allbasis_V.at(col_idx), -1, -1, 0, 2, boundary_type);
				ulft_vxxjp.at(row_idx, col_idx) = ul_vxxr - ul_vxxl;
			}

			// boundary terms
			u_v_bdrleft.at(row_idx, col_idx) = allbasis_U.at(row_idx).product_boundary(allbasis_V.at(col_idx), 0, 0, -1);
			u_v_bdrright.at(row_idx, col_idx) = allbasis_U.at(row_idx).product_boundary(allbasis_V.at(col_idx), 0, 0, 1);
		}		
	}	
}