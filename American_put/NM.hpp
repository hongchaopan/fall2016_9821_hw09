#ifndef NM_HPP
#define NM_HPP
#include<boost/tuple/tuple.hpp>
//#include<Eigen\Dense>
#include <Eigen/Dense>
#include<iostream>

using namespace Eigen;
using namespace std;

VectorXd forward_subst(const MatrixXd& L, const VectorXd& b)
{
	int n = L.rows();
	VectorXd x(n);
	double sum;
	x(0) = b(0) / L(0, 0);

	for (int j = 1; j < n ; j++)
	{
		sum = 0;
		for (int k = 0; k < j ; k++)
		{
			sum += L(j, k)*x(k);
		}
		x(j) = (b(j) - sum) / L(j, j);
	}
	return x;
}

VectorXd backward_subst(const MatrixXd& U, const VectorXd& b)
{
	int n = U.rows(); 
	VectorXd x(n); 
	double sum; 
	x(n - 1) = b(n - 1) / U(n - 1, n - 1); 

	for (int j = n - 2; j >= 0; j--)
	{
		sum = 0.0; 
		for (int k = j+1; k < n; ++k)
		{
			sum += U(j, k)*x(k);
		}
		x(j) = (b(j) - sum) / U(j, j);
	}
	return x; 
}

tuple<MatrixXd, MatrixXd> lu_no_pivoting(MatrixXd A)
{
	int n = A.rows(); 
	MatrixXd L(n, n), U(n, n); 
	L.setZero(); U.setZero();

	for (int i = 0; i <= n - 1; i++)
	{
		for (int k = i; k < n; ++k)
		{
			U(i, k) = A(i, k);
			L(k, i) = A(k, i) / U(i, i); 
		}
		A.block(i + 1, i + 1, n - 1 - i, n - 1 - i) -= L.block(i + 1, i, n - 1 - i, 1) * U.block(i, i + 1, 1, n - 1 - i);
	}
	L(n - 1, n - 1) = 1; U(n - 1, n - 1) = A(n - 1, n - 1);
	auto x = make_tuple(L, U); 
	return x; 
}

VectorXd linear_solver_no_pivoting(const MatrixXd& A, const VectorXd& b)
{
	auto p = lu_no_pivoting(A); 
	MatrixXd L = get<0>(p); 
	MatrixXd U = get<1>(p); 
	VectorXd y(A.rows()); 
	y = forward_subst(L, b); 
	VectorXd x(A.rows()); 
	x = backward_subst(U, y);
	return x; 
}

tuple<MatrixXd, MatrixXd, MatrixXd> lu_row_pivoting(MatrixXd A)
{
	int n = A.rows(); 
	MatrixXd L(n, n), U(n, n); 
	L.setIdentity(); U.setIdentity(); 
	MatrixXd P(n, n); 
	P.setIdentity(); 
	ArrayXd::Index r, c; 
	for (int i = 0; i < n; i++)
	{
		A.block(i, i, n - i, 1).array().abs().maxCoeff(&r, &c);
		r += i; 
		A.row(r).swap(A.row(i)); 
		P.row(r).swap(P.row(i)); 
		if (i > 0)	L.block(i, 0, 1, i).swap(L.block(r, 0, 1, i)); 
		
		for (int k = i; k < n; k++)
		{
			U(i, k) = A(i, k); 
			L(k, i) = A(k, i) / U(i, i);
		}
		A.block(i + 1, i + 1, n - 1 - i, n - 1 - i) -= L.block(i + 1, i, n - 1 - i, 1) * U.block(i, i + 1, 1, n - 1 - i);
	}
	L(n - 1, n - 1) = 1; U(n - 1, n - 1) = A(n - 1, n - 1); 
	auto x = make_tuple(L, U, P);
	return x;
}

VectorXd linear_solver_row_pivoting(MatrixXd A, VectorXd b)
{
	auto x = lu_row_pivoting(A);
	MatrixXd L = get<0>(x); 
	MatrixXd U = get<1>(x); 
	MatrixXd P = get<2>(x); 
	VectorXd bPrime = P * b;
	VectorXd y(A.rows()); 
	y = forward_subst(L, bPrime); 
	VectorXd x1 = backward_subst(U, y); 
	return x1; 
}
//Cubic spline interpolation 
tuple<VectorXd, VectorXd, VectorXd, VectorXd> cubic_spline_interpolation(VectorXd x, VectorXd v)
{
	int n = x.size() - 1;
	double p1, p2, p3, p4; 
	VectorXd a(n), b(n), c(n), d(n);
	VectorXd z(n - 1);
	for (int i = 0; i < n - 1; i++)
	{
		p1 = v(i + 2) - v(i+1);
		p2 = x(i + 2) - x(i+1);
		p3 = v(i+1) - v(i);
		p4 = x(i+1) - x(i);
		z(i) = 6 * ((p1 / p2) - (p3 / p4));
	}
	MatrixXd M(n - 1, n - 1); 
	M.setZero(); 
	for (int i = 0; i < n - 1; i++) { M(i, i) = 2 * (x(i + 2) - x(i));  }
	for (int i = 0; i < n - 2; i++) { M(i, i + 1) = x(i + 2) - x(i + 1);  } 
	for (int i = 1; i < n - 1; i++) { M(i, i - 1) = x(i + 1) - x(i); }
	VectorXd w1(n-1); w1 = linear_solver_no_pivoting(M, z);
	VectorXd w(n+1); 
	w(0) = 0; for (int i = 0; i < n - 1; i++) { w(i + 1) = w1(i); }
	w(n) = 0; 

	for (int i = 0; i < n; i++)
	{
		p1 = w(i)*x(i + 1) - w(i + 1)*x(i); 
		p2 = 2 * (x(i + 1) - x(i)); 
		p3 = w(i + 1) - w(i); 
		p4 = 6 * (x(i + 1) - x(i));
		c(i) = p1 / p2; 
		d(i) = p3 / p4; 
	}
	VectorXd q(n), r(n);
	for (int i = 0; i < n; i++)
	{
		q(i) = v(i) - c(i) *x(i)*x(i) - d(i)*x(i)*x(i)*x(i); 
		r(i) = v(i + 1) - c(i) * x(i + 1)*x(i + 1) - d(i) * x(i + 1)*x(i + 1)*x(i + 1);
	}
	for (int i = 0; i < n; i++)
	{
		a(i) = (q(i)*x(i + 1) - r(i)*x(i)) / (x(i + 1) - x(i)); 
		b(i) = (r(i) - q(i)) / (x(i + 1) - x(i));
	}
	tuple<VectorXd, VectorXd, VectorXd, VectorXd> ret(a, b, c, d); 
	return ret; 
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Iteration solvers
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
tuple<MatrixXd, MatrixXd, MatrixXd> split(MatrixXd A)
{
	int n = A.rows();
	MatrixXd L(n, n), D(n, n), U(n, n);
	L.setZero(); U.setZero(); D.setZero();
	for (int j = 0; j < n; j++)
	{
		for (int k = 0; k < n; k++)
		{
			if (j == k) D(j, k) = A(j, k);
			if (j > k) L(j, k) = A(j, k); 
			if (j < k) U(j, k) = A(j, k);

		}
	}
	return make_tuple(L, D, U);
}
//jacobi 
tuple<VectorXd, int> jacobi_solver(const MatrixXd& A, const VectorXd& b, const VectorXd& x0, double tol, bool criterion)
{
	auto s = split(A); 
	MatrixXd L = get<0>(s); 
	MatrixXd D = get<1>(s);
	MatrixXd U = get<2>(s);
	VectorXd b_new = D.inverse()*b; 
	VectorXd x_old = x0; 
	VectorXd x_new = x0; 
	VectorXd x = x0; 
	int iteration = 0; 
	if (criterion == true)
	{
		double r0 = (b - A * x0).norm();

		double r = (b - A * x_new).norm();

		while (r >= (tol * r0))
		{
			x_new = -D.inverse()*(L*x_old + U*x_old) + b_new;


			r = (b - A * x_new).norm();

			x_old = x_new;

			iteration++;
			cout << iteration << endl;
		};

		x = x_new;

	}
	else
	{
		double approximation_difference = 2 * tol;

		while (approximation_difference >= tol)
		{
			x_new = -D.inverse()*(L*x_old + U*x_old) + b_new;

			approximation_difference = (x_new - x_old).norm();

			x_old = x_new;

			iteration++;
		};

		x = x_new;
	}
	return make_tuple(x, iteration);

}

// gauss sidel
tuple<VectorXd, int> gauss_sidel_solver(const MatrixXd& A, const MatrixXd& b, const MatrixXd& x0, double tol, bool criterion)
{
	auto s = split(A);
	MatrixXd L = get<0>(s);
	MatrixXd D = get<1>(s);
	MatrixXd U = get<2>(s);
	VectorXd b_new = forward_subst(D + L, b);
	VectorXd x_old = x0; 
	VectorXd x_new = x0; 
	VectorXd x = x0; 
	int iteration = 0; 
	if (criterion == true)
	{
		double r0 = (b - A * x0).norm();

		double r = (b - A * x_new).norm();

		while (r >= (tol * r0))
		{
			x_new = -forward_subst(D + L, U * x_old) + b_new;
			r = (b - A * x_new).norm();

			x_old = x_new;

			iteration++;
		};

		x = x_new;
	}
	else
	{
		double approximation_difference = 2 * tol;

		while (approximation_difference >= tol)
		{
			x_new = -forward_subst(D + L, U * x_old) + b_new;

			approximation_difference = (x_new - x_old).norm();

			x_old = x_new;

			iteration++;
		};

		x = x_new;
	}
	return make_tuple(x, iteration);
}


// SOR solver
tuple<VectorXd, int> SOR_solver(const MatrixXd& A, const VectorXd& b, const VectorXd& x0, const double& tol, const bool& criterion, const double& omega)
{
	auto s = split(A); 
	MatrixXd L = get<0>(s);
	MatrixXd D = get<1>(s);
	MatrixXd U = get<2>(s);
	VectorXd b_new = omega * forward_subst(D + omega*L, b);
	VectorXd x_old = x0; 
	VectorXd x_new = x0; 
	VectorXd x = x0; 
	int iteration = 0; 
	if (criterion == true)
	{
		double r0 = (b - A * x0).norm();

		double r = (b - A * x_new).norm();

		while (r >= (tol * r0))
		{
			x_new = forward_subst(D + omega * L, ((1 - omega) * D - omega * U)* x_old) + b_new;


			r = (b - A * x_new).norm();

			x_old = x_new;

			iteration++;
		};

		x = x_new;
	}
	else
	{
		double approximation_difference = 2 * tol;

		while (approximation_difference >= tol)
		{
			x_new = forward_subst(D + omega * L, ((1 - omega) * D - omega * U)* x_old) + b_new;

			approximation_difference = (x_new - x_old).norm();

			x_old = x_new;

			iteration++;
		};

		x = x_new;

	}

	return make_tuple(x, iteration);
}

/* Define Stop Criterion
- Residual-based stopping criterion
- Consecutive approximation stopping criterion*/
enum class Criterion_Type { Resiudal_Based, Consec_Approx };

class StopCriterion {
public:
	StopCriterion(double _tol, Criterion_Type _type, const VectorXd& _r0)
		: tol(_tol), stop_iter_residual(_tol*_r0.norm()), type(_type) {}

	bool operator()(const VectorXd& x_old, const VectorXd& x_new, const VectorXd& r) {

		if (type == Criterion_Type::Resiudal_Based)
			return r.norm() > stop_iter_residual;
		else {
			return (x_old - x_new).norm() > tol;
		}
	}

private:
	double tol;
	double stop_iter_residual;
	Criterion_Type type;
};



std::tuple<VectorXd, int> linear_solve_sor_iter_proj(const MatrixXd & A, const VectorXd & b, const VectorXd & x0,
	double w, double tol, Criterion_Type type, std::function<double(double) > cons) {
	assert(w > 0 && w < 2);
	VectorXd x_new = x0;
	// Init value of x_old that dissatisfy stop criterion
	VectorXd x_old = x_new + VectorXd::Constant(x0.size(), tol);
	VectorXd r = b - A * x0;
	MatrixXd D(A.diagonal().asDiagonal());
	MatrixXd U = A.triangularView<Eigen::StrictlyUpper>();
	MatrixXd L = A.triangularView<Eigen::StrictlyLower>();
	VectorXd b_new = w * forward_subst(D + w*L, b);
	int ic = 0;

	StopCriterion stop_crtr(tol, type, r);

	while (stop_crtr(x_old, x_new, r)) {
		x_old = x_new;
		for (int j = 0; j<A.rows(); ++j) {
			double temp = 0;
			for (int k = 0; k<A.rows(); ++k) {
				if (k<j)
					temp = temp + A(j, k)*x_new(k);
				else if (k>j)
					temp = temp + A(j, k)*x_old(k);
			}
			x_new(j) = (1 - w)*x_old(j) - w / A(j, j)*temp + w*b(j) / A(j, j);
			x_new(j) = std::max(x_new(j), cons(j));
		}
		//        x_new = forward_subst(D + w*L, (1 - w) * D * x_old - w * U * x_old) + b_new;
		//        for (int i = 0; i < x_new.rows(); ++i){
		//            x_new(i) = std::max(x_new(i), cons(i));
		//        }
		ic++;
	}
	return std::make_tuple(x_new, ic);
}


std::tuple<VectorXd, int> linear_solve_sor_iter(const MatrixXd & A, const VectorXd & b, const VectorXd & x0,
	double w, double tol, Criterion_Type type) {
	assert(w > 0 && w < 2);
	VectorXd x_new = x0;
	// Init value of x_old that dissatisfy stop criterion
	VectorXd x_old = x_new + VectorXd::Constant(x0.size(), tol);
	VectorXd r = b - A * x0;
	MatrixXd D(A.diagonal().asDiagonal());
	MatrixXd U = A.triangularView<Eigen::StrictlyUpper>();
	MatrixXd L = A.triangularView<Eigen::StrictlyLower>();
	VectorXd b_new = w * forward_subst(D + w*L, b);
	int ic = 0;

	StopCriterion stop_crtr(tol, type, r);

	while (stop_crtr(x_old, x_new, r)) {
		x_old = x_new;
		x_new = forward_subst(D + w*L, (1 - w) * D * x_old - w * U * x_old) + b_new;
		ic++;
	}
	return std::make_tuple(x_new, ic);
}


#endif