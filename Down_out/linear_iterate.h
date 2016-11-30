#pragma once
#include "stdafx.h"
#include "Eigen"
#include <tuple>

using namespace Eigen;
enum { MAX_ITERATION = 1000000 };

Eigen::VectorXd linear_iterate_triangularamer
(MatrixXd early,const Eigen::VectorXd & x0,
	const Eigen::MatrixXd & M,
	const Eigen::MatrixXd & N,
	const Eigen::VectorXd & b,
	double tol);

// x0: initial vector 
// M: a lower triangular (or diagonal) matrix
// N: a matrix, not necessarily triangular or banded
// b: a right-hand side vector
// tol: tolerance factor
// return: solution to Mx <- b-Nx
std::tuple<Eigen::VectorXd, int> linear_iterate_triangular
(const Eigen::VectorXd & x0,
	const Eigen::MatrixXd & M,
	const Eigen::MatrixXd & N,
	const Eigen::VectorXd & b,
	double tol);
// x0: initial vector 
// d: a matrix diagonal vector
// N: a matrix, not necessarily triangular or banded
// b: a right-hand side vector
// tol: tolerance factor
// return: solution to Dx <- b-Nx
std::tuple<Eigen::VectorXd, int> linear_iterate_diagonal
(const Eigen::VectorXd & x0,
	const Eigen::VectorXd & d,
	const Eigen::MatrixXd & N,
	const Eigen::VectorXd & b,
	double tol);

// x0: initial vector 
// M: a lower triangular (or diagonal) matrix, banded
// N: an upper triangular matrix, banded
// m: band width of matrix M and N
// b: a right-hand side vector
// tol: tolerance factor
// return: solution to Mx <- b-Nx
std::tuple<Eigen::VectorXd, int> linear_iterate_triangular_banded
(const Eigen::VectorXd & x0,
	const Eigen::ArrayXXd & M,
	const Eigen::ArrayXXd & N,
	const Eigen::VectorXd & b,
	int m, double tol);

// x0: initial vector 
// d: a matrix diagonal vector
// N: a matrix, banded, not necessarily triangular
// b: a right-hand side vector
// tol: tolerance factor
// return: solution to Dx <- b-Nx
std::tuple<Eigen::VectorXd, int> linear_iterate_diagonal_banded
(const Eigen::VectorXd & x0,
	const Eigen::ArrayXd & d,
	const Eigen::ArrayXXd & N,
	const Eigen::VectorXd & b,
	int m, double tol);
