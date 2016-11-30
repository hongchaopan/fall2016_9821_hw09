#pragma once
#include "stdafx.h"
#include "Eigen"
#include <tuple>

using namespace std;
using namespace Eigen;
 
VectorXd soramer(MatrixXd early, double omega,
	const MatrixXd & A,
	const VectorXd & b,
	double tol,
	const VectorXd & x0
	= VectorXd::Zero(0));

tuple<VectorXd, int> sor(double omega,
	const MatrixXd & A,
	const VectorXd & b,
	double tol,
	const VectorXd & x0
	= VectorXd::Zero(0));

std::tuple<VectorXd, int> sor(double omega,
	const ArrayXXd & A, int m,
	const VectorXd & b,
	double tol,
	const VectorXd & x0
	= VectorXd::Zero(0));

std::tuple<VectorXd, int> gs(const MatrixXd & A,
	const VectorXd & b,
	double tol,
	const VectorXd & x0
	= VectorXd::Zero(0));

std::tuple<VectorXd, int> gs(const ArrayXXd & A, int m,
	const VectorXd & b,
	double tol,
	const VectorXd & x0
	= VectorXd::Zero(0));

