#pragma once
#include "Eigen"

using namespace std;
using namespace Eigen;

MatrixXd ForwardEuler(int M, int N, double alpha, MatrixXd &u);
MatrixXd BackwardEulerbyLU(int M, int N, double alpha, MatrixXd &u);
MatrixXd BackwardEulerbySOR(int M, int N, double alpha, double w, MatrixXd &u, MatrixXd);
MatrixXd Crank_NicolsenbyLU(int M, int N, double alpha, MatrixXd &u, MatrixXd);
MatrixXd Crank_NicolsenbySOR(int M, int N, double w, double alpha, MatrixXd &u);