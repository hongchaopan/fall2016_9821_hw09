#include "stdafx.h"
#include "Eigen"
#include "solvefd.hpp"
#include "iostream"
#include "vector"
#include "Eigen"
#include "math.h"
#include "lu.h"
#include "sor.h"
#include "tuple"
#include "triangular_solve.h"


using namespace std;
using namespace Eigen;

MatrixXd ForwardEuler(int M, int N, double alpha, MatrixXd &u) {
	for (int j = 1; j <= M; j++) {
		for (int i = 1; i <N; i++) {
			u(i, j) = alpha*u(i + 1, j - 1) + (1.0 - 2.0 * alpha)*u(i, j - 1) + alpha*u(i - 1, j - 1);
		}
	}
	/*for (int i = 1; i < N; i++) {
		u(i, M) = max(u(i, M), early(i - 1, M - 1));
	}*/
	return u;
}

MatrixXd getalphaEU(double alpha, int N) {
	//alpha = 0.25*alpha;
	MatrixXd Malpha(N - 1, N - 1);
	Malpha.setZero();
	Malpha(0, 0) = 1.0 + 2.0 * alpha;
	Malpha(0, 1) = -alpha;
	Malpha(N - 2, N - 2) = 1.0 + 2.0 * alpha;
	Malpha(N - 2, N - 3) = -alpha;
	for (int i = 1; i <N - 2; i++) {
		Malpha(i, i - 1) = -alpha;
		Malpha(i, i) = 1.0 + 2.0 * alpha;
		Malpha(i, i + 1) = -alpha;
	}
	return Malpha;
}

MatrixXd BackwardEulerbyLU(int M, int N, double alpha, MatrixXd &u) {
	MatrixXd Malpha = getalphaEU(alpha, N);
	for (int i = 1; i <= M; i++) {
		//VectorXd ul(N - 1);
		VectorXd ur(N - 1);
		//ul.setZero();
		ur.setZero();
		//ul(0) = u(0, i);
		//ul(N - 2) = u(N, i);
		ur(0) = -alpha*u(0, i);
		ur(N - 2) = -alpha*u(N, i);
		VectorXd ui(N - 1);
		ui.setZero();
		ui = u.block(1, i - 1, N - 1, 1);
		ui = ui - ur;
		tuple<MatrixXd, MatrixXd> tu = lu_no_pivoting(Malpha);
		MatrixXd L = get<0>(tu);
		MatrixXd U = get<1>(tu);
		u.block(1, i, N - 1, 1) = forward_subst(L, ui);
		//for (int j = 0; j < N - 1; j++) cout << "u" << j << ": " << u(j, i) << endl;
		u.block(1, i, N - 1, 1) = backward_subst(U, u.block(1, i, N - 1, 1));
		//for (int j = 0; j < N - 1; j++) cout << "u" << j << ": " << u(j, i) << endl;
	}
	return u;
}

MatrixXd BackwardEulerbySOR(int M, int N, double alpha, double w, MatrixXd &u, MatrixXd early) {
	MatrixXd Malpha = getalphaEU(alpha, N);
	for (int i = 1; i <= M; i++) {
		//VectorXd ul(N - 1);
		VectorXd ur(N - 1);
		//ul.setZero();
		ur.setZero();
		//ul(0) = u(0, i);
		//ul(N - 2) = u(N, i);
		ur(0) = -alpha*u(0, i);
		ur(N - 2) = -alpha*u(N, i);
		VectorXd ui(N - 1);
		ui.setZero();
		ui = u.block(1, i - 1, N - 1, 1);
		ui = ui - ur;
		u.block(1, i, N - 1, 1) = get<0>(sor(w, Malpha, ui, 1e-10, u.block(1, i, N - 1, 1)));
	}
	return u;
}


tuple<MatrixXd, MatrixXd> getalpha(double alpha, int N) {
	MatrixXd Malpha(N - 1, N - 1);
	MatrixXd Malphal(N - 1, N - 1);
	Malpha.setZero();
	Malphal.setZero();
	Malpha(0, 0) = 2.0 - 2.0 * alpha;
	Malpha(0, 1) = alpha;
	Malpha(N - 2, N - 2) = 2.0 - 2.0 * alpha;
	Malpha(N - 2, N - 3) = alpha;
	for (int i = 1; i <N - 2; i++) {
		Malpha(i, i - 1) = alpha;
		Malpha(i, i) = 2.0 - 2.0 * alpha;
		Malpha(i, i + 1) = alpha;
	}
	Malphal.setZero();
	Malphal(0, 0) = 2.0 + 2.0 * alpha;
	Malphal(0, 1) = -alpha;
	Malphal(N - 2, N - 2) = 2.0 + 2.0 * alpha;
	Malphal(N - 2, N - 3) = -alpha;
	for (int i = 1; i < N - 2; i++) {
		Malphal(i, i - 1) = -alpha;
		Malphal(i, i) = 2.0 + 2.0 * alpha;
		Malphal(i, i + 1) = -alpha;
	}
	tuple<MatrixXd, MatrixXd> ta = make_tuple(Malpha, Malphal);
	return ta;
}


MatrixXd Crank_NicolsenbyLU(int M, int N, double alpha, MatrixXd &u, MatrixXd early) {
	MatrixXd Malpha = get<0>(getalpha(alpha, N));
	MatrixXd Malphal = get<1>(getalpha(alpha, N));
	for (int i = 1; i <= M; i++) {
		VectorXd ul(N - 1);
		VectorXd ur(N - 1);
		ul.setZero();
		ur.setZero();
		ul(0) = -alpha*u(0, i);
		ul(N - 2) = -alpha*u(N, i);
		ur(0) = alpha*u(0, i - 1);
		ur(N - 2) = alpha*u(N, i - 1);
		VectorXd ui(N - 1);
		ui.setZero();
		ui = u.block(1, i - 1, N - 1, 1);
		VectorXd b = ur - ul + Malpha*ui;
		tuple<MatrixXd, MatrixXd> tu = lu_no_pivoting(Malphal);
		MatrixXd L = get<0>(tu);
		MatrixXd U = get<1>(tu);
		u.block(1, i, N - 1, 1) = forward_subst(L, b);
		u.block(1, i, N - 1, 1) = backward_subst(U, u.block(1, i, N - 1, 1));
		//for (int j = 0; j < N - 1; j++) cout << "u" << j << ": " << u(j, i) << endl;
	}
	return u;
}

VectorXd soramer(double w, double alpha, double tol, VectorXd x0,VectorXd b,int N,double u0,double un) {
	VectorXd x_new(N);
	x_new.setZero();
	VectorXd early = x0;
	//for (int i = 0; i < N - 1; i++) cout << early(i) << endl;
	for (int i = 1; i <N-1; i++) {
		x_new(i) = (1 - w)*x0(i) + w*alpha / (2 * (1 + alpha))*(x0(i-1) + x0(i + 1)) + w / (1 + alpha)*b(i);
		x_new(i) = max(x_new(i), early(i));
	}
	x_new(0) = (1 - w)*x0(0) + w*alpha / (2 * (1 + alpha))*(u0 + x0(1)) + w / (1 + alpha)*b(0);
	x_new(0) = max(x_new(0), early(0));
	x_new(N-1) = (1 - w)*x0(N-1) + w*alpha / (2 * (1 + alpha))*(x0(N-2) +un) + w / (1 + alpha)*b(N-1);
	x_new(N-1) = max(x_new(N-1), early(N-1));
	//cout << "sornew" << endl;
	while ((x_new - x0).norm() > tol) {
		x0 = x_new;
		//for (int i = 0; i < N - 1; i++) cout <<x0(i) << endl;
		for (int i = 1; i <N - 1; i++) {
			x_new(i) = (1 - w)*x0(i) + w*alpha / (2 * (1 + alpha))*(x0(i - 1) + x0(i + 1)) + w / (1 + alpha)*b(i);
			x_new(i) = max(x_new(i), early(i));
		}
		x_new(0) = (1 - w)*x0(0) + w*alpha / (2 * (1 + alpha))*(u0 + x0(1)) + w / (1 + alpha)*b(0);
		x_new(0) = max(x_new(0), early(0));
		x_new(N - 1) = (1 - w)*x0(N - 1) + w*alpha / (2 * (1 + alpha))*(x0(N - 2) + un) + w / (1 + alpha)*b(N - 1);
		x_new(N - 1) = max(x_new(N - 1), early(N - 1));
	}
	return x_new;
}

MatrixXd Crank_NicolsenbySOR(int M, int N, double w, double alpha, MatrixXd &u) {
	MatrixXd Malpha = get<0>(getalpha(alpha, N));
	MatrixXd Malphal = get<1>(getalpha(alpha, N));
	for (int i = 1; i <=M; i++) {
		VectorXd ul(N - 1);
		VectorXd ur(N - 1);
		ul.setZero();
		ur.setZero();
		ul(0) = -alpha*u(0, i);
		ul(N - 2) = -alpha*u(N, i);
		ur(0) = alpha*u(0, i - 1);
		ur(N - 2) = alpha*u(N, i - 1);
		VectorXd ui(N - 1);
		ui.setZero();
		ui = u.block(1, i - 1, N - 1, 1);
		VectorXd b = ur - ul + Malpha*ui;
		//cout << "startsoramer" << endl;
		u.block(1, i, N-1, 1) =get<0>(sor(w,Malphal,b, 1e-6));
	}
	//u.block(1, M, N - 1, 1) = soramer(w, alpha, 1e-6, early.col(M - 1), u.block(1, M - 1, N - 1, 1), N - 1, u(0,M), u(N,M));
	return u;
}


