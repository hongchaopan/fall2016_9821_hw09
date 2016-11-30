#include "stdafx.h"
#include "down_out.hpp"
#include "solvefd.hpp"
#include "math.h"
#include "iostream"
#include "Eigen"

using namespace std;
using namespace Eigen;

down_out::down_out():K(40.0), T(0.75), r(0.05), sigma(0.3), q(0.03), s(42.0),B(35.0)
{
}

double down_out::Blacksholes() {
	double pi = 3.1415926535897;
	//int type = this->gettype();    //??? why char const char*
	double d11 = (log(s / K) + (r - q + sigma*sigma / 2)*T) / (sigma*sqrt(T));
	double d12 = d11 - sigma*sqrt(T);
	double Nd11 = erfc(-d11 / std::sqrt(2)) / 2;
	double Nd12 = erfc(-d12 / std::sqrt(2)) / 2;
	double BS1 = s*exp(-q*T)*Nd11 - K*exp(-r*T)*Nd12;
	double d21 = (log(B*B / s / K) + (r - q + sigma*sigma / 2)*T) / (sigma*sqrt(T));
	double d22 = d21 - sigma*sqrt(T);
	double Nd21 = erfc(-d21 / std::sqrt(2)) / 2;
	double Nd22 = erfc(-d22 / std::sqrt(2)) / 2;
	double BS2 = B*B/s*exp(-q*T)*Nd21 - K*exp(-r*T)*Nd22;
	double BS = BS1 - pow(B / s, 2.0*(r - q) / (sigma*sigma) - 1.0)*BS2;
	return BS;
}

VectorXd down_out::para(int M,double alpha_temp) {
	VectorXd p(6);
	p.setZero();
	double x_compute = log(s / K);
	//double x_left = log(s / K) + (r - q - sigma*sigma / 2)*T - 3 * sigma*sqrt(T);
	double x_left = log(B / K);
	double x_right_temp = log(s / K) + (r - q - sigma*sigma / 2)*T + 3 * sigma*sqrt(T);
	//cout << "x_right: " << x_right << endl;
	double t_final = T*sigma*sigma / 2.0;
	double dt = t_final / double(M);
	double dx_temp = sqrt(dt / alpha_temp);
	int N_left = floor((x_compute - x_left) / dx_temp);
	double dx = (x_compute - x_left) / double(N_left);
	double alpha = dt / (dx*dx);
	int N_right = ceil((x_right_temp - x_compute) / dx);
	int N = N_left + N_right;
	double x_right = x_left + double(N)*dx;
	p[0] = alpha;
	p[1] = x_left;
	p[2] = x_right;
	p[3] = N;
	p[4] = dx;
	p[5] = dt;
	return p;
}

VectorXd down_out::price(int M, double alpha_temp, int method, double w) {
	VectorXd value(5);
	value.setZero();
	double x_compute = log(s / K);
	//double x_left = log(s / K) + (r - q - sigma*sigma / 2)*T - 3 * sigma*sqrt(T);
	double x_left = log(B / K);
	double x_right_temp = log(s / K) + (r - q - sigma*sigma / 2)*T+3 * sigma*sqrt(T);
	//cout << "x_right: " << x_right << endl;
	double t_final = T*sigma*sigma / 2.0;
	double dt = t_final / double(M);
	double dx_temp = sqrt(dt / alpha_temp);
	int N_left = floor((x_compute - x_left)/dx_temp);
	double dx = (x_compute- x_left) / double(N_left);
	double alpha = dt / (dx*dx);
	int N_right = ceil((x_right_temp - x_compute) / dx);
	int N = N_left + N_right;
	double x_right = x_left + double(N)*dx;
	double a = 0.5 * (2.0 * (r - q) / (sigma*sigma) - 1.0);
	double b = 0.25 * (2.0 * (r - q) / (sigma*sigma) + 1.0)*(2.0 * (r - q) / (sigma*sigma) + 1.0)+2.0*q/(sigma*sigma);
	//cout << "alpha:" << alpha << endl;
	MatrixXd u(N + 1, M + 1);
	u.setZero();
	for (int i = 0; i <= N; i++) {
		double x = x_left + double(i)*dx;
		u(i, 0) = K*exp(a*x)*max(exp(x)-1, 0.0);
		//cout << u(i, 0) << endl;
	}
	for (int i = 0; i <= M; i++) {
		double t = double(i)*dt;
		u(N, i) = K*exp(a*x_right + b*t)*(exp(x_right - 2 * q*t / (sigma*sigma))-exp(-2 * r*t / (sigma*sigma)));
		//u(N, i) = K*exp(a*x_right + b*t)*(1 - exp(x_left));
		//cout << u(0, i) << endl;
	}
	/*MatrixXd early(N - 1, M);
	early.setZero();
	for (int j = 0; j <M; j++) {
		for (int i = 0; i < N - 1; i++) {
			double x = x_left + double(i)*dx + dx;
			double t = double(j)*dt + dt;
			early(i, j) = K*exp(a*x + b*t)*max(1.0 - exp(x), 0.0);
		}
	}*/
	//cout << "early done"<< endl;
	switch (method) {
	case 0:	u = ForwardEuler(M, N, alpha, u); break;
	case 1:	u = BackwardEulerbyLU(M, N, alpha, u); break;
	//case 2:	u = BackwardEulerbySOR(M, N, alpha, w, u); break;
	//case 3:	u = Crank_NicolsenbyLU(M, N, alpha, u); break;
	case 2:	u = Crank_NicolsenbySOR(M, N, w, alpha,u); break;
	}
	if ((method == 0) || (method == 1)) {
		if ((alpha_temp == 0.4)&(M==4) ){
			for (int i = 0; i <= M; i++) {
				for (int j = 0; j <= N; j++) cout << u(j, i) << " ";
				cout << endl;
			}
		}
	}
	int index = 0;
	for (int i = 0; i <= N; i++) {
		if ((x_left + double(i)*dx <= x_compute)&(x_left + double(i)*dx + dx >= x_compute))
			index = i;
	}
	//cout << "index:" <<index<< endl;
	double v0 = u(index - 1, M)*exp(-a*(x_left + index*dx - dx) - b*t_final);
	double v1 = u(index, M)*exp(-a*(x_left + index*dx) - b*t_final);
	double v2 = u(index + 1, M)*exp(-a*(x_left + index*dx + dx) - b*t_final);
	double v3 = u(index, M-1)*exp(-a*x_left- b*t_final+b*dt);
	//cout << "dx: " << dx << endl;
	//cout << "s0:" << K*exp(x_left) << endl;
	double s0 = K*exp(x_left + index*dx - dx);
	double s1 = K*exp(x_left + index*dx);
	double s2 = K*exp(x_left + index*dx + dx);
	double s3 = K*exp(x_left + index*dx + 2.0*dx);
	//cout << "s1:" << s1 <<" s2:" << s2 << endl;
	//cout << "v1:" << v1 <<" v2:" << v2 << endl;
	value[0] = u(index, M);
	value[1] =v1;
	value[2] = (v2 - v0) / (s2 - s0);
	value[3]= ((s1 - s0)*v2 - (s2 - s0)*v1 + (s2 - s1)*v0) / ((s1 - s0)*(s2 - s0)*(s2 - s1) / 2);
	value[4]= (v1 - v3) / (2 * dt / (sigma*sigma));
	return value;
}


down_out::~down_out()
{
}
