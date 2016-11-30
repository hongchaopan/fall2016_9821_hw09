#pragma once
#include "Eigen"

using namespace std;
using namespace Eigen;

class down_out
{
public:
	double K;
	double T;
	double r;
	double sigma;
	double q;
	double s;
	double B;
	down_out();
	double Blacksholes();
	VectorXd para(int,double);
	VectorXd price(int, double, int, double);
	~down_out();
};

