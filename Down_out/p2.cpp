#include "stdafx.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <cmath>
#include <cassert>
#include <utility>
#include <vector>
#include "American.hpp"
#include "down_out.hpp"
#include "Eigen"

using namespace std;
using namespace Eigen;


VectorXd computeApproximationError(const std::vector<double> & u,double xl, double xr)
{
	VectorXd error(2);
	error.setZero();
	int N = u.size() - 1;
	double dx = (xr - xl) / N;

	double maxPointwiseError = 0.0;
	double rmsError = 0.0;
	for (int i = 1; i<N; i++) {
		double x = xl + i*dx;
		double uValue =exp(x+1);
		double diff = fabs(u[i] - uValue);
		if (diff > maxPointwiseError) { maxPointwiseError = diff; }
		rmsError += (diff*diff)/(uValue*uValue);
	}
	rmsError /= double(N + 1);
	rmsError = sqrt(rmsError);
	error[0] = maxPointwiseError;
	error[1] = rmsError;
	return error;
	/*cout << setprecision(9)
		<< maxPointwiseError << ","
		<< rmsError
		<< endl;*/
}

int main()
{
	//American amer;
	down_out downout;
	double exact = downout.Blacksholes();
	cout << "exactprice:" << exact<< endl;
	int method[5] = { 0,1,2};
	int M[4] = { 4,16,64,256};
	double alpha[2] = { 0.4,4.0};
	//int N[] = {};
	double w = 1.2;
	for (int j = 0; j < 2; j++) {
		for (int i= 0; i< 4; i++) {
			VectorXd p = downout.para(M[i],alpha[j]);
			for (int k = 0; k < 6; k++) {
				cout << setprecision(9) << p[k] << " ";
			}
			cout << endl;
		}
	}
	cout << "ForwardEuler,alpha=" << alpha[0] << endl;
	for (int i = 0; i <4; i++) {
		//cout << "M:" << M[i] << endl;
		VectorXd value = downout.price(M[i], alpha[0], method[0], w);
		for(int j=0;j<2;j++)	cout <<setprecision(9)<<value[j]<<" ";
		cout << setprecision(9) << abs(value[1]-exact) << " ";
		for (int j = 2; j<5; j++)	cout << setprecision(9) << value[j] << " ";
		cout << endl;
	}
	cout << "BackwardEulerbyLU,alpha=" << alpha[0] << endl;
	for (int i = 0; i <4; i++) {
		//cout << "M:" << M[i] << endl;
		VectorXd value = downout.price(M[i], alpha[0], method[0], w);
		for (int j = 0; j<2; j++)	cout << setprecision(9) << value[j] << " ";
		cout << setprecision(9) << abs(value[1] - exact) << " ";
		for (int j = 2; j<5; j++)	cout << setprecision(9) << value[j] << " ";
		cout << endl;
	}
	cout << "BackwardEulerbyLU,alpha=" << alpha[1] << endl;
	for (int i = 0; i <4; i++) {
		//cout << "M:" << M[i] << endl;
		VectorXd value = downout.price(M[i], alpha[0], method[0], w);
		for (int j = 0; j<2; j++)	cout << setprecision(9) << value[j] << " ";
		cout << setprecision(9) << abs(value[1] - exact) << " ";
		for (int j = 2; j<5; j++)	cout << setprecision(9) << value[j] << " ";
		cout << endl;
	}
	cout << "CrankNicolsenbySOR,alpha=" << alpha[0] << endl;
	for (int i = 0; i <4; i++) {
		//cout << "M:" << M[i] << endl;
		VectorXd value = downout.price(M[i], alpha[0], method[0], w);
		for (int j = 0; j<2; j++)	cout << setprecision(9) << value[j] << " ";
		cout << setprecision(9) << abs(value[1] - exact) << " ";
		for (int j = 2; j<5; j++)	cout << setprecision(9) << value[j] << " ";
		cout << endl;
	}
	cout << "CrankNicolsenbySOR,alpha=" << alpha[1] << endl;
	for (int i = 0; i <4; i++) {
		//cout << "M:" << M[i] << endl;
		VectorXd value = downout.price(M[i], alpha[0], method[0], w);
		for (int j = 0; j<2; j++)	cout << setprecision(9) << value[j] << " ";
		cout << setprecision(9) << abs(value[1] - exact) << " ";
		for (int j = 2; j<5; j++)	cout << setprecision(9) << value[j] << " ";
		cout << endl;
	}
	return 0;
}