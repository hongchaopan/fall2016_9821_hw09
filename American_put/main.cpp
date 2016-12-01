// This main cpp file is written for Q3 of HW8 (MTH9821 Fall2016)

#include<iostream>
#include"HeatDiscrete_Q3.hpp"
//#include"HeatDiscrete.h"
#include "BS.hpp"
#include<Eigen/Dense>
#include<functional>
#include <tuple>
#include <vector>
#include "HeatDiscrete_Amer.hpp"
#include "VarRed.hpp"
#include <fstream>

using namespace std;
using namespace Eigen;

//Output to csv
const IOFormat matrix_format(FullPrecision, 0, ", ", "\n", "", "", "", "");
const IOFormat FormFormat(FullPrecision, 0, ",", "\n");

// Set Global Variable
//const double S0 = 41, K = 40, T = 0.75, q = 0.02, vol = 0.35, r = 0.04;

double a = (r - q)*1.0 / (pow(vol, 2)) - 0.5;
double b = pow(((r - q)*1.0 / (pow(vol, 2)) + 0.5),2) + 2 * 1.0*q / (pow(vol, 2));
double xLeft = log(S0*1.0 / K) + (r - q - 0.5*vol*vol)*T - 3 * vol*sqrt(T);


double gLeft(double x, double t)
{
	return K*exp(a*xLeft + b*t)*(exp(-2 * r*t*1.0 / (pow(vol, 2))) - exp(xLeft - 2 * q*t*1.0 / pow(vol, 2)));
	//return K*exp(a*x + b*t)*(exp(-2 * r*t*1.0 / (pow(vol, 2))) - exp(x - 2 * q*t*1.0 / pow(vol, 2)));
}

// gLeft for American put options
double gLeft_Amer(double x, double t) {
	return K*exp(a*xLeft + b*t)*(1 - exp(xLeft));
	//return K*exp(a*x + b*t)*(1 - exp(x));
}

double gRight(double x, double t)
{
	return 0.0;
}
double f(double x)
{
	return  K*exp(a*x)*(max(1 - exp(x), 0.0));
}


void American_put() {

	// Part 1
	ofstream file("HW9_P1.csv");
	double xRight = xLeft + 6 * vol*sqrt(T);
	double tauFinal = T*0.5*vol*vol;

	int M = 4;
	double alpha_tmp = 0.45;

	HeatDiscrete_Amer Am_put(xLeft, xRight, tauFinal, gLeft_Amer, gRight, f, M, alpha_tmp);
	cout.precision(12);
	file.precision(12);
	MatrixXd A_FE = Am_put.ForwardEuler();
	MatrixXd A_CNSOR = Am_put.CrankNicolson_sor(0.000001, 1.2);
	cout << "**************************\n";
	cout << "Forward Euler\n";
	cout << A_FE << endl;
	cout << "**************************\n";
	cout << "Crank Nicolson with SOR\n";
	cout << A_CNSOR << endl;
	cout << "Done Part 1\n****************************\n";

	file << "Forward Euler with alpha = 0.45" << endl;
	file << A_FE.format(FormFormat) << endl;
	file << "Crank-Nicolson with SOR and alpha = 0.45" << endl;
	file << A_CNSOR.format(FormFormat) << endl;
	file << "Done Part 1\n****************************" << endl;
	file.close();
	// Part 2

	ofstream file2("HW9_P1_Part2.csv");
	double BS_P = black_schole(0.0, T, S0, K, r, q, vol, "PUT", "V");

	double P_exact = 4.083817051176386;


	for (int M = 4; M <= 256; M *= 4) {
		// American Put
		HeatDiscrete_Amer am(xLeft, xRight, tauFinal, gLeft_Amer, gRight, f, M, alpha_tmp);
		cout.precision(12);
		MatrixXd A_FE = am.ForwardEuler();
		MatrixXd A_CNSOR = am.CrankNicolson_sor(0.000001, 1.2);

		// European Put
		HeatDiscrete eu(xLeft, xRight, tauFinal, gLeft, gRight, f, M, alpha_tmp);
		MatrixXd EU_FE = eu.ForwardEuler();
		MatrixXd EU_CNSOR = eu.CrankNicolson_sor(0.000001, 1.2);


		vector<double> error_FE, error_BELU, error_BESOR, error_CNLU, error_CNSOR;
		vector<double> greeks_FE, greeks_BELU, greeks_BESOR, greeks_CNLU, greeks_CNSOR;
		//vector<double> VarRed_FE, VarRed_CNSOR, VaR_err_FE, VaR_err_CNSOR;

		// Get the VarRed and corresponding errors
		double Var_V_FE = VarRed(eu, EU_FE, am, A_FE, BS_P);
		//VarRed_FE.push_back(Var_V_FE);
		double Var_V_CNSOR = VarRed(eu, EU_CNSOR, am, A_CNSOR, BS_P);
		//VarRed_CNSOR.push_back(Var_V_CNSOR);
		//cout << "Var CNSOR: " << Var_V_CNSOR << endl;

		double VaR_err_FE = abs(Var_V_FE - P_exact);
		double VaR_err_CNSOR = abs(Var_V_CNSOR - P_exact);

		tuple<double, double, double> error;
		// Don't record RMS

		error = error_Amer_pde(am, A_FE, P_exact);
		error_FE.push_back(get<0>(error));
		error_FE.push_back(get<1>(error));
		//error_FE.push_back(get<2>(error));

		error = error_Amer_pde(am, A_CNSOR, P_exact);
		error_CNSOR.push_back(get<0>(error));
		error_CNSOR.push_back(get<1>(error));
		//error_CNSOR.push_back(get<2>(error));

		tuple<double, double, double> greeks;

		greeks = greeks_Amer_pde(am, A_FE);
		greeks_FE.push_back(get<0>(greeks));
		greeks_FE.push_back(get<1>(greeks));
		greeks_FE.push_back(get<2>(greeks));

		greeks = greeks_Amer_pde(am, A_CNSOR);
		greeks_CNSOR.push_back(get<0>(greeks));
		greeks_CNSOR.push_back(get<1>(greeks));
		greeks_CNSOR.push_back(get<2>(greeks));

		cout << "************************\n";
		cout << "FE: \n";
		cout << "Errors of M = " << M << " are: \n";
		for (auto elem : error_FE) {
			cout << elem << ", ";
		}cout << endl;
		cout << "Greeks are: \n";
		for (auto elem : greeks_FE) {
			cout << elem << ", ";
		}cout << endl;

		cout << "Variance Reduction and Pointwise Error are: \n";
		cout << Var_V_FE << ", " << VaR_err_FE << endl;
		cout << "************************\n";

		cout << "CNSOR: \n";
		cout << "Errors of M = " << M << " are: \n";
		for (auto elem : error_CNSOR) {
			cout << elem << ", ";
		}cout << endl;
		cout << "Greeks are: \n";
		for (auto elem : greeks_CNSOR) {
			cout << elem << ", ";
		}cout << endl;


		cout << "Variance Reduction and Pointwise Error are: \n";
		cout << Var_V_CNSOR << ", " << VaR_err_CNSOR << endl;
		cout << "************************\n";

		

		file2 << "************************" << endl;
		file2 << "FE with given alpha attemp 0.45: " << endl;
		//file2 << "Errors of M = " << M << " are: " << endl;
		for (auto elem : error_FE) {
			file2 << elem << ", " << ", ";
		}//file2 << endl;
		//file2 << "Greeks are: \n";
		for (auto elem : greeks_FE) {
			file2 << elem << ", ";
		}//file2 << endl;

		//file2 << "Variance Reduction and Pointwise Error are: " << endl;
		file2 << Var_V_FE << ", " << VaR_err_FE << endl;
		file2 << "************************" << endl;

		file2 << "CNSOR with given alpha attemp 0.45: " << endl;
		//file2 << "Errors of M = " << M << " are: " << endl;
		for (auto elem : error_CNSOR) {
			file2 << elem << ", " << ", ";
		}//file2 << endl;
		//file2 << "Greeks are: \n";
		for (auto elem : greeks_CNSOR) {
			file2 << elem << ", ";
		}//file2 << endl;


		//file2 << "Variance Reduction and Pointwise Error are: " << endl;
		file2 << Var_V_CNSOR << ", " << VaR_err_CNSOR << endl;
		file2 << "************************" << endl;

	}

	// Part 2 Crank-Niclson with alpha=5
	cout << "CNSOR with given alpha attemp is 5: \n";
	file2 << "CNSOR with given alpha attemp is 5: " << endl;
	for (int M = 4; M <= 256; M *= 4) {
		alpha_tmp = 5;
		// American Put
		HeatDiscrete_Amer am(xLeft, xRight, tauFinal, gLeft_Amer, gRight, f, M, alpha_tmp);
		cout.precision(12);
		MatrixXd A_CNSOR = am.CrankNicolson_sor(0.000001, 1.2);

		// European Put
		HeatDiscrete eu(xLeft, xRight, tauFinal, gLeft, gRight, f, M, alpha_tmp);
		MatrixXd EU_CNSOR = eu.CrankNicolson_sor(0.000001, 1.2);


		vector<double> error_CNSOR;
		vector<double> greeks_CNSOR;
		//vector<double> VarRed_FE, VarRed_CNSOR, VaR_err_FE, VaR_err_CNSOR;

		// Get the VarRed and corresponding errors
		double Var_V_CNSOR = VarRed(eu, EU_CNSOR, am, A_CNSOR, BS_P);

		double VaR_err_CNSOR = abs(Var_V_CNSOR - P_exact);

		tuple<double, double, double> error;
		// Don't record RMS
		error = error_Amer_pde(am, A_CNSOR, P_exact);
		error_CNSOR.push_back(get<0>(error));
		error_CNSOR.push_back(get<1>(error));
		//error_CNSOR.push_back(get<2>(error));

		tuple<double, double, double> greeks;

		greeks = greeks_Amer_pde(am, A_CNSOR);
		greeks_CNSOR.push_back(get<0>(greeks));
		greeks_CNSOR.push_back(get<1>(greeks));
		greeks_CNSOR.push_back(get<2>(greeks));

		
		cout << "Errors of M = " << M << " are: \n";
		for (auto elem : error_CNSOR) {
			cout << elem << ", ";
		}cout << endl;
		cout << "Greeks are: \n";
		for (auto elem : greeks_CNSOR) {
			cout << elem << ", ";
		}cout << endl;


		//file2 << "Variance Reduction and Pointwise Error are: " << endl;
		//file2 << Var_V_CNSOR << ", " << VaR_err_CNSOR << endl;
		//file2 << "************************" << endl;

		//file2 << "Errors of M = " << M << " are: " << endl;
		for (auto elem : error_CNSOR) {
			file2 << elem << ", " << ", ";
		}//file2 << endl;
		//file2 << "Greeks are: \n";
		for (auto elem : greeks_CNSOR) {
			file2 << elem << ", ";
		}//file2 << endl;


		//file2 << "Variance Reduction and Pointwise Error are: \n";
		file2 << Var_V_CNSOR << ", " << VaR_err_CNSOR << endl;
		file2 << "************************\n";

	}
	file2.close();
	// Part 3
	double S0 = 41, K = 40, T = 0.75, q = 0.02, vol = 0.35, r = 0.04;
	double xLeft = log(S0*1.0 / K) + (r - q - 0.5*vol*vol)*T - 3 * vol*sqrt(T);
	double a = (r - q)*1.0 / (pow(vol, 2)) - 0.5;
	double b = (r - q)*1.0 / (pow(vol, 2)) + 0.5 + 2 * 1.0*q / (pow(vol, 2));

	ofstream file3("HW9_P1_Part3.csv");
	//double xRight = xLeft + 6 * vol*sqrt(T);
	//double tauFinal = T*0.5*vol*vol;

	M = 16;
	alpha_tmp = 0.45;

	HeatDiscrete_Amer Am_put2(xLeft, xRight, tauFinal, gLeft_Amer, gRight, f, M, alpha_tmp);
	cout.precision(12);
	file.precision(12);
	MatrixXd A_CNSOR2 = Am_put2.CrankNicolson_sor(0.000001, 1.2);



	int N = Am_put2.GetN();
	double deltaX = (xRight - xLeft) / (double)N;
	double deltaTau = tauFinal / (double)M;


	auto cons = [K, a, b](double x, double tau) {
		return K * exp(a * x + b * tau) * std::max(1 - exp(x), 0.0);
	};

	std::function<double(double) > temp;
	for (int i = 1; i < A_CNSOR2.rows(); ++i) {
		temp = [cons, xLeft, deltaX, deltaTau, i](double j) {
			return cons(xLeft + j * deltaX, i * deltaTau);
		};
		int j = 0;

		for (; j < A_CNSOR2.cols() && A_CNSOR2(i, j) <= temp(j); ++j);
		file3 << K * (exp(xLeft + (j - 1) * deltaX) + exp(xLeft + j * deltaX)) / 2 << endl;
	}
	file3.close();
	





}



void Euro_PUT() {
	
	double xRight = xLeft + 6 * vol*sqrt(T);
	double tauFinal = T*0.5*vol*vol;
	double tauDividend = 0;

	long M = 4;
	//long N = 8;
	double alpha_tmp = 0.45;
	HeatDiscrete x(xLeft, xRight, tauFinal, gLeft, gRight, f, M,alpha_tmp);
	cout.precision(10);
	MatrixXd A_FE = x.ForwardEuler();
	MatrixXd A_BELU = x.BackwardEuler(LU,0.000001,1.2);
	MatrixXd A_BESOR = x.BackwardEuler(SOR, 0.000001, 1.2);
	MatrixXd A_CNLU = x.CrankNicolson(LU, 0.000001, 1.2);
	MatrixXd A_CNSOR = x.CrankNicolson(SOR, 0.000001, 1.2);
	cout << "**************************\n";
	cout << "Forward Euler\n";
	cout << A_FE << endl;
	cout << "**************************\n";
	cout << "Backward Euler with LU\n";
	cout << A_BELU << endl;
	cout << "**************************\n";
	cout << "Backward Euler with SOR\n";
	cout << A_BESOR << endl;
	cout << "**************************\n";
	cout << "Crank Nicolson with LU\n";
	cout << A_CNLU << endl;
	cout << "**************************\n";
	cout << "Crank Nicolson with SOT\n";
	cout << A_CNSOR << endl;
	cout << "Done Part 1\n****************************\n";

	double BS_P = black_schole(0, T, S0, K, r, q, vol, "PUT", "V");

	for (int M = 4; M <= 256; M*=4) {
	HeatDiscrete x(xLeft, xRight, tauFinal, gLeft, gRight, f, M, alpha_tmp);
	cout.precision(10);
	MatrixXd A_FE = x.ForwardEuler();
	MatrixXd A_BELU = x.BackwardEuler(LU, 0.000001, 1.2);
	MatrixXd A_BESOR = x.BackwardEuler(SOR, 0.000001, 1.2);
	MatrixXd A_CNLU = x.CrankNicolson(LU, 0.000001, 1.2);
	MatrixXd A_CNSOR = x.CrankNicolson(SOR, 0.000001, 1.2);

	vector<double> error_FE, error_BELU, error_BESOR, error_CNLU, error_CNSOR;
	vector<double> greeks_FE, greeks_BELU, greeks_BESOR, greeks_CNLU, greeks_CNSOR;

	tuple<double, double, double> error;
	error= error_eu_pde(x, A_FE, BS_P);
	error_FE.push_back(get<0>(error));
	error_FE.push_back(get<1>(error));
	error_FE.push_back(get<2>(error));

	error = error_eu_pde(x,A_BELU, BS_P);
	error_BELU.push_back(get<0>(error));
	error_BELU.push_back(get<1>(error));
	error_BELU.push_back(get<2>(error));

	error = error_eu_pde(x, A_BESOR, BS_P);
	error_BESOR.push_back(get<0>(error));
	error_BESOR.push_back(get<1>(error));
	error_BESOR.push_back(get<2>(error));

	error = error_eu_pde(x, A_CNLU, BS_P);
	error_CNLU.push_back(get<0>(error));
	error_CNLU.push_back(get<1>(error));
	error_CNLU.push_back(get<2>(error));

	error = error_eu_pde(x, A_CNSOR, BS_P);
	error_CNSOR.push_back(get<0>(error));
	error_CNSOR.push_back(get<1>(error));
	error_CNSOR.push_back(get<2>(error));

	tuple<double, double, double> greeks;
	greeks = greeks_eu_pde(x, A_FE);
	greeks_FE.push_back(get<0>(greeks));
	greeks_FE.push_back(get<1>(greeks));
	greeks_FE.push_back(get<2>(greeks));

	greeks = greeks_eu_pde(x, A_BELU);
	greeks_BELU.push_back(get<0>(greeks));
	greeks_BELU.push_back(get<1>(greeks));
	greeks_BELU.push_back(get<2>(greeks));

	greeks = greeks_eu_pde(x, A_BESOR);
	greeks_BESOR.push_back(get<0>(greeks));
	greeks_BESOR.push_back(get<1>(greeks));
	greeks_BESOR.push_back(get<2>(greeks));

	greeks = greeks_eu_pde(x, A_CNLU);
	greeks_CNLU.push_back(get<0>(greeks));
	greeks_CNLU.push_back(get<1>(greeks));
	greeks_CNLU.push_back(get<2>(greeks));

	greeks = greeks_eu_pde(x, A_CNSOR);
	greeks_CNSOR.push_back(get<0>(greeks));
	greeks_CNSOR.push_back(get<1>(greeks));
	greeks_CNSOR.push_back(get<2>(greeks));

	cout << "************************\n";
	cout << "FE: \n";
	cout << "Errors of M = " << M << " are: \n";
	for (auto elem : error_FE) {
	cout << elem << ", ";
	}cout << endl;
	cout << "Greeks are: \n";
	for (auto elem : greeks_FE) {
	cout << elem << ", ";
	}cout << endl;

	cout << "************************\n";
	cout << "BELU: \n";
	cout << "Errors of M = " << M << " are: \n";
	for (auto elem : error_BELU) {
	cout << elem << ", ";
	}cout << endl;
	cout << "Greeks are: \n";
	for (auto elem : greeks_BELU) {
	cout << elem << ", ";
	}cout << endl;


	cout << "************************\n";
	cout << "BESOR: \n";
	cout << "Errors of M = " << M << " are: \n";
	for (auto elem : error_BESOR) {
	cout << elem << ", ";
	}cout << endl;
	cout << "Greeks are: \n";
	for (auto elem : greeks_BESOR) {
	cout << elem << ", ";
	}cout << endl;

	cout << "************************\n";
	cout << "CNLU: \n";
	cout << "Errors of M = " << M << " are: \n";
	for (auto elem : error_CNLU) {
	cout << elem << ", ";
	}cout << endl;
	cout << "Greeks are: \n";
	for (auto elem : greeks_CNLU) {
	cout << elem << ", ";
	}cout << endl;

	cout << "************************\n";
	cout << "CNSOR: \n";
	cout << "Errors of M = " << M << " are: \n";
	for (auto elem : error_CNSOR) {
	cout << elem << ", ";
	}cout << endl;
	cout << "Greeks are: \n";
	for (auto elem : greeks_CNSOR) {
	cout << elem << ", ";
	}cout << endl;

	}

	
}


int main() {
	American_put();	// HW9	
	//Euro_PUT();	// HW8

	int stop;
	cout << "Enter number to stop (0): ";
	cin >> stop;
	//system("Pause");
}
