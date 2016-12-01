#pragma once

#ifndef VARRED_HPP
#define VARRED_HPP

#include "HeatDiscrete_Q3.hpp"
#include "HeatDiscrete_Amer.hpp"
#include <iostream>
#include <Eigen/Dense>
#include <functional>
#include "NM.hpp"
#include <boost/math/distributions/normal.hpp>	// For Normal distribution
#include <tuple>
using boost::math::normal;

using namespace Eigen;
using namespace std;

double VarRed(HeatDiscrete &eupde, MatrixXd &eu_approx,  HeatDiscrete_Amer &ampde, MatrixXd &am_approx, double BSVal) {
	// European Put option
	// Get the value from the pde object
	int M = eupde.m_M;
	int N = eupde.GetN();
	double alpha = eupde.GetAlpha();

	double xLeft = eupde.m_xLeft;
	double xRight = eupde.m_xRight;
	double tauFinal = eupde.m_tauFinal;

	double a = (r - q)*1.0 / (pow(vol, 2)) - 0.5;
	double b = pow(((r - q)*1.0 / (pow(vol, 2)) + 0.5), 2) + 2 * 1.0*q / (pow(vol, 2));


	double deltaTau = tauFinal / static_cast<double>(M);
	double deltaX = (xRight - xLeft)*1.0 / N;

	double x_compute = log(S0 / K);
	int i = floor((x_compute - xLeft) / deltaX);

	double xi = xLeft + i*deltaX;
	double xi1 = xLeft + (i + 1)*deltaX;
	double Si = K*exp(xi);
	double Si1 = K*exp(xi1);

	double Vi = exp(-a*xi - b*tauFinal)*eu_approx(M, i);
	double Vi1 = exp(-a*xi1 - b*tauFinal)*eu_approx(M, i + 1);

	double V_approx_eu = ((Si1 - S0)*Vi + (S0 - Si)*Vi1)*1.0 / (Si1 - Si);

	// American Put option
	// Get the value from the pde object
	int A_M = ampde.m_M;
	int A_N = ampde.GetN();
	double A_alpha = ampde.GetAlpha();

	double A_xLeft = ampde.m_xLeft;
	double A_xRight = ampde.m_xRight;
	double A_tauFinal = ampde.m_tauFinal;
	/*
	double a = (r - q)*1.0 / (pow(vol, 2)) - 0.5;
	double b = (r - q)*1.0 / (pow(vol, 2)) + 0.5 + 2 * 1.0*q / (pow(vol, 2));
	*/
	double A_deltaTau = A_tauFinal / static_cast<double>(A_M);
	double A_deltaX = (A_xRight - A_xLeft)*1.0 / A_N;

	double A_x_compute = log(S0 / K);
	int A_i = floor((A_x_compute - A_xLeft) / A_deltaX);

	double A_xi = A_xLeft + A_i*A_deltaX;
	double A_xi1 = A_xLeft + (A_i + 1)*A_deltaX;
	double A_Si = K*exp(A_xi);
	double A_Si1 = K*exp(A_xi1);

	double A_Vi = exp(-a*A_xi - b*A_tauFinal)*am_approx(A_M, A_i);
	double A_Vi1 = exp(-a*A_xi1 - b*A_tauFinal)*am_approx(A_M, A_i + 1);

	double V_approx_am = ((A_Si1 - S0)*A_Vi + (S0 - A_Si)*A_Vi1)*1.0 / (A_Si1 - A_Si);

	cout << "V_am, V_eu, V_BS: " << V_approx_am << ", " << V_approx_eu << ", " << BSVal << endl;
	return V_approx_am + BSVal - V_approx_eu;


}








#endif // !VARRED_HPP
