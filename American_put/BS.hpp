#pragma once
#ifndef BS_HPP
#define BS_HPP

#include <iostream>
#include <string>
#include <boost/math/distributions/normal.hpp>	// For Normal distribution
#include <boost/math/constants/constants.hpp>	// For PI

using boost::math::normal;

using namespace std;

double PI = boost::math::constants::pi<double>();

double black_schole(double t, double T, double S0, double K, double r, double q, double vol, string opt_type, string flag) {
	
	cout << S0 << "," << K << "," << r << "," << q << "," << vol << "," << T << "," << t << endl;
	double d1 = (log(S0*1.0 / K) + (r - q + 0.5*vol*vol)*(T - t))*1.0 / (vol*sqrt(T - t));
	double d2 = d1 - vol*sqrt(T - t);

	normal s;	// Standard normal
	//cout << "d1: " << d1 << ",d2: " << d2 << endl;
	double C = S0*exp(-q*(T - t))*cdf(s, d1) - K*exp(-r*T)*cdf(s, d2);
	double P = K*exp(-r*(T-t))*cdf(s, -d2) - S0*exp(-q*(T - t))*cdf(s, -d1);
	//cout << "normal cdf of -d2: " << cdf(s, -d2) << endl;

	double delta_C = exp(-q*(T - t))*cdf(s, d1);
	double delta_P = -exp(-q*(T - t))*cdf(s, -d1);
	double gamma = exp(-q*(T - t))*exp(-0.5*pow(d1, 2))*1.0 / (S0*vol*sqrt((T - t) * 2 * PI));
	double vega = S0*exp(-q*(T - t))*sqrt(T - t)*exp(-0.5*pow(d1, 2))*1.0 / sqrt(2 * PI);

	if (opt_type == "PUT") {
		if (flag == "V") {
			return P;
		}
		if (flag == "D") {
			return delta_P;
		}
		if (flag == "G") {
			return gamma;
		}
		if (flag == "VEGA") {
			return vega;
		}
		else {
			return 0;
		}
	}

	if (opt_type == "CALL") {
		if (flag == "V") {
			return C;
		}
		if (flag == "D") {
			return delta_C;
		}
		if (flag == "G") {
			return gamma;
		}
		if (flag == "VEGA") {
			return vega;
		}
		else {
			return 0;
		}
	}

}

#endif // !BS_HPP

