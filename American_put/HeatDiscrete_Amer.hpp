// For American PUT option

// LINE: 520


#ifndef HEATDISCRETE_AMER_HPP
#define HEATDISCRETE_AMER_HPP

#include <Eigen/Dense>
#include <functional>
#include"NM.hpp"
#include <boost/math/distributions/normal.hpp>	// For Normal distribution
#include <tuple>
using boost::math::normal;

using namespace Eigen;
using namespace std;

enum A_LinearSolverMethod { ALU, ASOR };
// LU and SOR for American options

// Set Global Variable
//const double S0 = 41, K = 40, T = 0.75, q = 0.02, vol = 0.35, r = 0.04;

class HeatDiscrete_Amer {
public:
	double m_xLeft;
	double m_xRight;
	double m_tauFinal;
	double m_deltaTau, m_deltaX;
	function<double(double, double)> m_gLeft;
	function<double(double, double)> m_gRight;
	function<double(double)> m_f;
	long m_M;
	long m_N;
	double m_alpha_temp;

	VectorXd CreateInitialUvector();
	VectorXd EulerCreateBvector(long timeStep);
	VectorXd CrankNicolsonCreateBvector(long timeIndex);
	MatrixXd ForwardEulerCreateMatrix();
	MatrixXd BackwardEulerCreateMatrix();
	MatrixXd CrankNicolsonCreateMatrixA();
	MatrixXd CrankNicolsonCreateMatrixB();

	//public:
	HeatDiscrete_Amer::HeatDiscrete_Amer(double xLeft, double xRight, double tauFinal, const function<double(double, double)>& gLeft, const function<double(double, double)>& gRight, const function<double(double)>& f, long M, double _alpha_temp) : m_xLeft(xLeft), m_xRight(xRight), m_tauFinal(tauFinal), m_gLeft(gLeft), m_gRight(gRight), m_f(f), m_M(M), m_alpha_temp(_alpha_temp){
		m_deltaTau = tauFinal / (double)M;
		m_N = GetN();
		m_deltaX = (xRight - xLeft) / (double)m_N;
		cout << "m_deltaX: " << m_deltaX << endl;
	};
	double GetAlpha();
	long GetN();
	double early_ex_premium(int i, int m);
	VectorXd Sopt(double tolerance, double omega);

	double a = (r - q)*1.0 / (pow(vol, 2)) - 0.5;
	double b = pow(((r - q)*1.0 / (pow(vol, 2)) + 0.5), 2) + 2 * 1.0*q / (pow(vol, 2));
	
	std::function<double(double, double) > cons= [this](double x, double tau) {
		return K * exp(a * x + b * tau) * std::max(1 - exp(x), 0.0);
	};


	MatrixXd ForwardEuler();
	MatrixXd BackwardEuler(A_LinearSolverMethod linearSolverMethod, double tolerance, double omega);
	//MatrixXd CrankNicolson(A_LinearSolverMethod linearSolverMethod, double tolerance, double omega);
	MatrixXd CrankNicolson_sor(double tolerance, double omega);
};

double HeatDiscrete_Amer::GetAlpha()
{
	int M = m_M;
	//long N = m_N;
	double alpha_temp = m_alpha_temp;

	double xLeft = m_xLeft;
	double xRight = m_xRight;
	double tauFinal = m_tauFinal;

	//double deltaX = (xRight - xLeft) / static_cast<double>(N);
	double deltaTau = tauFinal / static_cast<double>(M);
	double deltaX = sqrt(deltaTau*1.0 / alpha_temp);

	long m_N = floor((xRight - xLeft)*1.0 / deltaX);
	deltaX = (xRight - xLeft)*1.0 / m_N;
	//cout << "N is: " << m_N<<endl;
	//cout << "Alpha: " << deltaTau / pow(deltaX, 2.0) << endl;
	return deltaTau / pow(deltaX, 2.0);
	//cout << "Done alpha\n";
}

long HeatDiscrete_Amer::GetN()
{
	int M = m_M;
	//long N = m_N;
	double alpha_temp = m_alpha_temp;

	double xLeft = m_xLeft;
	double xRight = m_xRight;
	double tauFinal = m_tauFinal;

	//double deltaX = (xRight - xLeft) / static_cast<double>(N);
	double deltaTau = tauFinal / static_cast<double>(M);
	double deltaX = sqrt(deltaTau*1.0 / alpha_temp);

	long m_N = floor((xRight - xLeft)*1.0 / deltaX);

	//cout << "N is: " << m_N << endl;

	return m_N;

}

// Needs to be done in the future
double HeatDiscrete_Amer::early_ex_premium(int i, int m) {
	// Generate the early exercise for given i and m
	// Get inputs
	//std::cout << "Early Exercise\n";
	int M = m_M;
	int N = GetN();
	int size = N - 1;
	double xLeft = m_xLeft;
	double xRight = m_xRight;

	//VectorXd ex_vector(size);
	double deltaX = (xRight - xLeft)*1.0 / N;
	double tauFinal = m_tauFinal;
	double deltaTau = tauFinal / static_cast<double>(M);

	double a = (r - q)*1.0 / (pow(vol, 2)) - 0.5;
	double b = pow(((r - q)*1.0 / (pow(vol, 2)) + 0.5), 2) + 2 * 1.0*q / (pow(vol, 2));

	double x = xLeft + i*deltaX;
	double tau = m*deltaTau;

	return K*exp(a*x + b*tau)*max((1 - exp(x)), 0.0);


}

VectorXd HeatDiscrete_Amer::CreateInitialUvector()
{

	long N = GetN();
	long size = N - 1;
	double deltaX = (m_xRight - m_xLeft) / N;

	VectorXd U0 = VectorXd(size);
	U0.setZero();

	for (long i = 0; i < size; ++i)
	{
		U0(i) = m_f(m_xLeft + (i + 1) * deltaX);
	}

	//cout << U0 << endl;
	return U0;

}

VectorXd HeatDiscrete_Amer::EulerCreateBvector(long timeIndex)
{
	long N = GetN();
	long size = N - 1;
	double alpha = GetAlpha();
	double deltaTau = m_tauFinal / m_M;

	VectorXd b(size);
	b.setZero();

	b(0) = alpha * m_gLeft(0, timeIndex * deltaTau);
	b(size - 1) = alpha * m_gRight(0, timeIndex * deltaTau);

	return b;
}

MatrixXd HeatDiscrete_Amer::ForwardEulerCreateMatrix()
{
	long N = GetN();
	long size = N - 1;
	double alpha = GetAlpha();

	MatrixXd A(size, size);
	A.setZero();

	for (long i = 0; i < size; ++i)
	{
		A(i, i) = 1 - 2 * alpha;
	}

	for (long i = 0; i < size - 1; ++i)
	{
		A(i, i + 1) = alpha;
	}

	for (long i = 1; i < size; ++i)
	{
		A(i, i - 1) = alpha;
	}

	return A;
}


/*
MatrixXd HeatDiscrete_Amer::ForwardEuler()
{
	long M = m_M;
	long N = GetN();
	MatrixXd valuesAtNodes(M + 1, N + 1);
	valuesAtNodes.setZero();

	long numberTimeSteps = M;

	MatrixXd A = ForwardEulerCreateMatrix();
	VectorXd U = CreateInitialUvector();
	VectorXd b;

	valuesAtNodes(0, 0) = m_gLeft(0, 0);
	valuesAtNodes(0, N) = m_gRight(0, 0);

	for (long i = 1; i < N; ++i)
	{
		valuesAtNodes(0, i) = U(i - 1);
	}

	for (long timeIndex = 1; timeIndex <= numberTimeSteps; ++timeIndex)
	{
		b = EulerCreateBvector(timeIndex - 1);
		U = A * U + b;

		valuesAtNodes(timeIndex, 0) = m_gLeft(0, timeIndex * m_tauFinal / M);
		valuesAtNodes(timeIndex, N) = m_gRight(0, timeIndex * m_tauFinal / M);

		for (long i = 1; i < N; ++i)
		{
			// Compare with early exercise premium here
			valuesAtNodes(timeIndex, i) = max(U(i - 1),early_ex_premium(i,timeIndex));
		}
	}
	//cout << "Create U\n";
	return valuesAtNodes;
}
*/

MatrixXd HeatDiscrete_Amer::BackwardEulerCreateMatrix()
{
	long N = GetN();
	long size = N - 1;
	double alpha = GetAlpha();

	MatrixXd A = MatrixXd(size, size);
	A.setZero();

	for (long i = 0; i < size; ++i)
	{
		A(i, i) = 1 + 2 * alpha;
	}

	for (long i = 0; i < size - 1; ++i)
	{
		A(i, i + 1) = -alpha;
	}

	for (long i = 1; i < size; ++i)
	{
		A(i, i - 1) = -alpha;
	}
	//cout << A << endl;
	return A;
}

MatrixXd HeatDiscrete_Amer::BackwardEuler(A_LinearSolverMethod linearSolverMethod, double tolerance, double omega)
{
	long M = m_M;
	long N = GetN();
	MatrixXd valuesAtNodes = MatrixXd(M + 1, N + 1);

	long numberTimeSteps = M;

	MatrixXd A = BackwardEulerCreateMatrix();
	VectorXd U = CreateInitialUvector();
	VectorXd b;
	//cout << U << endl;

	VectorXd x0 = VectorXd(N - 1);

	valuesAtNodes(0, 0) = m_gLeft(0, 0);
	valuesAtNodes(0, N) = m_gRight(0, 0);

	for (long i = 1; i < N; ++i)
	{
		valuesAtNodes(0, i) = U(i - 1);
	}
	//cout << U << endl;

	for (long timeIndex = 1; timeIndex <= numberTimeSteps; ++timeIndex)
	{
		b = EulerCreateBvector(timeIndex);
		//cout << b << endl;

		switch (linearSolverMethod)
		{
		case ALU:
			//cout << U << endl;
			U = linear_solver_row_pivoting(A, U + b);
			//cout << U << endl;
			//cout << A << endl;
			break;
		case ASOR:
			//auto x = SOR_solver(A, U + b, x0, tolerance, true, omega);
			//U = get<0>(x);
			U = get<0>(SOR_solver(A, U + b, x0, tolerance, false, omega));
			//cout << U << endl;
			break;
		}

		valuesAtNodes(timeIndex, 0) = m_gLeft(0, timeIndex * m_tauFinal / M);
		valuesAtNodes(timeIndex, N) = m_gRight(0, timeIndex * m_tauFinal / M);

		for (long i = 1; i < N; ++i)
		{
			valuesAtNodes(timeIndex, i) = U(i - 1);
		}
	}

	return valuesAtNodes;
}

VectorXd HeatDiscrete_Amer::CrankNicolsonCreateBvector(long timeIndex)
{
	long N = GetN();
	long size = N - 1;
	double alpha = GetAlpha();
	double deltaTau = m_tauFinal / m_M;
	double xLeft = m_xLeft;
	double xRight = m_xRight;

	VectorXd b(size);
	b.setZero();

	b(0) = alpha / 2.0 * (m_gLeft(xLeft, timeIndex * deltaTau) + m_gLeft(xLeft, (timeIndex - 1) * deltaTau));
	b(size - 1) = alpha / 2.0 * (m_gRight(xRight, timeIndex * deltaTau) + m_gRight(xRight, (timeIndex - 1) * deltaTau));

	return b;
}

MatrixXd HeatDiscrete_Amer::CrankNicolsonCreateMatrixA()
{
	long N = GetN();
	long size = N - 1;
	double alpha = GetAlpha();

	MatrixXd A(size, size);
	A.setZero();

	for (long i = 0; i < size; ++i)
	{
		A(i, i) = 1 + alpha;
	}

	for (long i = 0; i < size - 1; ++i)
	{
		A(i, i + 1) = -alpha / 2.0;
	}

	for (long i = 1; i < size; ++i)
	{
		A(i, i - 1) = -alpha / 2.0;
	}

	return A;
}

MatrixXd HeatDiscrete_Amer::CrankNicolsonCreateMatrixB()
{
	long N = GetN();
	long size = N - 1;
	double alpha = GetAlpha();

	MatrixXd B(size, size);
	B.setZero();

	for (long i = 0; i < size; ++i)
	{
		B(i, i) = 1 - alpha;
	}

	for (long i = 0; i < size - 1; ++i)
	{
		B(i, i + 1) = alpha / 2;
	}

	for (long i = 1; i < size; ++i)
	{
		B(i, i - 1) = alpha / 2;
	}

	return B;
}

/*
MatrixXd HeatDiscrete_Amer::CrankNicolson(A_LinearSolverMethod linearSolverMethod, double tolerance, double omega)
{
	long M = m_M;
	long N = GetN();
	MatrixXd valuesAtNodes(M + 1, N + 1);
	valuesAtNodes.setZero();
	double xLeft = m_xLeft;
	double xRight = m_xRight;

	long numberTimeSteps = M;

	MatrixXd A = CrankNicolsonCreateMatrixA();
	MatrixXd B = CrankNicolsonCreateMatrixB();
	VectorXd U;
	VectorXd b;

	//if (m_afterDividend == false)
	//{
	U = CreateInitialUvector();
	//}
	//else
	//{
	//U = m_startingNodes;
	//}

	VectorXd x0(N - 1);
	x0.setZero();
	// Create a initial guess vector by using early exercise premium condition
	for (int j = 1; j < N - 1; ++j) {
		//cout << "Create x0\n";
		x0(j) = early_ex_premium(j, M);
	}

	valuesAtNodes(0, 0) = m_gLeft(xLeft, 0);
	valuesAtNodes(0, N) = m_gRight(xRight, 0);

	for (long i = 1; i < N; ++i)
	{
		valuesAtNodes(0, i) = U(i - 1);// Has early exercise premium in CreatingInitialU
	}

	for (long timeIndex = 1; timeIndex <= numberTimeSteps; ++timeIndex)
	{
		b = CrankNicolsonCreateBvector(timeIndex);

		switch (linearSolverMethod)
		{
		case ALU:
			U = linear_solver_no_pivoting(A, B * U + b);
			break;
		case ASOR:
			U = get<0>(SOR_solver(A, B * U + b, x0, tolerance, true, omega));
			//Compare the U to early exercise entry by entry
			for (int j = 1; j < N - 1; ++j) {
				//cout << "Update U\n";
				U(j) = max(U(j), early_ex_premium(j, timeIndex));
			}
			//cout << "check U\n";
			//cout << U << endl;
			break;
		}

		//cout << "Check0";
		valuesAtNodes(timeIndex, 0) = m_gLeft(xLeft, timeIndex * m_tauFinal / M);
		//cout << "Check1";
		valuesAtNodes(timeIndex, N) = m_gRight(xRight, timeIndex * m_tauFinal / M);

		for (long i = 1; i < N; ++i)
		{
			//cout << "Update valuesAtNodes\n";
			valuesAtNodes(timeIndex, i) = U(i - 1);
		}
	}

	return valuesAtNodes;
}
*/

VectorXd HeatDiscrete_Amer::Sopt(double tolerance, double omega) {
	// Get the early exercise domain and compute the corresponding the optimal spot oprice
	long M = m_M;
	long N = GetN();
	MatrixXd valuesAtNodes(M + 1, N + 1);
	valuesAtNodes.setZero();
	double xLeft = m_xLeft;
	double xRight = m_xRight;

	long numberTimeSteps = M;

	double tauFinal = m_tauFinal;
	double deltaX = (xRight - xLeft)*1.0 / N;

	double Sopt = 0, Sopt1 = 0;

	MatrixXd A = CrankNicolsonCreateMatrixA();
	MatrixXd B = CrankNicolsonCreateMatrixB();
	VectorXd U;
	VectorXd b;

	//if (m_afterDividend == false)
	//{
	U = CreateInitialUvector();
	//}
	//else
	//{
	//U = m_startingNodes;
	//}

	VectorXd x0(N - 1);
	x0.setZero();
	// Create a initial guess vector by using early exercise premium condition
	for (int j = 1; j < N - 1; ++j) {
		//cout << "Create x0\n";
		x0(j) = early_ex_premium(j, M);
	}

	valuesAtNodes(0, 0) = m_gLeft(xLeft, 0);
	valuesAtNodes(0, N) = m_gRight(xRight, 0);

	for (long i = 1; i < N; ++i)
	{
		valuesAtNodes(0, i) = U(i - 1);
	}

	VectorXd res(M+1);
	res.setZero();
	for (long timeIndex = 0; timeIndex <= numberTimeSteps; ++timeIndex)
	{
		b = CrankNicolsonCreateBvector(timeIndex);

		U = get<0>(SOR_solver(A, B * U + b, x0, tolerance, true, omega));

		for (int j = 1; j < N - 1; ++j) {
				//cout << "Update U\n";
				U(j) = max(U(j), early_ex_premium(j, timeIndex));
			}
		

		//cout << "Check0";
		valuesAtNodes(timeIndex, 0) = m_gLeft(xLeft, timeIndex * m_tauFinal / M);
		//cout << "Check1";
		valuesAtNodes(timeIndex, N) = m_gRight(xRight, timeIndex * m_tauFinal / M);

		for (long i = 1; i < N; ++i)
		{
			//cout << "Update valuesAtNodes\n";
			valuesAtNodes(timeIndex, i) = U(i - 1);
		}


		//U = valuesAtNodes;


		
	}

	for (int timeIndex = 0; timeIndex <= M; ++timeIndex) {

		//Compare the U to early exercise entry by entry
		for (int j = 1; j < N - 1; ++j) {
			//cout << "Update U\n";
			//U(j) = max(U(j), early_ex_premium(j, timeIndex));
			if (valuesAtNodes(timeIndex, j) = early_ex_premium(j, timeIndex) && valuesAtNodes(timeIndex, j + 1) > early_ex_premium(j + 1, timeIndex)) {
				std::cout << "Early_ex_premium" << early_ex_premium(j, timeIndex) << ", " << early_ex_premium(j + 1, timeIndex) << endl;
				std::cout << "m: " << timeIndex << "j: " << j << endl;
				double x = xLeft + j*deltaX;
				double x1 = xLeft + (j + 1)*deltaX;
				Sopt = K*exp(x);
				Sopt1 = K*exp(x1);
				res(timeIndex) = 0.5*(Sopt + Sopt1);
			}
		}


	}
	return res;
	

}


tuple<double, double, double> error_Amer_pde(HeatDiscrete_Amer &pde, MatrixXd u_approx, double V_exact) {

	// Get the value from the pde object
	int M = pde.m_M;
	int N = pde.GetN();
	double alpha = pde.GetAlpha();

	double xLeft = pde.m_xLeft;
	double xRight = pde.m_xRight;
	double tauFinal = pde.m_tauFinal;

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

	double Vi = exp(-a*xi - b*tauFinal)*u_approx(M, i);
	double Vi1 = exp(-a*xi1 - b*tauFinal)*u_approx(M, i + 1);

	double V_approx = ((Si1 - S0)*Vi + (S0 - Si)*Vi1)*1.0 / (Si1 - Si);

	double u_approx2 = ((xi1 - x_compute)*u_approx(M, i) + (x_compute - xi)*u_approx(M, i + 1))*1.0 / (xi1 - xi);
	double V_approx2 = exp(-a*x_compute - b*tauFinal)*u_approx2;

	double error_pointwise = abs(V_approx - V_exact);
	double error_pointwise2 = abs(V_approx2 - V_exact);
	normal s;
	int NRMS = 0;
	double xk, Vk_approx, Vk_exact, Sk, d1, d2;
	double error_RMS = 0;
	for (int k = 0; k <= N; ++k) {
		xk = xLeft + k*deltaX;
		Vk_approx = exp(-a*xk - b*tauFinal)*u_approx(M, k);
		Sk = K*exp(xk);

		d1 = (log(Sk / K) + (r - q + 0.5*pow(vol, 2)))*T*1.0 / (vol*sqrt(T));
		d2 = d1 - vol*sqrt(T);

		// European Put option
		Vk_exact = K*exp(-r*T)*cdf(s, -d2) - Sk*exp(-q*T)*cdf(s, -d1);
		if (Vk_exact > 0.00001*S0) {
			error_RMS += pow(abs(Vk_approx - Vk_exact), 2)*1.0 / pow(Vk_exact, 2);
			NRMS += 1;
		}
	}

	error_RMS = sqrt(error_RMS*1.0 / NRMS);

	return make_tuple(error_pointwise, error_pointwise2, error_RMS);

}


tuple<double, double, double> greeks_Amer_pde(HeatDiscrete_Amer &pde, MatrixXd u_approx) {

	// Get the value from the pde object
	int M = pde.m_M;
	int N = pde.GetN();
	double alpha = pde.GetAlpha();

	double xLeft = pde.m_xLeft;
	double xRight = pde.m_xRight;
	double tauFinal = pde.m_tauFinal;

	double a = (r - q)*1.0 / (pow(vol, 2)) - 0.5;
	double b = pow(((r - q)*1.0 / (pow(vol, 2)) + 0.5), 2) + 2 * 1.0*q / (pow(vol, 2));

	double deltaTau = tauFinal / static_cast<double>(M);
	double deltaX = (xRight - xLeft)*1.0 / N;

	double x_compute = log(S0 / K);

	int i = floor((x_compute - xLeft) / deltaX);

	double xi_1 = xLeft + (i - 1)*deltaX;
	double xi = xLeft + i*deltaX;
	double xi1 = xLeft + (i + 1)*deltaX;
	double xi2 = xLeft + (i + 2)*deltaX;

	double Si_1 = K*exp(xi_1);
	double Si = K*exp(xi);
	double Si1 = K*exp(xi1);
	double Si2 = K*exp(xi2);

	double Vi_1 = exp(-a*xi_1 - b*tauFinal)*u_approx(M, i - 1);
	double Vi = exp(-a*xi - b*tauFinal)*u_approx(M, i);
	double Vi1 = exp(-a*xi1 - b*tauFinal)*u_approx(M, i + 1);
	double Vi2 = exp(-a*xi2 - b*tauFinal)*u_approx(M, i + 2);

	double delta = (Vi1 - Vi)*1.0 / (Si1 - Si);
	double gamma = ((Vi2 - Vi1)*1.0 / (Si2 - Si1) - (Vi - Vi_1)*1.0 / (Si - Si_1)) / (0.5*(Si2 + Si1) - 0.5*(Si + Si_1));

	double delta_t = -2 * deltaTau / (vol*vol);
	double Vi_dt = exp(-a*xi - b*(tauFinal - deltaTau))*u_approx(M - 1, i);
	double Vi1_dt = exp(-a*xi1 - b*(tauFinal - deltaTau))*u_approx(M - 1, i + 1);
	double Vdt_approx = ((Si1 - S0)*Vi_dt + (S0 - Si)*Vi1_dt)*1.0 / (Si1 - Si);

	double V_approx = ((Si1 - S0) * Vi + (S0 - Si) * Vi1)*1.0 / (Si1 - Si);

	double theta = (V_approx - Vdt_approx) / delta_t;

	return make_tuple(delta, gamma, theta);

}

/*
VectorXd early_ex_domain(HeatDiscrete_Amer &ampde, MatrixXd u_approx) {

	// Find the optimal early exercise domain
	int M = ampde.m_M;
	int N = ampde.GetN();
	VectorXd res(M);
	double xLeft = ampde.m_xLeft;
	double xRight = ampde.m_xRight;
	double tauFinal = ampde.m_tauFinal;
	double deltaX = (xRight - xLeft)*1.0 / N;

	double Sopt = 0, Sopt1 = 0;
	for (int m = 0; m < M; ++m) {

		for (int i = 0; i < N; ++i) {
			if (u_approx(m,i) = ampde.early_ex_premium(i, m) && (u_approx(m,i + 1) > ampde.early_ex_premium(i + 1, m))) {
				std::cout << "m: " << m << "i: " << i << endl;
				double x = xLeft + i*deltaX;
				double x1 = xLeft + (i + 1)*deltaX;
				//double x= u_approx(m, i);
				//double x1 = u_approx(m, i + 1);
				Sopt = K*exp(x);
				Sopt1 = K*exp(x1);
				res(m) = 0.5*(Sopt + Sopt1);
			}

		}
		
	}
	return res;
}

*/


MatrixXd HeatDiscrete_Amer::ForwardEuler()
{
	int M = m_M;
	int N = GetN();
	double alpha = GetAlpha();
	double x_left = m_xLeft;
	double x_right = m_xRight;
	double deltaTau = m_tauFinal / static_cast<double>(M);
	double deltaX = (x_right - x_left)*1.0 / N;

	MatrixXd temp(M + 1, N + 1);

	for (int i = 0; i <= N; ++i) {
		temp(0, i) = m_f(x_left + (x_right - x_left) * i / double(N));
		//cout << "x: " << x_left+deltaX*i << endl;
	}
	for (int j = 0; j <= M; ++j) {
		temp(j, 0) = m_gLeft(0,deltaTau * j);
		temp(j, N) = m_gRight(0, deltaTau * j);
		//cout << "t: " << j*deltaTau << endl;
		//cout << "temp(j,0): " << temp(j, 0) << endl;
	}
	for (int k = 1; k <= M; ++k)
		for (int n = 1; n <= N - 1; ++n) {
			temp(k, n) = alpha * temp(k - 1, n - 1) + (1 - 2 * alpha) * temp(k - 1, n) + alpha * temp(k - 1, n + 1);
			temp(k, n) = std::max(cons(x_left + n * deltaX, deltaTau * k), temp(k, n));
		}
	
	return temp;

}

MatrixXd HeatDiscrete_Amer::CrankNicolson_sor(double tolerance, double omega) {
	int M = m_M;
	int N = GetN();
	double alpha = GetAlpha();
	double x_left = m_xLeft;
	double x_right = m_xRight;
	double deltaTau = m_tauFinal / static_cast<double>(M);
	double deltaX = (x_right - x_left)*1.0 / N;

	MatrixXd temp(M + 1, N + 1);

	for (int i = 0; i <= N; ++i) {
		temp(0, i) = m_f(x_left + (x_right - x_left) * i / double(N));
	}
	for (int j = 0; j <= M; ++j) {
		temp(j, 0) = m_gLeft(0,m_tauFinal * j / double(M));
		temp(j, N) = m_gRight(0, m_tauFinal * j / double(M));
	}
	//Set the Crank_Nicolson Matrix A and B
	MatrixXd A(N - 1, N - 1), B(N - 1, N - 1);
	A.setZero();
	B.setZero();
	A(0, 0) = 1.0 + alpha;
	A(0, 1) = -alpha / 2.0;
	A(N - 2, N - 2) = 1.0 + alpha;
	A(N - 2, N - 3) = -alpha / 2.0;
	B(0, 0) = 1.0 - alpha;
	B(0, 1) = alpha / 2.0;
	B(N - 2, N - 2) = 1.0 - alpha;
	B(N - 2, N - 3) = alpha / 2.0;
	for (int i = 1; i < N - 2; ++i) {
		A(i, i) = 1.0 + alpha;
		A(i, i - 1) = -alpha / 2.0;
		A(i, i + 1) = -alpha / 2.0;
		B(i, i) = 1.0 - alpha;
		B(i, i - 1) = alpha / 2.0;
		B(i, i + 1) = alpha / 2.0;
	}
	//Iteratively solve for U^m+1
	VectorXd v0(temp.block(0, 1, 1, N - 1).transpose());
	for (int j = 1; j <= M; ++j) {
		//initialize b;
		VectorXd vtemp(N - 1);
		vtemp = B * (temp.block(j - 1, 1, 1, N - 1).transpose());
		vtemp(0) += alpha / 2.0 * (temp(j, 0) + temp(j - 1, 0));
		vtemp(N - 2) += alpha / 2.0 * (temp(j, N) + temp(j - 1, N));

		//Use linear_Sor_solver to solve for U;
		std::function<double(double)> consj = [this, j](double i) {
			return cons(m_xLeft + (i + 1) * m_deltaX, j * m_deltaTau);
		};
		tuple<VectorXd, int> res = linear_solve_sor_iter_proj(A, vtemp, v0, 1.2, 1e-6, Criterion_Type::Consec_Approx, consj);
		v0 = std::get<0>(res);
		temp.block(j, 1, 1, N - 1) = v0.transpose();
	}
	return temp;



}





#endif

