#ifndef HEATDISCRETE_HPP
#define HEATDISCRETE_HPP

#include <Eigen/Dense>
#include <functional>
#include"NM.hpp"
using namespace Eigen;
using namespace std;

enum LinearSolverMethod { LU, SOR };

class HeatDiscrete {
	double m_xLeft;
	double m_xRight;
	double m_tauFinal;
	function<double(double, double)> m_gLeft;
	function<double(double,double)> m_gRight;
	function<double(double, double)> m_fExact; 
	function<double(double)> m_f;
	long m_M;
	long m_N;

	VectorXd CreateInitialUvector();
	VectorXd EulerCreateBvector(long timeStep);
	VectorXd CrankNicolsonCreateBvector(long timeIndex);
	MatrixXd ForwardEulerCreateMatrix() ;
	MatrixXd BackwardEulerCreateMatrix() ;
	MatrixXd CrankNicolsonCreateMatrixA();
	MatrixXd CrankNicolsonCreateMatrixB();
	

public:
	HeatDiscrete::HeatDiscrete(double xLeft, double xRight, double tauFinal, const function<double(double, double)>& gLeft, const function<double(double, double)>& gRight, const function<double(double)>& f, long M, long N, const function<double(double, double)>& fExact)
		: m_xLeft(xLeft), m_xRight(xRight), m_tauFinal(tauFinal), m_gLeft(gLeft), m_gRight(gRight), m_f(f), m_M(M), m_N(N), m_fExact(fExact)
	{};
	double GetAlpha();

	MatrixXd ForwardEuler();
	MatrixXd BackwardEuler(LinearSolverMethod linearSolverMethod, double tolerance, double omega);
	MatrixXd CrankNicolson(LinearSolverMethod linearSolverMethod, double tolerance, double omega);
};

double HeatDiscrete::GetAlpha() 
{
	long M = m_M;
	long N = m_N;
	double xLeft = m_xLeft;
	double xRight = m_xRight;
	double tauFinal = m_tauFinal;

	double deltaX = (xRight - xLeft) / static_cast<double>(N);
	double deltaTau = tauFinal / static_cast<double>(M);

	return deltaTau / pow(deltaX, 2.0);
}

VectorXd HeatDiscrete::CreateInitialUvector()
{
	long N = m_N;
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

VectorXd HeatDiscrete::EulerCreateBvector(long timeIndex)
{
	long N = m_N;
	long size = N - 1;
	double alpha = GetAlpha();
	double deltaTau = m_tauFinal / m_M;

	VectorXd b(size);
	b.setZero();

	b(0) = alpha * m_gLeft(0,timeIndex * deltaTau);
	b(size - 1) = alpha * m_gRight(0,timeIndex * deltaTau);

	return b;
}

MatrixXd HeatDiscrete::ForwardEulerCreateMatrix()
{
	long N = m_N;
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

MatrixXd HeatDiscrete::ForwardEuler()
{
	long M = m_M;
	long N = m_N;
	MatrixXd valuesAtNodes(M + 1, N + 1);
	valuesAtNodes.setZero();

	long numberTimeSteps = M;

	MatrixXd A = ForwardEulerCreateMatrix();
	VectorXd U = CreateInitialUvector();
	VectorXd b;

	valuesAtNodes(0, 0) = m_gLeft(0,0);
	valuesAtNodes(0, N) = m_gRight(0,0);

	for (long i = 1; i < N; ++i)
	{
		valuesAtNodes(0, i) = U(i - 1);
	}

	for (long timeIndex = 1; timeIndex <= numberTimeSteps; ++timeIndex)
	{
		b = EulerCreateBvector(timeIndex - 1);
		U = A * U + b;

		valuesAtNodes(timeIndex, 0) = m_gLeft(0,timeIndex * m_tauFinal / M);
		valuesAtNodes(timeIndex, N) = m_gRight(0,timeIndex * m_tauFinal / M);

		for (long i = 1; i < N; ++i)
		{
			valuesAtNodes(timeIndex, i) = U(i - 1);
		}
	}

	return valuesAtNodes;
}

MatrixXd HeatDiscrete::BackwardEulerCreateMatrix()
{
	long N = m_N;
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

MatrixXd HeatDiscrete::BackwardEuler(LinearSolverMethod linearSolverMethod, double tolerance, double omega)
{
	long M = m_M;
	long N = m_N;
	MatrixXd valuesAtNodes = MatrixXd(M + 1, N + 1);

	long numberTimeSteps = M;

	MatrixXd A = BackwardEulerCreateMatrix();
	VectorXd U = CreateInitialUvector();
	VectorXd b;
	//cout << U << endl;

	VectorXd x0 = VectorXd(N - 1);

	valuesAtNodes(0, 0) = m_gLeft(0,0);
	valuesAtNodes(0, N) = m_gRight(0,0);

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
		case LU:
			//cout << U << endl;
			U = linear_solver_row_pivoting(A, U + b);
			//cout << U << endl;
			//cout << A << endl;
			break;
		case SOR:
			//auto x = SOR_solver(A, U + b, x0, tolerance, true, omega);
			//U = get<0>(x);
			U = get<0>(SOR_solver(A, U + b, x0, tolerance, false, omega));
			//cout << U << endl;
			break;
		}

		valuesAtNodes(timeIndex, 0) = m_gLeft(0,timeIndex * m_tauFinal / M);
		valuesAtNodes(timeIndex, N) = m_gRight(0,timeIndex * m_tauFinal / M);

		for (long i = 1; i < N; ++i)
		{
			valuesAtNodes(timeIndex, i) = U(i - 1);
		}
	}

	return valuesAtNodes;
}

VectorXd HeatDiscrete::CrankNicolsonCreateBvector(long timeIndex)
{
	long N = m_N;
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

MatrixXd HeatDiscrete::CrankNicolsonCreateMatrixA()
{
	long N = m_N;
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

MatrixXd HeatDiscrete::CrankNicolsonCreateMatrixB()
{
	long N = m_N;
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

MatrixXd HeatDiscrete::CrankNicolson(LinearSolverMethod linearSolverMethod, double tolerance, double omega)
{
	long M = m_M;
	long N = m_N;
	MatrixXd valuesAtNodes(M + 1, N + 1);
	valuesAtNodes.setZero();
	double xLeft = m_xLeft;
	double xRight = m_xRight;

	long numberTimeSteps = M;

	MatrixXd A = CrankNicolsonCreateMatrixA();
	MatrixXd B = CrankNicolsonCreateMatrixB();
	VectorXd U;
	VectorXd b;

	
		U = CreateInitialUvector();
	

	VectorXd x0(N - 1);
	x0.setZero();

	valuesAtNodes(0, 0) = m_gLeft(xLeft, 0);
	valuesAtNodes(0, N) = m_gRight(xRight, 0);

	for (long i = 1; i < N; ++i)
	{
		valuesAtNodes(0, i) = U(i - 1);
	}

	for (long timeIndex = 1; timeIndex <= numberTimeSteps; ++timeIndex)
	{
		b = CrankNicolsonCreateBvector(timeIndex);

		switch (linearSolverMethod)
		{
		case LU:
			U = linear_solver_no_pivoting(A, B * U + b);
			break;
		case SOR:
			U = get<0>(SOR_solver(A, B * U + b, x0, tolerance, true , omega));
			//cout << U << endl;
			break;
		}

		valuesAtNodes(timeIndex, 0) = m_gLeft(xLeft, timeIndex * m_tauFinal / M);
		valuesAtNodes(timeIndex, N) = m_gRight(xRight, timeIndex * m_tauFinal / M);

		for (long i = 1; i < N; ++i)
		{
			valuesAtNodes(timeIndex, i) = U(i - 1);
		}
	}

	return valuesAtNodes;
}



#endif

