#pragma once

#include "latentfactor.hpp"

class imagemodel : public latentfactor 
{
public:
	imagemodel(corpus* corp, double lambda, double eta, int K, int K2) : latentfactor(corp, lambda, K), K2(K2), eta(eta) {}
	~imagemodel(){}
	
	void init();
	void cleanUp();

	void getParametersFromVector(	double*   g,
									double**  alpha,
									double**  c_m,
									double**  beta_user, 
									double**  beta_item,
									double*** gamma_user, 
									double*** gamma_item,
									double*** theta_user,									
									double*** U,
									action_t action);

	double predict(int user, int item, double* k_space);
	double predictMatch(int itemIdA, int itemIdB, double* k_space, double* f_space);	
	double dl(double* grad);

	void train(int gradIterations);
	void trainValidTestError(double* train, double* valid, double* test);
	void saveModel(char* path);

	/* hyper-parameters */
	int K2;
	double eta;

	/* auxiliary variables */
	double** theta_user;

	/* Model parameters */
	double*  c_m;  	// item-item (MATCH)
	double** U;  	// embedding matrix (K by 4096)

	/* For multi-threading */
	double*** dtheta_user;
	double*** dU;
	double** dcm;

	double**  k_space;
	double**  f_space;
};
