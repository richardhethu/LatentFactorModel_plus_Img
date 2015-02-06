#pragma once

#include "model.hpp"

class latentfactor : public model
{
public:
	latentfactor(corpus* corp, double lambda, int K) : model(corp), lambda(lambda), K(K){}
	~latentfactor(){}
	
	void init();
	void cleanUp();

	double prediction(int user, int item);
	double dl(double* grad);
	void getParametersFromVector(	double* g,
									double** alpha,
									double** beta_user, 
									double** beta_item,
									double*** gamma_user, 
									double*** gamma_item,
									action_t action);
	void train(int gradIterations);
	void saveModel(char* path);

	/* auxiliary variables */
	double*  alpha;
	double*  beta_user;
	double*  beta_item;
	double** gamma_user;
	double** gamma_item;

	/* hyper-parameters */
	double lambda;
	int K;

	/* For multi-threading */
	int NT;
	double* llThread;

	// Separate gradient vectors for each thread
	double**  gradT;
	double**  dalpha;
	double**  dbeta_user;
	double**  dbeta_item;
	double*** dgamma_user;
	double*** dgamma_item;
};
