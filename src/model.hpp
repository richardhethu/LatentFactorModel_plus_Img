#pragma once
#include "common.hpp"
#include "corpus.hpp"
#include "HLBFGS.h"

void evalfunc(int N, double* x, double* prev_x, double* f, double* g);
void newiteration(int iter, int call_iter, double* x, double* f, double* g, double* gnorm);

enum action_t { COPY, INIT, FREE };

class model
{
public:
	model(corpus* corp);
	~model();

	inline double sigmoid(double x);
	virtual double prediction(int user, int item) = 0;
	virtual double dl(double* grad) = 0;
	virtual void trainValidTestError(double* train, double* valid, double* test);
	virtual void saveModel(char* modelPath) = 0;
	void copyBestValidModel();

	/* Corpus related */
	corpus* corp;
	int nUsers; // Number of users
	int nItems; // Number of items
	int nVotes; // Number of ratings
	int nEdges; // Number of edges
	
	int validStart;
	int testStart;

	int validStart_match;
	int testStart_match;

	/* Model parameters */
	int NW; // Total number of parameters
	double* W; // Contiguous version of all parameters, i.e., a flat vector containing all parameters in order (useful for lbfgs)
	
	/* Best model */
	double* bestValidModel;
	double bestValid;
};
