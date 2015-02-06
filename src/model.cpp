#include "model.hpp"
#include "latentfactor.hpp"
#include <cfloat>

model* gModel;

void evalfunc(int N, double* x, double* prev_x, double* f, double* g)
{
	// Negative signs because we want gradient ascent rather than descent
	*f = gModel->dl(g);
}

/// Required function by HLBFGS library: prints progress
void newiteration(int iter, int call_iter, double* x, double* f, double* g, double* gnorm)
{
	fprintf(stderr, ".");
	fflush(stderr);

	if (iter % 5 == 0) {
		fprintf(stderr, "Iter: %d, call_iter: %d \n", iter, call_iter);
		double train, valid, test;
		gModel->trainValidTestError(&train, &valid, &test);
		if (valid < gModel->bestValid) {
			gModel->copyBestValidModel();
			gModel->bestValid = valid;
		}
	}
}

model::model(corpus* corp) : corp(corp)
{
	nUsers = corp->nUsers;
	nItems = corp->nItems;
	nVotes = corp->nVotes;
	nEdges = corp->nEdges;

	// Use 1/10 of the data for train, 1/10 for validation
	double testFraction = 0.1;
	validStart = (int) ((1.0 - 2 * testFraction) * nVotes);
	testStart = (int) ((1.0 - testFraction) * nVotes);

	validStart_match = (int) ((1.0 - 2 * testFraction) * nEdges);
	testStart_match = (int) ((1.0 - testFraction) * nEdges);

	if (validStart < 1 or (testStart - validStart) < 1 or (nVotes - testStart) < 1) {
		printf("Didn't get enough ratings (%d/%d/%d)\n", validStart, testStart, nVotes);
		exit(1);
	}

	if (validStart_match < 1 or (testStart_match - validStart_match) < 1 or (nEdges - testStart_match) < 1) {
		printf("Didn't get enough edges (%d/%d/%d)\n", validStart_match, testStart_match, nEdges);
		exit(1);
	}

	W = 0;
	bestValidModel = 0;
	bestValid = FLT_MAX;
}

model::~model()
{
}

void model::trainValidTestError(double* train, double* valid, double* test)
{
	int NT = omp_get_max_threads();
	double* train_thread = new double [NT];
	double* valid_thread = new double [NT];
	double* test_thread = new double [NT];	
	for (int t = 0; t < NT; t ++) {
		train_thread[t] = valid_thread[t] = test_thread[t] = 0;
	}

#pragma omp parallel for
	for (int i = 0; i < nVotes; i ++) {
		int tid = omp_get_thread_num();

		vote* v = corp->V.at(i);
		double p = prediction(v->user, v->item);
		double se = square(v->value - p);

		if (i < validStart) {
			train_thread[tid] += se;
		} else if (i < testStart) {
			valid_thread[tid] += se;
		} else {
			test_thread[tid] += se;
		}
	}

	*train = 0;
	*valid = 0;
	*test = 0;

	for (int t = 0; t < NT; t ++) {
		*train += train_thread[t];
		*valid += valid_thread[t];
		*test += test_thread[t];
	}

	*train /= validStart;
	*valid /= (testStart - validStart);
	*test /= (nVotes - testStart);

	fprintf(stderr, "  \"error\": {\"train\": %f, \"valid\": %f, \"test\": %f}\n", *train, *valid, *test);
	fflush(stderr);

	delete [] train_thread;
	delete [] valid_thread;
	delete [] test_thread;
}

void model::copyBestValidModel()
{
	for(int w = 0; w < NW; w ++) {
		bestValidModel[w] = W[w];
	}
	fprintf(stderr, "  Model copied. #para: %d\n", NW);
	fflush(stderr);
}
