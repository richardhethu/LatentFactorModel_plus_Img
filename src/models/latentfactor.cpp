#include "latentfactor.hpp"
#include <ctime>

using namespace std;

extern latentfactor* gModel;

void latentfactor::init()
{
	gModel = this;

	NW = 1 + // offset
		 (K + 1) * (nUsers + nItems); // bias and latent factors

	W = new double[NW];
	for (int w = 0; w < NW; w ++) {
		W[w] = 0;
	}

	getParametersFromVector(W, &alpha, &beta_user, &beta_item, &gamma_user, &gamma_item, INIT);

	for (int u = 0; u < nUsers; u ++) {
		for (int k = 0; k < K; k ++) {
			gamma_user[u][k] = rand() * 1.0 / RAND_MAX;
		}
	}
	for (int i = 0; i < nItems; i ++) {
		for (int k = 0; k < K; k ++) {
			gamma_item[i][k] = rand() * 1.0 / RAND_MAX;
		}
	}

	/* copy best model */
	bestValidModel = new double[NW];
	for (int w = 0; w < NW; w ++) {
		bestValidModel[w] = W[w];
	}

	/* for multi-threads - begin */
	NT = omp_get_max_threads();

	// Separate gradient vectors for each thread
	gradT = new double* [NT];
	dalpha = new double* [NT];
	dbeta_user = new double* [NT];
	dbeta_item = new double* [NT];
	dgamma_user = new double** [NT];
	dgamma_item = new double** [NT];
	for (int t = 0; t < NT; t ++) {
		gradT[t] = new double [NW];
		getParametersFromVector(gradT[t], dalpha + t, dbeta_user + t, dbeta_item + t, dgamma_user + t, dgamma_item + t, INIT);
	}
	llThread = new double [NT];
	/* for multi-threads - end */
}

void latentfactor::cleanUp()
{
	getParametersFromVector(W, &alpha, &beta_user, &beta_item, &gamma_user, &gamma_item, FREE);

	delete [] W;
	delete [] bestValidModel;

	delete [] gradT;
	for (int t = 0; t < NT; t ++) {
		delete [] gradT[t];
		getParametersFromVector(0, dalpha + t, dbeta_user + t, dbeta_item + t, dgamma_user + t, dgamma_item + t, FREE);
	}
	delete [] dalpha;
	delete [] dbeta_user;
	delete [] dbeta_item;
	delete [] dgamma_user;
	delete [] dgamma_item;

	delete [] llThread;
}

void latentfactor::getParametersFromVector(	double*   g,
											double**  alpha, 
											double**  beta_user, 
											double**  beta_item, 
											double*** gamma_user, 
											double*** gamma_item,
											action_t  action)
{
	if (action == FREE) {
		delete [] (*gamma_user);
		delete [] (*gamma_item);
		return;
	}

	if (action == INIT)	{
		*gamma_user = new double* [nUsers];
		*gamma_item = new double* [nItems];
	}

	int ind = 0;
	*alpha = g + ind;
	ind ++;

	*beta_user = g + ind;
	ind += nUsers;
	*beta_item = g + ind;
	ind += nItems;

	for (int u = 0; u < nUsers; u ++) {
		(*gamma_user)[u] = g + ind;
		ind += K;
	}
	for (int i = 0; i < nItems; i ++) {
		(*gamma_item)[i] = g + ind;
		ind += K;
	}

	if (ind != NW) {
		printf("Got bad index (latentfactor.cpp, line %d)", __LINE__);
		exit(1);
	}
}

double latentfactor::prediction(int user, int item)
{
	return *alpha + beta_user[user] + beta_item[item] + inner(gamma_user[user], gamma_item[item], K);
}

double latentfactor::dl(double* grad)
{
	double l_dlStart = clock_();
	
#pragma omp parallel for
	for (int w = 0; w < NW; w ++) {
		grad[w] = 0;
	}

	for (int t = 0; t < NT; t ++) {
		llThread[t] = 0;
	}

	for (int t = 0; t < NT; t ++) {
		for (int w = 0; w < NW; w ++) {
			gradT[t][w] = 0;
		}
	}

#pragma omp parallel for
	for (int x = 0; x < validStart; x++) {
		int tid = omp_get_thread_num();

		vote* vi = corp->V.at(x);
		int user = vi->user;
		int item = vi->item;
		double val = vi->value;

		double pred = prediction(user, item);

		llThread[tid] += square(pred - val);
		
		double deri = dsquare(pred - val);
		
		*(dalpha[tid]) += deri;
		dbeta_user[tid][user] += deri;
		dbeta_item[tid][item] += deri;
		
		for (int k = 0; k < K; k ++) {
			dgamma_item[tid][item][k] += deri * gamma_user[user][k];
			dgamma_user[tid][user][k] += deri * gamma_item[item][k];
		}
	}

	int l2_start = 1 + nUsers + nItems;
	if (lambda > 0) {
		for (int w = l2_start; w < NW; w ++) {
			llThread[0] += lambda * W[w] * W[w];
		}
	}
	double llTotal = 0;
	for (int t = 0; t < NT; t ++) {
		llTotal += llThread[t];
	}

	// Add up the gradients from all threads
	for (int t = 0; t < NT; t ++) {
		for (int w = 0; w < NW; w ++) {
			grad[w] += gradT[t][w];
		}
	}
	if (lambda > 0) {
		for (int w = l2_start; w < NW; w ++) {
			grad[w] += 2 * lambda * W[w];
		}
	}

	fprintf(stderr, "took %f\n", clock_() - l_dlStart);
	return llTotal;
}

void latentfactor::train(int gradIterations)
{
	fprintf(stderr, "\n===== Latent Factor Model =====\n");
	fprintf(stderr, "Hyper-param: K = %d\n", K);
	fprintf(stderr, "Hyper-param: lambda = %f\n\n", lambda);

	double parameter[20];
	int info[20];
	//initialize
	INIT_HLBFGS(parameter, info);
	info[4] = gradIterations;
	info[5] = 0;
	info[6] = 0;
	info[7] = 0;
	info[10] = 0;
	info[11] = 1;
	HLBFGS(NW, 20, W, evalfunc, 0, HLBFGS_UPDATE_Hessian, newiteration, parameter, info);
	fprintf(stderr, "\n");

	/* copy back best model */
	for(int w = 0; w < NW; w ++) {
		W[w] = bestValidModel[w];
	}
}

void latentfactor::saveModel(char* savePath)
{
	FILE* f = fopen_(savePath, "w");
	fprintf(f, "{\n");
	fprintf(f, "  \"NW\": %d,\n", NW);
	fprintf(f, "  \"alpha\": %f,\n", *alpha);
	
	fprintf(f, "  \"beta_user\":");
	fprintf(f, "  [\n");
	for (int u = 0; u < nUsers; u ++) {
		fprintf(f, "%f", beta_user[u]);
		if (u < nUsers - 1) fprintf(f, ", ");
	}
	fprintf(f, "]");

	fprintf(f, "  \"beta_item\":");
	fprintf(f, "  [\n");
	for (int i = 0; i < nItems; i ++) {
		fprintf(f, "%f", beta_item[i]);
		if (i < nItems - 1) fprintf(f, ", ");
	}
	fprintf(f, "]");

	fprintf(f, "  \"gamma_user\":");
	fprintf(f, "  [\n");
	for (int u = 0; u < nUsers; u ++) {
		for (int k = 0; k < K; k ++) {
			fprintf(f, "%f", gamma_user[u][k]);
			if (u < nUsers - 1 || k < K - 1) fprintf(f, ", ");
		}
	}
	fprintf(f, "]");

	fprintf(f, "  \"gamma_item\":");
	fprintf(f, "  [\n");
	for (int i = 0; i < nItems; i ++) {
		for (int k = 0; k < K; k ++) {
			fprintf(f, "%f", gamma_item[i][k]);
			if (i < nItems - 1 || k < K - 1) fprintf(f, ", ");
		}
	}
	fprintf(f, "]");

	fprintf(f, "  \"W\": [");
	for (int w = 0; w < NW; w ++) {
		fprintf(f, "%f", W[w]);
		if (w < NW - 1) fprintf(f, ", ");
	}
	fprintf(f, "]\n");
	fprintf(f, "}\n");
	fclose(f);
}