#include "imagemodel.hpp"

using namespace std;

extern imagemodel* gModel;

void imagemodel::init()
{
	gModel = this;

	NW = 2 + (K + 1) * (nUsers + nItems) 	// latent factors + 1 visual bias
		 + K2 * nUsers + K2 * corp->imFeatureDim; // visual factors

	W = new double[NW];
	for (int w = 0; w < NW; w ++) {
		if (w < 2 + nUsers + nItems) { // bias terms
			W[w] = 0;
		} else {
			W[w] = rand() * 1.0 / RAND_MAX;
		}
	}

	getParametersFromVector(W, &alpha, &c_m, &beta_user, &beta_item, &gamma_user, &gamma_item, &theta_user, &U, INIT);

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
	dtheta_user = new double** [NT];
	dcm = new double* [NT];
	dU = new double** [NT];
	for (int t = 0; t < NT; t ++) {
		gradT[t] = new double [NW];
		getParametersFromVector(gradT[t], dalpha + t, dcm + t, dbeta_user + t, dbeta_item + t, dgamma_user + t, dgamma_item + t, dtheta_user + t, dU +t, INIT);
	}
	k_space = new double* [NT];
	f_space = new double* [NT];
	for (int t = 0; t < NT; t ++) {
		k_space[t] = new double [K2];
		f_space[t] = new double [corp->imFeatureDim];
	}
	llThread = new double [NT];
	/* for multi-threads - end */
}

void imagemodel::cleanUp()
{
	getParametersFromVector(W, &alpha, &c_m, &beta_user, &beta_item, &gamma_user, &gamma_item, &theta_user, &U, FREE);

	delete [] W;
	delete [] bestValidModel;

	delete [] gradT;
	for (int t = 0; t < NT; t ++) {
		delete [] gradT[t];
		delete [] k_space[t];
		delete [] f_space[t];
		getParametersFromVector(0, dalpha + t, dcm + t, dbeta_user + t, dbeta_item + t, dgamma_user + t, dgamma_item + t, dtheta_user + t, dU +t, FREE);
	}
	delete [] dalpha;
	delete [] dbeta_user;
	delete [] dbeta_item;
	delete [] dgamma_user;
	delete [] dgamma_item;
	delete [] dtheta_user;
	delete [] dU;
	delete [] dcm;
	delete [] k_space;
	delete [] f_space;
	delete [] llThread;
}

void imagemodel::getParametersFromVector(	double*   g,
											double**  alpha,
											double**  c_m,
											double**  beta_user, 
											double**  beta_item,
											double*** gamma_user, 
											double*** gamma_item,
											double*** theta_user,
											double*** U,
											action_t action)
{
	if (action == FREE) {
		delete [] (*gamma_user);
		delete [] (*gamma_item);
		delete [] (*theta_user);
		delete [] (*U);
		return;
	}

	if (action == INIT)	{
		*gamma_user = new double* [nUsers];
		*gamma_item = new double* [nItems];
		*theta_user = new double* [nUsers];
		*U = new double* [K2];
	}

	int ind = 0;
	*alpha = g + ind;
	ind ++;

	*c_m = g + ind;
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

	for (int u = 0; u < nUsers; u ++) {
		(*theta_user)[u] = g + ind;
		ind += K2;
	}

	for (int k = 0; k < K2; k ++) {
		(*U)[k] = g + ind;
		ind += corp->imFeatureDim;
	}

	if (ind != NW) {
		printf("Got bad index (imagemodel.cpp, line %d)", __LINE__);
		exit(1);
	}
}

double imagemodel::predict(int user, int item, double* k_space)
{
	for (int k = 0; k < K2; ++k) {
		k_space[k] = 0;  // embed features to K2-dim
		std::vector<std::pair<int, float> >* feat = &corp->imageFeatures.at(item);
		for (unsigned i = 0; i < feat->size(); i ++) {
			k_space[k] += U[k][feat->at(i).first] * feat->at(i).second;
		}
	}
	return *alpha + beta_user[user] + beta_item[item] + inner(gamma_user[user], gamma_item[item], K) + inner(k_space, theta_user[user], K2);
}

/// Predict whether two items match each other
double imagemodel::predictMatch(int itemIdA, int itemIdB, double* k_space, double* f_space)
{
	/* Calculate diff in image feature space */
	for (int j = 0; j < corp->imFeatureDim; j ++) {
		f_space[j] = 0;
	}
	for (unsigned j = 0; j < corp->imageFeatures.at(itemIdA).size(); j ++) {
		f_space[corp->imageFeatures.at(itemIdA).at(j).first] = corp->imageFeatures.at(itemIdA).at(j).second;
	}
	for (unsigned j = 0; j < corp->imageFeatures.at(itemIdB).size(); j ++) {
		f_space[corp->imageFeatures.at(itemIdB).at(j).first] -= corp->imageFeatures.at(itemIdB).at(j).second;
	}

	double dist = 0;
	for (int k = 0; k < K2; ++k) {
		k_space[k] = 0;  // embed features to K-dim
		for (int i = 0; i < corp->imFeatureDim; ++i) {
			if (f_space[i] != 0) {  // avoid useless float multiplying
				k_space[k] += U[k][i] * f_space[i];
			}
		}
		dist += k_space[k] * k_space[k];
	}
	double z = (*c_m) - dist;
	return z; //Note: require sigmoid to be prob
}

double imagemodel::dl(double* grad)
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

		double pred = predict(user, item, k_space[tid]);

		llThread[tid] += square(pred - val);
		
		double deri = dsquare(pred - val);
		
		*(dalpha[tid]) += deri;
		dbeta_user[tid][user] += deri;
		dbeta_item[tid][item] += deri;
		
		for (int k = 0; k < K; k ++) {
			dgamma_item[tid][item][k] += deri * gamma_user[user][k];
			dgamma_user[tid][user][k] += deri * gamma_item[item][k];
		}

		for (int k = 0; k < K2; k ++) {
			dtheta_user[tid][user][k] += deri * k_space[tid][k];
		}

		for (int r = 0; r < K2; r ++) {
			for (unsigned f = 0; f < corp->imageFeatures.at(item).size(); f ++) {
				int c = corp->imageFeatures.at(item).at(f).first;
				double val = corp->imageFeatures.at(item).at(f).second;
				dU[tid][r][c] += deri * theta_user[user][r] * val;
			}
		}
	}

	if (eta > 0) {
#pragma omp parallel for schedule(dynamic)
		for (int i = 0; i < validStart_match; ++i) {
			int tid = omp_get_thread_num();
			
			edge* e = corp->M.at(i);
			int itemA = e->productFrom;
			int itemB = e->productTo;
			int label = e->label;  // match or not

			double z = predictMatch(itemA, itemB, k_space[tid], f_space[tid]);

			llThread[tid] -= eta * (label * z - safeLog(z));

			double frac = 1.0 / (1.0 + exp(-z));
			double deri = - eta * (label - frac);

			*(dcm[tid]) += deri;

			for (int r = 0; r < K2; r ++) {
				for (int c = 0; c < corp->imFeatureDim; c ++) {
					dU[tid][r][c] -= deri * 2 * k_space[tid][r] * f_space[tid][c];
				}
			}
		} // end of edge loop 
	}

	int l2_start = 2 + nUsers + nItems;
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

void imagemodel::trainValidTestError(double* train, double* valid, double* test)
{
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
		double p = predict(v->user, v->item, k_space[tid]);
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

	fprintf(stderr, "  \"Vote error\": {\"train\": %f, \"valid\": %f, \"test\": %f}\n", *train, *valid, *test);
	fflush(stderr);


	for (int t = 0; t < NT; t ++) {
		train_thread[t] = valid_thread[t] = test_thread[t] = 0;
	}

#pragma omp parallel for
	for(int i = 0; i < nEdges; i++) {
		int tid = omp_get_thread_num();

		int label = corp->M.at(i)->label;
		double z = predictMatch(corp->M.at(i)->productFrom, corp->M.at(i)->productTo, k_space[tid], f_space[tid]);
		
		if ((z > 0 and label == 0) or // Predicted an edge when there wasn't one
			(z <= 0 and label == 1)) { // Didn't predict an edge when there was one
			if (i < validStart_match) {
				train_thread[tid] += 1;
			} else if (i < testStart_match) {
				valid_thread[tid] += 1;
			} else {
				test_thread[tid] += 1;
			}
		}
	}

	double train_sum = 0, valid_sum = 0, test_sum = 0;
	for (int t = 0; t < NT; t ++) {
		train_sum += train_thread[t];
		valid_sum += valid_thread[t];
		test_sum += test_thread[t];
	}

	train_sum /= validStart_match;
	valid_sum /= (testStart_match - validStart_match);
	test_sum /= (nEdges - testStart_match);

	fprintf(stderr, "  \"Match error\": {\"train\": %f, \"valid\": %f, \"test\": %f}\n", train_sum, valid_sum, test_sum);
	fflush(stderr);

	delete [] train_thread;
	delete [] valid_thread;
	delete [] test_thread;
}

void imagemodel::train(int gradIterations)
{
	fprintf(stderr, "\n===== Image Model =====\n");
	fprintf(stderr, "Hyper-param: K = %d\n", K);
	fprintf(stderr, "Hyper-param: K2 = %d\n", K2);
	fprintf(stderr, "Hyper-param: eta = %f\n", eta);
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

void imagemodel::saveModel(char* savePath)
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

	fprintf(f, "  \"theta_user\":");
	fprintf(f, "  [\n");
	for (int u = 0; u < nUsers; u ++) {
		for (int k = 0; k < K2; k ++) {
			fprintf(f, "%f", theta_user[u][k]);
			if (u < nUsers - 1 || k < K2- 1) fprintf(f, ", ");
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