#include "corpus.hpp"
#include "latentfactor.hpp"
#include "imagemodel.hpp"

using namespace std;

void experiment_latentfactor(char* reviewPath, char* duplicatePath, char* duplicateImgPath, char* featurePath, char* graphPath, int K, double lambda, int iter)
{
	fprintf(stderr, "{\n");
	fprintf(stderr, "  \"corpus\": \"%s\",\n", reviewPath);

	corpus corp(reviewPath, duplicatePath, duplicateImgPath, featurePath, 0, 0, graphPath);

	double train, valid, test;

	latentfactor M_latentfactor(&corp,lambda, K);
	M_latentfactor.init();
	M_latentfactor.train(iter);
	M_latentfactor.trainValidTestError(&train, &valid, &test);
	M_latentfactor.cleanUp();

	fprintf(stderr, "}\n");
}

void experiment_latExtention(char* reviewPath, char* duplicatePath, char* duplicateImgPath, char* featurePath, char* graphPath, int K, int K2, double eta, double lambda, int iter)
{
	fprintf(stderr, "{\n");
	fprintf(stderr, "  \"corpus\": \"%s\",\n", reviewPath);

	corpus corp(reviewPath, duplicatePath, duplicateImgPath, featurePath, 0, 0, graphPath);
	
	double train, valid, test;

	imagemodel M_image(&corp, lambda, eta, K, K2);
	M_image.init();
	M_image.train(iter);
	M_image.trainValidTestError(&train, &valid, &test);
	M_image.cleanUp();

	fprintf(stderr, "}\n");
}

int main(int argc, char** argv)
{
	srand(0);
	
	if (argc < 11) {
		printf(" Parameters as following: \n");
		printf(" 1. review path\n");
		printf(" 2. duplicate path\n");
		printf(" 3. duplicate Img path\n");
		printf(" 4. Img feature path\n");
		printf(" 5. Graph file (currently supports ONE)\n");
		printf(" 6. Latent Feature length (K)\n");
		printf(" 7. Visual Feature length (K2)\n");
		printf(" 8. eta (for match)\n");
		printf(" 9. lambda (for L2)\n");
		printf(" 10. iterations\n");
		
		exit(1);
	}

	char* reviewPath = argv[1];
	char* duplicatePath = argv[2];
	char* duplicateImgPath = argv[3];
	char* featurePath = argv[4];
	char* graphPath = argv[5];
	int K = atoi(argv[6]);
	int K2 = atoi(argv[7]);
	double eta = atof(argv[8]);
	double lambda = atof(argv[9]);
	int iter = atoi(argv[10]);

	experiment_latExtention(reviewPath, duplicatePath, duplicateImgPath, featurePath, graphPath, K, K2, eta, lambda, iter);
	//experiment_latentfactor(reviewPath, duplicatePath, duplicateImgPath, featurePath, graphPath, K, lambda, iter);
	return 0;
}
