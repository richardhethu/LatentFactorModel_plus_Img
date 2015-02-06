#include "corpus.hpp"
#include "common.hpp"
#include <boost/algorithm/string/trim.hpp>

using namespace std;


corpus::corpus(char* voteFile, char* duplicatePath, char* duplicateImagePath, char* featurePath, int userMin, int itemMin, char* graphPath)
{
	nItems = 0;
	nUsers = 0;
	nVotes = 0;
	nEdges = 0;

	imFeatureDim = 4096;

	/// Note that order matters here
	loadImgFeatures(featurePath, duplicateImagePath);  // get items & features
	loadGraph(graphPath, duplicatePath);
	loadVotes(voteFile, userMin, itemMin);	// get users
	
	fprintf(stderr, "\n  \"nUsers\": %d, \"nItems (from img)\": %d, \"nVotes\": %d, \"nEdges\": %d\n", nUsers, nItems, nVotes, nEdges);
}

corpus::~corpus()
{	
	for (vector<vote*>::iterator it = V.begin(); it != V.end(); it++) {
		delete *it;
	}

	for (vector<edge*>::iterator it = M.begin(); it != M.end(); it++) {
		delete *it;
	}
}

void corpus::loadVotes(char* voteFile, int userMin, int itemMin) 
{
	fprintf(stderr, "  Loading votes from %s, userMin = %d, itemMin = %d  ", voteFile, userMin, itemMin);
	
	map<string, int> uCounts;
	map<string, int> bCounts;

	string uName; // User name
	string bName; // Item name
	float value; // Rating
	int voteTime; // Time rating was entered
	int nRead = 0; // Progress
	string line;

	igzstream in;
	in.open(voteFile);

	// Read the input file. It is read twice in case its contents are too large to fit in memory. The first pass is just to gather statistics about the corpus
	// The first time the file is read it is only to compute word counts, in order to select the top "maxWords" words to include in the dictionary
	while (getline(in, line)) {
		stringstream ss(line);
		ss >> uName >> bName >> value >> voteTime;

		nRead++;
		if (nRead % 100000 == 0) {
			fprintf(stderr, ".");
			fflush(stderr);
		}

		if (itemIds.find(bName) == itemIds.end()) {
			continue;
		}

		if (value > 5 or value < 0) { // Ratings should be in the range [0,5]
			printf("Got bad value of %f\nOther fields were %s %s %d\n", value, uName.c_str(), bName.c_str(), voteTime);
			exit(1);
		}

		if (uCounts.find(uName) == uCounts.end()) {
			uCounts[uName] = 0;
		}
		if (bCounts.find(bName) == bCounts.end()) {
			bCounts[bName] = 0;
		}
		uCounts[uName]++;
		bCounts[bName]++;
	}
	in.close();

	// Re-read the entire file, this time building structures from those words in the dictionary
	nUsers = 0;
	
	igzstream in2;
	in2.open(voteFile);
	nRead = 0;
	while (getline(in2, line)) {
		stringstream ss(line);
		ss >> uName >> bName >> value >> voteTime;

		nRead++;
		if (nRead % 100000 == 0) {
			fprintf(stderr, ".");
			fflush(stderr);
		}

		if (itemIds.find(bName) == itemIds.end()) {
			//fprintf(stderr, "Item Id (%s) not found.\n", bName.c_str());
			continue;
		}

		if (uCounts[uName] < userMin or bCounts[bName] < itemMin) {
			continue;
		}

		if (userIds.find(uName) == userIds.end()) {
			rUserIds[nUsers] = uName;
			userIds[uName] = nUsers++;
		}

		vote* v = new vote();
		v->item = itemIds[bName];		
		v->user = userIds[uName];
		v->value = value;
		v->voteTime = voteTime;

		V.push_back(v);
	}
	in2.close();

	fprintf(stderr, "\n");

	random_shuffle(V.begin(), V.end());
	nVotes = V.size();
}

void corpus::loadImgFeatures(char* featurePath, char* duplicateImagePath)
{
	/* Duplicate image list */
	map<string, string> duplicateImages;
	igzstream inDup;
	inDup.open(duplicateImagePath);
	string sAsin;
	string dupof;
	while (inDup >> sAsin >> dupof) {
		duplicateImages[sAsin] = dupof;
	}
	inDup.close();

	FILE* f = fopen_(featurePath, "rb");
	fprintf(stderr, "\n  Loading image features from %s", featurePath);

	double ma = 58.388599; // Largest feature observed
	float* feat = new float [imFeatureDim];
	char* asin = new char [11];
	asin[10] = '\0';
	int a;

	while (!feof(f)) {
		if ((a = fread(asin, sizeof(*asin), 10, f)) != 10) {
			//printf("Expected to read %d chars, got %d\n", 10, a);
			continue;
		}
		for (int c = 0; c < 10; c ++) {
			if (not isascii(asin[c])) {
				printf("Expected asin to be 10-digit ascii\n");
				exit(1);
			}
		}
		if (not (nItems % 10000)) {
			fprintf(stderr, ".");
			fflush(stderr);
		}

		if ((a = fread(feat, sizeof(*feat), imFeatureDim, f)) != imFeatureDim) {
			printf("Expected to read %d floats, got %d\n", imFeatureDim, a);
			exit(1);
		}

		string sAsin(asin);
		if (duplicateImages.find(sAsin) != duplicateImages.end()) {
			continue;
		}

		itemIds[sAsin] = nItems;
		rItemIds[nItems] = sAsin;
		nItems ++;

		std::vector<std::pair<int, float> > vec;
		for (int f = 0; f < imFeatureDim; f ++) {
			if (feat[f] != 0) {  // compression
				vec.push_back(std::make_pair(f, feat[f]/ma));
			}
		}
		imageFeatures.push_back(vec);
	}
	fprintf(stderr, "\n");

	delete[] asin;
	delete [] feat;
	fclose(f);
}

/// Parse G product graphs
void corpus::loadGraph(char* graphPath, char* duplicatePath)
{
	fprintf(stderr, "  Loading graph from %s", graphPath);

	map<string, string> duplicates;
	igzstream inDup;
	inDup.open(duplicatePath);
	string asin;
	string dupof;
	while (inDup >> asin >> dupof) {
		duplicates[asin] = dupof;
	}
	inDup.close();

	igzstream in;
	in.open(graphPath);
	string n1;
	string n2;
	string edgename;
	string line;
	nEdges = 0;
	int count = 0;

	while (getline(in, line)) {
		stringstream ss(line);
		ss >> n1 >> edgename; // Second word of each line should be the edge type
		if (itemIds.find(n1) == itemIds.end()) {
			continue;
		}
		int bid1 = itemIds[n1];
		while (ss >> n2) {
			if (itemIds.find(n2) == itemIds.end()) {
				if (duplicates.find(n2) != duplicates.end()) {
					n2 = duplicates[n2];
				}
				if (itemIds.find(n2) == itemIds.end()) {
					continue;
				}
			}
			int bid2 = itemIds[n2];

			nEdges ++;
			productGraph.insert(make_pair(bid1, bid2));
			nodesInSomeEdge.insert(bid1);
			nodesInSomeEdge.insert(bid2);
		}

		/* print process */
		if ((++ count) % 10000 == 0) {
			fprintf(stderr, ".");
			fflush(stderr);
		}
	}
	fprintf(stderr, "\t\"%s\": %d\n", edgename.c_str(), nEdges);

	for (set<int>::iterator it = nodesInSomeEdge.begin(); it != nodesInSomeEdge.end(); it ++) {
		nodesInSomeEdgeV.push_back(*it);
	}

	initEdges();
}

void corpus::initEdges()
{
	set<pair<int,int> >* G = &productGraph;

	fprintf(stderr, "  Generating edges ");
	int count = 0;
	for (set<pair<int,int> >::iterator it = G->begin(); it != G->end(); it ++) {
		int productFrom = it->first;
		int productTo = it->second;
		int label = 1;
		int reverseLabel = 0;

		if (G->find(make_pair(productTo,productFrom)) != G->end()) {
			reverseLabel = 1;
		}
		
		edge* e = new edge(	productFrom,
							productTo,
							label,
							reverseLabel);
		M.push_back(e);

		/* print process */
		if ((++ count) % 10000 == 0) {
			fprintf(stderr, ".");
			fflush(stderr);
		}
	}

	fprintf(stderr, "\n  Generating non-edges ");

	int NN = nodesInSomeEdgeV.size();
	count = 0;
	while (M.size() < 2 * G->size()) {
		int productFrom = nodesInSomeEdgeV[rand() % NN];
		int productTo = nodesInSomeEdgeV[rand() % NN];
		int label = 0;
		int reverseLabel = 0;

		if (productFrom == productTo or G->find(make_pair(productFrom, productTo)) != G->end()) {
			continue;
		}

		if (G->find(make_pair(productTo, productFrom)) != G->end()) {
			reverseLabel = 1;
		}

		edge* e = new edge(	productFrom,
							productTo,
							label,
							reverseLabel);
		M.push_back(e);

		/* print process */
		if ((++ count) % 10000 == 0) {
			fprintf(stderr, ".");
			fflush(stderr);
		}
	}

	fprintf(stderr, "\n");

	random_shuffle(M.begin(), M.end());
	nEdges = M.size();
}
