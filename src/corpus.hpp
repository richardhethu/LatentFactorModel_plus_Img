#pragma once

#include "common.hpp"

class corpus
{
public:
	corpus(char* voteFile, char* duplicatePath, char* duplicateImagePath, char* featurePath, int userMin, int itemMin, char* graphPath);
	~corpus();

	std::vector<vote*> V;	// vote
	std::vector<edge*> M;  // match

	int nUsers; // Number of users
	int nItems; // Number of items
	int nVotes; // Number of ratings
	int nEdges; // Number of edges

	std::map<std::string, int> userIds; // Maps a user's string-valued ID to an integer
	std::map<std::string, int> itemIds; // Maps an item's string-valued ID to an integer

	std::map<int, std::string> rUserIds; // Inverse of the above maps
	std::map<int, std::string> rItemIds;

	std::vector<std::vector<std::pair<int, float> > > imageFeatures;
	int imFeatureDim;  // fixed to 4096/400

	std::set<std::pair<int,int> > productGraph; // Edgelist per graph

	std::set<int> nodesInSomeEdge; // Set of nodes that appear in some edge
	std::vector<int> nodesInSomeEdgeV; // Same thing as a vector

	void loadVotes(char* voteFile, int userMin, int itemMin);
	void loadImgFeatures(char* featurePath, char* duplicateImagePath);
	void loadGraph(char* graphPath, char* duplicatePath);
	void initEdges();
};
