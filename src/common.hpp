#pragma once

#include "stdio.h"
#include "stdlib.h"
#include "vector"
#include "math.h"
#include "string.h"
#include <string>
#include <iostream>
#include "omp.h"
#include "map"
#include "set"
#include "vector"
#include "common.hpp"
#include "algorithm"
#include "sstream"
#include "gzstream.h"

/// Safely open a file
FILE* fopen_(const char* p, const char* m);

/// Data associated with a rating
typedef struct vote
{
	int user; // ID of the user
	int item; // ID of the item
	float value; // Rating

	int voteTime; // Unix time of the rating
	std::vector<int> words; // IDs of the words in the review
} vote;

inline double inner(double* x, double* y, int K)
{
	double res = 0;
	for (int k = 0; k < K; k ++) {
		res += x[k]*y[k];
	}
	return res;
}

inline double square(double x)
{
	return x*x;
}

inline double dsquare(double x)
{
	return 2*x;
}

inline double safeLog(double p)
{
	double x = log(1 + exp(p));
	if (isnan(x) or isinf(x)) {
		if (isnan(p) or isinf(p)) {
			printf("Bad prediction\n");
			exit(1);
		}
		return p;
	}
	return x;
}

class edge
{
public:
	edge(int productFrom,
		int productTo,
		int label, // Is this an edge or a non-edge
		int reverseLabel); // Does the graph have an edge going the other direction?

	~edge();

	int productFrom;
	int productTo;

	int label;
	int reverseLabel;
};