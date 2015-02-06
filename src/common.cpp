#include "common.hpp"
using namespace std;

/// Safely open a file
FILE* fopen_(const char* p, const char* m)
{
	FILE* f = fopen(p, m);
	if (!f) {
		printf("Failed to open %s\n", p);
		exit(1);
	}
	return f;
}

edge::edge(int productFrom,
           int productTo,
           int label,
           int reverseLabel) :
    productFrom(productFrom),
    productTo(productTo),
    label(label),
    reverseLabel(reverseLabel)
  {}

edge::~edge()
{
}