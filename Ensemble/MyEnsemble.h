#pragma once
#include <vector>
#include "MyTree.h"

using namespace std;

class MyEnsemble
{
public:
	vector<MyTree> forest;
	bool warm_start;
	int num_trees, num_classes;
	float reg_param, lr;
	int max_depth;
	int verbose;
	bool bootstrap;
	int bootstrap_seed;

	MyEnsemble();
	
	~MyEnsemble();
	
	MyEnsemble(int num_classes, float reg_param, float lr, int num_trees,
		bool warm_start, int max_depth, int verbose, int bootstrap_seed);
	
	void fit(const vector< vector<float> > &X, const vector<int> &y);
	
	vector<int> predict(const vector< vector<float> > &X);
	
	vector< vector<float> > predict_proba(const vector< vector<float> > &X);
	
	vector< vector<float> > get_forest_votes(const vector< vector<float> > &X);
	
	vector<int> get_forest_res(const vector< vector<float> > &votes);
	
	vector< vector<int> > warm_predict(const vector< vector<float> > &X);
	
	vector< vector< vector<float> > > warm_predict_proba(const vector< vector<float> > &X);
	
	vector< vector< vector<float> > > get_each_tree_votes(const vector< vector<float> > &X);
	
	vector< vector< vector<float> > > warm_get_forest_votes(const vector< vector<float> > &X);
	
	vector< vector<int> > warm_get_forest_res(const vector< vector< vector<float> > > &votes);
};