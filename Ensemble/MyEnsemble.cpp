#include <vector>
#include <tuple>
#include <iostream>
#include "MyTree.h"
#include "MyEnsemble.h"
#include <algorithm> // for max_element
#include <iterator> // for std::distance
#include <climits>

using namespace std;

MyEnsemble::MyEnsemble() {}


MyEnsemble::~MyEnsemble() {}


MyEnsemble::MyEnsemble(int num_classes, float reg_param, float lr,
                       int num_trees, bool warm_start, int max_depth, int verbose, int bootstrap_seed)
	: num_classes(num_classes), reg_param(reg_param), lr(lr), num_trees(num_trees),
	  warm_start(warm_start), max_depth(max_depth == -1 ? INT_MAX : max_depth), verbose(verbose),
	  bootstrap_seed(bootstrap_seed), bootstrap(bootstrap_seed > -2) {}


void MyEnsemble::fit(const vector< vector<float> > &X, const vector<int> &y) {
	vector<float> probs(X[0].size(), 0.0); // Obj_num
	vector< vector<float> > ens_probs(num_classes, probs), votes(num_classes, probs);// if not warm_start
	vector< DataObject > Data_vec; // if not warm_start
	vector< vector<float> > unq_features; // if not warm_start

	if (forest.size() == 0) {
		if (bootstrap) {
			if ( bootstrap_seed == -1 )
				srand( time(NULL) );
			else
				srand(bootstrap_seed);
		}
		float reg_param = 0;
		MyTree new_tree = MyTree(num_classes, reg_param, lr, max_depth,
		                         verbose, bootstrap);

		if (verbose >= 1)
			cout << ">> Tree number: " << 1 << " <<" << endl;

		if (!warm_start) {
			std::tie(Data_vec, unq_features) = new_tree.splitter.transform_data(X, y, ens_probs);
			new_tree.fit(Data_vec, unq_features);
			vector<int> _;
			vector< vector<float> > tmp_probs;
			tie(_, tmp_probs) = new_tree.predict(X);
			for (int k = 0; k < num_classes; ++k)
				for (int t = 0; t < X[0].size(); ++t) {
					Data_vec[t].ens_proba[k] = tmp_probs[t][k];
					votes[k][t] += tmp_probs[t][k];
				}
		} else
			new_tree.fit(X, y, ens_probs);
		forest.emplace_back(move(new_tree));
	}

	int start_trees_cnt = forest.size();
	// fit new trees with labels: count = num_trees - forest.size()
	for (int i = 0; i < num_trees - start_trees_cnt; ++i) {
		MyTree new_tree = MyTree(num_classes, reg_param, lr,
		                         max_depth, verbose, bootstrap);

		if (verbose >= 1)
			cout << ">> Tree number: " << start_trees_cnt + 1 + i << " <<" << endl;

		if (warm_start) {
			vector< vector<float> > votes = get_forest_votes(X);
			for (int k = 0; k < num_classes; ++k)
				for (int t = 0; t < X[0].size(); ++t)
					ens_probs[k][t] = votes[t][k] / (forest.size() * 1.0);

			new_tree.fit(X, y, ens_probs);
		} else {
			new_tree.fit(Data_vec, unq_features);
			vector<int> _;
			vector< vector<float> > tmp_probs;
			tie(_, tmp_probs) = new_tree.predict(X);
			for (int k = 0; k < num_classes; ++k)
				for (int t = 0; t < X[0].size(); ++t) {
					float sz = forest.size();
					votes[k][t] += tmp_probs[t][k];
					Data_vec[t].ens_proba[k] = votes[k][t] / (sz + 1.0);
				}
		}
		forest.emplace_back(move(new_tree));
		if (verbose >= 1)
			cout << ">> Ensemble size: " << forest.size() << " <<" << endl;
	}
}

vector<int>  MyEnsemble::predict(const vector< vector<float> > &X) {
	// get labels and start election
	vector< vector<float> > votes = get_forest_votes(X);
	return get_forest_res(votes);
}


vector< vector<float> >  MyEnsemble::predict_proba(const vector< vector<float> > &X) {
	// get labels and start election
	vector< vector<float> > votes = get_forest_votes(X);
	for (int i = 0; i < votes.size(); ++i) {
		float sum = 0.0;
		for (float el : votes[i])
			sum += el;
		for (int j = 0; j < votes[i].size(); ++j)
			votes[i][j] /= sum;
	}
	return votes;
}


vector< vector<float> > MyEnsemble::get_forest_votes(const vector< vector<float> > &X) {
	vector<float> hist(num_classes, 0);
	vector< vector<float> > votes(X[0].size(), hist); //  samples_num * classes_num
	for (int i = 0; i < forest.size(); ++i) {
		vector<int> tmp_y;
		vector< vector<float> > tmp_probs;
		tie(tmp_y, tmp_probs) = forest[i].predict(X);
		for (int j = 0; j < votes.size(); ++j) {
			float sum = 0.0;
			for (int k = 0; k < votes[0].size(); ++k) {
				votes[j][k] += tmp_probs[j][k];
				sum += tmp_probs[j][k];
			}
			if (abs(sum - 1) > 0.00001)
				cout << "Error in MyEnsemble::get_forest_votes" << sum << endl;
		}
	}
	return votes;
}


vector<int> MyEnsemble::get_forest_res(const vector< vector<float> > &votes) {
	vector<int> res_vote(votes.size());
	for (int i = 0; i < res_vote.size(); ++i) {
		res_vote[i] = distance(votes[i].begin(), \
		                       max_element(votes[i].begin(), votes[i].end()));
	}
	return res_vote;
}

vector< vector<int> > MyEnsemble::warm_predict(const vector< vector<float> > &X) {
	vector< vector< vector<float> > >votes = warm_get_forest_votes(X);
	return warm_get_forest_res(votes);
}


vector< vector< vector<float> > > MyEnsemble::warm_predict_proba(const vector< vector<float> > &X) {
	vector< vector< vector<float> > > votes = warm_get_forest_votes(X);
	for (int k = 0; k < votes.size(); ++k) {
		for ( auto arr : votes[k]) {
			float sum = 0.0;
			for (float el : arr)
				sum += el;
			for (int i = 0; i < arr.size(); ++i)
				arr[i] /= sum;
		}
	}
	return votes;
}


vector< vector< vector<float> > > MyEnsemble::get_each_tree_votes(const vector< vector<float> > &X) {
	// get labels and start election
	vector<float> hist(num_classes, 0.0);
	vector< vector<float> > votes(X[0].size(), hist); // samples_num * classes_num
	vector< vector< vector<float> > > res(forest.size(), votes); // trees_num * smples_num * classes_num

	for (int i = 0; i < forest.size(); ++i) {
		vector<int> tmp_y;
		vector< vector<float> > tmp_probs;
		tie(tmp_y, tmp_probs) = forest[i].predict(X);
		for (int j = 0; j < votes.size(); ++j) {
			float sum = 0.0;
			for (int k = 0; k < votes[0].size(); ++k) {
				res[i][j][k] += tmp_probs[j][k];
				sum += tmp_probs[j][k];
			}
			if (abs(sum - 1) > 0.00001)
				cout << "Error in MyEnsemble::get_each_tree_votes" << sum << endl;
		}
	}
	return res;
}


vector< vector< vector<float> > > MyEnsemble::warm_get_forest_votes(const vector< vector<float> > &X) {
	// get labels and start election
	vector<float> hist(num_classes, 0.0);
	vector< vector<float> > votes(X[0].size(), hist); // samples_num * classes_num
	vector< vector< vector<float> > > res(forest.size(), votes); // trees_num * smples_num * classes_num

	for (int i = 0; i < forest.size(); ++i) {
		vector<int> tmp_y;
		vector< vector<float> > tmp_probs;
		tie(tmp_y, tmp_probs) = forest[i].predict(X);
		for (int j = 0; j < votes.size(); ++j) {
			float sum = 0.0;
			for (int k = 0; k < votes[0].size(); ++k) {
				res[i][j][k] += tmp_probs[j][k];
				if (i > 0)
					res[i][j][k] += res[i - 1][j][k];
				sum += tmp_probs[j][k];
			}
			if (abs(sum - 1) > 0.00001)
				cout << "Error in MyEnsemble::warm_get_forest_votes" << sum << endl;
		}
	}
	return res;
}


vector< vector<int> > MyEnsemble::warm_get_forest_res(const vector< vector< vector<float> > > &votes) {
	vector<int> res_vote(votes[0].size(), 0); // samples_num
	vector< vector<int> > res(votes.size(), res_vote); // trees_num

	for (int k = 0; k < res.size(); ++k) {
		for (int i = 0; i < res_vote.size(); ++i) {
			res[k][i] = distance(votes[k][i].begin(), \
			                     max_element(votes[k][i].begin(), votes[k][i].end()));
		}
	}
	return res;
}