#pragma once
#include <vector>
#include <tuple>

using namespace std;

template <typename T>
vector<size_t> sort_permutation(
    const vector<T>& vec);

template <typename T>
vector<T> apply_permutation(
    const vector<T>& vec,
    const vector<size_t>& p);

struct DataObject {
	int label;
	vector<float> ens_proba; // size: class_num
	vector<int> feat_hist_indxs; // size: feature size
};

class MySplitterClass {
  public:
	int num_classes;
	float lambda;
	float lr;

	MySplitterClass();

	~MySplitterClass();

	MySplitterClass(int num_classes, float lambda, float lr);

	tuple< vector<DataObject>, vector< vector<float> > >
	transform_data(const vector< vector<float> > &X,
	               const vector<int> &y, const vector< vector<float> > &proba);

	vector<float> get_my_entropy(const vector< vector<int> > &l_real_cum_hist,
	                             const vector< vector<int> > &r_real_cum_hist,
	                             const vector< vector<float> > &l_proba_cum_hist,
	                             const vector< vector<float> > &r_proba_cum_hist,
	                             const vector<int> &l_obj_num, int obj_num, int depth);

	tuple<int, vector<float> > get_label(const vector<DataObject*> &data);

	tuple<vector<int>, vector<float>> get_thresholds(const vector<float> &feature_vec);

	tuple<int, int, int, bool> get_best(const vector<DataObject*> &data,
	                                    const vector< vector<float> > &thresholds, int verbose, int depth);

	tuple< vector< DataObject* >, vector< DataObject* > > split_data(
	    const vector< DataObject* > &data, float threshold_indx, int feat_indx);
};