#include <limits>
#include <algorithm>
#include <vector>
#include <tuple>
#include <cmath>
#include <iostream>
#include <numeric>
#include <cfloat>
#include <functional>
#include "MySplitter.h"


using namespace std;

template <typename T>
vector<size_t> sort_permutation(
    const vector<T>& vec) {
	vector<size_t> p(vec.size());
	iota(p.begin(), p.end(), 0);
	sort(p.begin(), p.end(),
	[&](size_t i, size_t j) { return vec[i] < vec[j]; }); //Sort order can be changed here
	return p;
}

template <typename T>
vector<T> apply_permutation(
    const vector<T>& vec,
    const vector<size_t>& p) {
	vector<T> sorted_vec(vec.size());
	transform(p.begin(), p.end(), sorted_vec.begin(),
	[&](size_t i) { return vec[i]; });
	return sorted_vec;
}

/*
this->num_classes - number of classes in classification task
0 <= label < num_classes
this->lambda - ensemble influence coefficient
*/
MySplitterClass::MySplitterClass() {}


MySplitterClass::~MySplitterClass() {}


MySplitterClass::MySplitterClass(int num_classes, float lambda, float lr)
	: num_classes(num_classes), lambda(lambda), lr(lr) {
}

tuple< vector<DataObject>, vector< vector<float> > >
MySplitterClass::transform_data(const vector< vector<float> > &X,
                                const vector<int> &y,
                                const vector< vector<float> > &proba) {
	/*
		X shape is: features_num * obj_num
		y shape is: obj_num
		proba shape is: class_num * obj_num
	*/
	vector<DataObject> data(X[0].size());
	vector< vector<int> > hist_places;
	vector< vector<float> > unq_features; //features_num * thr_i_num + 1

	for (vector<float> feature_vec : X) {
		vector<int> tmp_hist_places;
		vector<float> tmp_unq;
		tie(tmp_hist_places, tmp_unq) = get_thresholds(feature_vec);
		hist_places.push_back(tmp_hist_places);
		unq_features.push_back(tmp_unq);
	}

	for (int i = 0; i < X[0].size(); ++i) {
		data[i].label = y[i];
		for ( auto el : proba)
			data[i].ens_proba.push_back(el[i]);
		for ( vector<int> el : hist_places)
			data[i].feat_hist_indxs.push_back(el[i]);
	}

	return make_tuple(data, unq_features);
}

tuple<int, vector<float> > MySplitterClass::get_label(const vector<DataObject*> &data) {
	// return label and proba
	vector<int> hist(num_classes, 0);
	int res_label, max = 0;
	vector<float> res_proba(hist.size(), 0.0);

	for (auto obj : data) {
		int ind = obj->label;
		hist[ind] += 1;
		if (hist[ind] > max) {
			res_label = ind;
			max = hist[ind];
		}
	}
	for (int i = 0; i < hist.size(); ++i)
		res_proba[i] = (hist[i] * 1.0) / (data.size() * 1.0);

	return make_tuple(res_label, res_proba);

}

vector<float> MySplitterClass::get_my_entropy(const vector< vector<int> > &l_real_cum_hist,
        const vector< vector<int> > &r_real_cum_hist,
        const vector< vector<float> > &l_proba_cum_hist,
        const vector< vector<float> > &r_proba_cum_hist,
        const vector<int> &l_obj_num, int obj_num, int depth) {
	/*
		l_real_cum_hist - (feat_thr_num)*class_num; Vector's  each cell contains
			number of appropriate class objects, which value less then appropriate threshold.
		l_real_cum_hist - (feat_thr_num)*class_num; Vector's  each cell contains
			number of appropriate class objects, which value greater then appropriate threshold.
		l_proba_cum_hist - (feat_thr_num)*class_num; Vector's  each cell contains
			number of appropriate class objects, which value less then appropriate threshold,
			and this number multiplied with probability of belonging to this class.
		r_proba_cum_hist - (feat_thr_num)*class_num; Vector's  each cell contains
			number of appropriate class objects, which value greater then appropriate threshold,
			and this number multiplied with probability of belonging to this class.
		l_obj_num - feat_thr_num; Number of objects, which value less then appropriate threshold.
		obj_num - total objects number.
		return H_l + H_r = (N_l / N)*left_S + (N_r /N)*right_S size:
	*/
	vector<float> entropy(l_real_cum_hist.size(), 0.0);
	float step_size = 1.0;

	for (int j = 0; j < depth; ++j)
		step_size = step_size * lr;

	for (int i = 0; i < entropy.size(); ++i) {
		float left_S = 0.0, right_S = 0.0;
		int R_l = l_obj_num[i], R = obj_num, R_r = R - R_l;

		if (R_l == 0 or R_r == 0 or (i > 0 and l_obj_num[i] == l_obj_num[i - 1])) {
			entropy[i] = FLT_MAX;
			continue;
		}

		for (int k = 0; k < num_classes; ++k) {
			float p1_lk = l_real_cum_hist[i][k] / (R_l * 1.0);
			float p1_rk = r_real_cum_hist[i][k] / (R_r * 1.0);
			float p2_lk = l_proba_cum_hist[i][k] / (R_l * 1.0);
			float p2_rk = r_proba_cum_hist[i][k] / (R_r * 1.0);

			left_S -= (p1_lk == 0 ? 0 : p1_lk * log(p1_lk));
			left_S += (step_size * lambda) * (p2_lk == 0 ? 0 : p2_lk * log(p2_lk));
			right_S -= (p1_rk == 0 ? 0 : p1_rk * log(p1_rk));
			right_S += (step_size * lambda) * (p2_rk == 0 ? 0 : p2_rk * log(p2_rk));
		}
		entropy[i] += (R_l / (R * 1.0) ) * left_S + ( (R - R_l) / (R * 1.0) ) * right_S;
	}

	return entropy;
}


tuple<vector<int>, vector<float>> MySplitterClass::get_thresholds(const vector<float> &feature_vec) {
	vector<int> indxs;
	vector<size_t> permutation;
	permutation = sort_permutation(feature_vec);
	vector<float> spec_feature_vec = apply_permutation(feature_vec, permutation);
	vector<float> unq_feature_vec; // unq_feature_vec - ascending vector of feature's unique values
	vector<int> tmp_range(feature_vec.size()); // size: objects_num
	iota(tmp_range.begin(), tmp_range.end(), 0);
	vector<int> obj_ind = apply_permutation(tmp_range, permutation); // size: objects_num
	vector<int> hist_places(obj_ind.size(), 0);
	int hist_place = 0;

	unq_feature_vec.push_back(spec_feature_vec[0]);
	for (int i = 0; i < spec_feature_vec.size() - 1; ++i) {
		hist_places[ obj_ind[i] ] = hist_place;
		if (spec_feature_vec[i] != spec_feature_vec[i + 1]) {
			unq_feature_vec.push_back(spec_feature_vec[i + 1]);
			indxs.push_back(i); // it will be index of last object, which goes left
			hist_place += 1;
		}
	}
	hist_places[ obj_ind[spec_feature_vec.size() - 1] ] = hist_place;

	return make_tuple(hist_places, unq_feature_vec);
}


tuple<int, int, int, bool> MySplitterClass::get_best(const vector<DataObject*> &data,
        const vector< vector<float> > &unq_features,
        int verbose, int depth) {
	bool is_leaf(false), prod(true);

	for (auto obj : data)
		prod *= (obj->label == data[0]->label);
	if (prod) {
		if (verbose >= 3)
			cout << "This leaf contains " << data.size() << " objects" << endl;

		is_leaf = true;
		return make_tuple( -1, -1, -1, is_leaf);
	}

	float min_S = FLT_MAX;
	int volume = 0, res_feat_indx = 0, l_unq_indx = 0, r_unq_indx = 1;
	vector< int > res_l_obj_num;

	for (int f = 0; f < unq_features.size(); ++f) {
		vector<float> prob_inn_hist(num_classes, 0.0);
		vector< vector<float> > l_prob_cum_hist(unq_features[f].size() - 1, prob_inn_hist),
		        r_prob_cum_hist(unq_features[f].size() - 1, prob_inn_hist);
		vector<int> real_inn_hist(num_classes, 0),
		       l_obj_num(unq_features[f].size() - 1, 0);
		vector< vector<int> > l_real_cum_hist(unq_features[f].size() - 1, real_inn_hist),
		        r_real_cum_hist(unq_features[f].size() - 1, real_inn_hist);

		for (int i = 0; i < data.size(); ++i) {
			int k = data[i]->label, h_i = data[i]->feat_hist_indxs[f];
			vector< float > p = data[i]->ens_proba;
			if (h_i < unq_features[f].size() - 1) {
				l_real_cum_hist[h_i][k] += 1;
				l_obj_num[h_i] += 1;
			}
			if (h_i > 0)
				r_real_cum_hist[h_i - 1][k] += 1;
			for (int cl = 0; cl < num_classes; ++cl) {
				if (h_i < unq_features[f].size() - 1)
					l_prob_cum_hist[h_i][cl] += p[cl];

				if (h_i > 0)
					r_prob_cum_hist[h_i - 1][cl] += p[cl];
			}
		}
		for (int h_i = 1; h_i < unq_features[f].size() - 1; ++h_i) {
			for (int k = 0; k < num_classes; ++k) {
				int rh_i = (unq_features[f].size() - 1) - 1 - h_i;

				l_prob_cum_hist[h_i][k] += l_prob_cum_hist[h_i - 1][k];
				r_prob_cum_hist[rh_i][k] += r_prob_cum_hist[rh_i + 1][k];
				l_real_cum_hist[h_i][k] += l_real_cum_hist[h_i - 1][k];
				r_real_cum_hist[rh_i][k] += r_real_cum_hist[rh_i + 1][k];
			}
			l_obj_num[h_i] += l_obj_num[h_i - 1];
		}

		vector<float> S = get_my_entropy(l_real_cum_hist, r_real_cum_hist,
		                                 l_prob_cum_hist, r_prob_cum_hist, l_obj_num, data.size(), depth);

		for (int i = 0; i < S.size(); ++i) {
			int R_l = l_obj_num[i], R_r = data.size() - l_obj_num[i];

			if (R_l == 0 or R_r == 0 or (i > 0 and l_obj_num[i] == l_obj_num[i - 1]))
				continue;
			if (min_S > S[i] or (min_S == S[i] and volume < min(R_l, R_r))) {
				volume = min(R_l, R_r);
				l_unq_indx = i;
				res_feat_indx = f;
				min_S = S[i];
				res_l_obj_num = l_obj_num;
			}
		}
	}
	r_unq_indx = l_unq_indx + 1;
	while (res_l_obj_num[r_unq_indx] == res_l_obj_num[r_unq_indx - 1])
		r_unq_indx += 1;
	if (verbose >= 2) {
		vector<float> unq_vec = unq_features[res_feat_indx];
		float thr = (unq_vec[l_unq_indx] + unq_vec[r_unq_indx]) / 2.0;

		cout << "Threshold: " << thr;
		cout << " Feature: " << res_feat_indx << " Entropy: " << min_S << endl;
	}

	return make_tuple(res_feat_indx, l_unq_indx, r_unq_indx, is_leaf);
	//last flag in tuple signalise when time to stop, True => time to stop
}


tuple< vector< DataObject* >, vector< DataObject* > >  MySplitterClass::split_data(
    const vector< DataObject* > &data, float threshold_indx, int feat_indx) {
	vector< DataObject* > l_data, r_data;
	for ( auto ptr : data) {
		if (ptr->feat_hist_indxs[feat_indx] <= threshold_indx)
			l_data.push_back(ptr);
		else
			r_data.push_back(ptr);
	}
	return make_tuple(l_data, r_data); // can't be empty
}