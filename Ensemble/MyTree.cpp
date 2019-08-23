#include <vector>
#include <iostream>
#include <tuple>
#include <memory>
#include "MyTree.h"


using namespace std;

NodeInfoContainer::NodeInfoContainer(MyNode &current_node, const std::vector<DataObject*> &data)
	: node_ptr(&current_node), data(data) {
}

MyNode::MyNode() {}

MyNode::~MyNode() {}

MyNode::MyNode(float threshold, int feature_indx, int label, bool is_leaf)
	: threshold(threshold), feature_indx(feature_indx), label(label), is_leaf(is_leaf) {}

MyNode::MyNode(MyNode const &node) {
	cout << ">> COPY CONSTRUCTOR <<" << endl;
	threshold = node.threshold;
	feature_indx = node.feature_indx;
	is_leaf = node.is_leaf;

	if (node.is_leaf) {
		left_node_ptr = nullptr;
		right_node_ptr = nullptr;
		label = node.label;
		proba = node.proba;
	} else {
		MyNode left_child(*node.left_node_ptr);
		MyNode right_child(*node.right_node_ptr);
		unique_ptr<MyNode> lp = make_unique<MyNode>(left_child);
		unique_ptr<MyNode> rp = make_unique<MyNode>(right_child);
		left_node_ptr = move(lp);
		right_node_ptr = move(rp);
	}
}

void MyNode::append_children() {
	if (is_leaf) {
		unique_ptr<MyNode> p1(nullptr);
		left_node_ptr = move(p1);
		unique_ptr<MyNode> p2(nullptr);
		right_node_ptr = move(p2);
	} else {
		unique_ptr<MyNode> p1(new MyNode);
		left_node_ptr = move(p1);
		unique_ptr<MyNode> p2(new MyNode);
		right_node_ptr = move(p2);
	}
}

MyNode& MyNode::operator=(const MyNode &node) {
	if ( this != &node) {
		MyNode copy(node);
		threshold = copy.threshold;
		feature_indx = copy.feature_indx;
		label = copy.label;
		proba = copy.proba;
		is_leaf = copy.is_leaf;
		left_node_ptr = move(copy.left_node_ptr);
		right_node_ptr = move(copy.right_node_ptr);
	}
	return *this;
}

MyTree::MyTree() {}

MyTree::~MyTree() noexcept {}

MyTree::MyTree(int num_class, float lambda, float lr, int max_depth, int verbose, bool bootstrap)
	: splitter(num_class, lambda, lr), root_ptr(new MyNode(-1, -1, -1, true)),  max_depth(max_depth),
	  verbose(verbose), bootstrap(bootstrap) {}

MyTree::MyTree(MyTree const &tree)
	: splitter(tree.splitter), max_depth(max_depth), verbose(verbose), bootstrap(bootstrap),
	  root_ptr(move(make_unique<MyNode>(MyNode(*tree.root_ptr)))) {}

MyTree::MyTree(MyTree &&tree) noexcept
	: splitter(move(tree.splitter)), root_ptr(move(tree.root_ptr)) {}

MyTree &MyTree::operator=(const MyTree &tree) {
	if ( this != &tree) {
		MyTree copy(tree);
		root_ptr = move(copy.root_ptr);
		splitter = copy.splitter;
		max_depth = copy.max_depth;
		verbose = copy.verbose;
		bootstrap = copy.bootstrap;
	}
	return *this;
}

MyTree &MyTree::operator=(MyTree &&tree) noexcept {
	if ( this != &tree) {
		root_ptr = move(tree.root_ptr);
		splitter = move(tree.splitter);
		max_depth = tree.max_depth;
		verbose = tree.verbose;
		bootstrap = tree.bootstrap;
	}
	return *this;
}

void MyTree::fit(const std::vector< DataObject >  &Data_vec,
                 const std::vector< std::vector<float> > &unq_features) {
	std::vector< DataObject > global_vec;
	std::vector< DataObject* > ptr_vec;
	std::vector<  NodeInfoContainer >  vec1, vec2, *info_ptr = &vec1, *new_info_ptr = &vec2;

	for (int i = 0; i < Data_vec.size(); ++i) {
		int indx = i;
		if (bootstrap)
			indx = random() % Data_vec.size();
		if (verbose >= 1 and i == 0)
			cout << "> FIRST INDX IN BAGGING: " << indx << endl;
		global_vec.push_back(Data_vec[indx]);
	}
	for (int i = 0; i < Data_vec.size(); ++i)
		ptr_vec.push_back( &(global_vec[i]) );
	vec1.push_back(NodeInfoContainer(*(root_ptr), ptr_vec));

	int I = 0, depth = 0, nodes = 0;

	while (info_ptr->size() != 0 and depth < max_depth) {
		nodes += info_ptr->size();
		depth++;
		for (auto info : *info_ptr) {
			std::vector<DataObject*> left_data, right_data;
			int l_unq_indx, r_unq_indx = 0;

			tie(info.node_ptr->feature_indx, l_unq_indx, r_unq_indx, info.node_ptr->is_leaf) =
			    splitter.get_best(info.data, unq_features, verbose, depth);

			if (depth == max_depth)
				info.node_ptr->is_leaf = true;

			if (info.node_ptr->is_leaf) {
				tie(info.node_ptr->label, info.node_ptr->proba) =
				    splitter.get_label(info.data);
				continue;
			}

			vector<float> unq_vec = unq_features[info.node_ptr->feature_indx];

			info.node_ptr->threshold = (unq_vec[l_unq_indx] + unq_vec[r_unq_indx]) / 2.0;
			info.node_ptr->append_children();
			tie(left_data, right_data) =
			    splitter.split_data(info.data, l_unq_indx,
			                        info.node_ptr->feature_indx);

			new_info_ptr->push_back(
			    NodeInfoContainer((*info.node_ptr->left_node_ptr.get()), left_data));
			new_info_ptr->push_back(
			    NodeInfoContainer((*info.node_ptr->right_node_ptr.get()), right_data));
		}
		info_ptr->clear();
		auto tmp_ptr = info_ptr;
		info_ptr = new_info_ptr;
		new_info_ptr = tmp_ptr;
	}
	if (verbose >= 1) {
		cout << "Глубина дерева: " << depth << endl;
		cout << "Число Звеньев: " << nodes << endl;
	}
}

void MyTree::fit(const std::vector< std::vector<float> > &X,
                 const std::vector<int> &labels,
                 const std::vector< vector<float> > &probs) {
	std::vector< DataObject > global_vec;
	std::vector< std::vector<float> > unq_features;

	std::tie(global_vec, unq_features) = splitter.transform_data(X, labels, probs);
	fit(global_vec, unq_features);
}

std::tuple<std::vector<int>, std::vector< vector<float> > >
MyTree::predict(const std::vector< std::vector<float> > &X) {
	/*
		X size: feat_num*obj_num
	*/
	std::vector<int> result_labels;
	std::vector< vector<float> > result_probas;

	for (int i = 0; i < X[0].size(); ++i) {
		vector<float> obj;

		for (int f = 0; f < X.size(); ++f)
			obj.push_back(X[f][i]);

		MyNode *node_ptr = root_ptr.get();

		while (!node_ptr->is_leaf) {
			if (obj[node_ptr->feature_indx] < node_ptr->threshold)
				node_ptr = node_ptr->left_node_ptr.get();
			else
				node_ptr = node_ptr->right_node_ptr.get();
		}
		result_labels.push_back(node_ptr->label);
		result_probas.push_back(node_ptr->proba);
	}

	return make_tuple(result_labels, result_probas);
}