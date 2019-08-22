#include <vector>
#include <iostream>
#include <tuple>
#include <memory>
#include "MyTree.h"


using namespace std;

NodeInfoContainer::NodeInfoContainer(MyNode &current_node,
                                     const std::vector<DataObject*> &data) {
	this->node_ptr = &current_node;
	this->data = data;
}

MyNode::MyNode() {}

MyNode::~MyNode() {}

MyNode::MyNode(float threshold, int feature_indx, int label, bool is_leaf) {
	this->threshold = threshold;
	this->feature_indx = feature_indx;
	this->label = label;
	this->is_leaf = is_leaf;
}

MyNode::MyNode(MyNode const &node) {
	cout << ">> COPY CONSTRUCTOR <<" << endl;
	this->threshold = node.threshold;
	this->feature_indx = node.feature_indx;
	this->is_leaf = node.is_leaf;

	if (node.is_leaf) {
		this->left_node_ptr = nullptr;
		this->right_node_ptr = nullptr;
		this->label = node.label;
		this->proba = node.proba;
	}

	else {
		MyNode left_child(*node.left_node_ptr);
		MyNode right_child(*node.right_node_ptr);
		unique_ptr<MyNode> lp = make_unique<MyNode>(left_child);
		unique_ptr<MyNode> rp = make_unique<MyNode>(right_child);
		this->left_node_ptr = move(lp);
		this->right_node_ptr = move(rp);
	}
}

void MyNode::append_children() {
	if (this->is_leaf) {
		unique_ptr<MyNode> p1(nullptr);
		this->left_node_ptr = move(p1);
		unique_ptr<MyNode> p2(nullptr);
		this->right_node_ptr = move(p2);
	} else {
		unique_ptr<MyNode> p1(new MyNode);
		this->left_node_ptr = move(p1);
		unique_ptr<MyNode> p2(new MyNode);
		this->right_node_ptr = move(p2);
	}
}

MyNode& MyNode::operator=(const MyNode &node) {
	if ( this != &node) {
		MyNode copy(node);
		this->threshold = copy.threshold;
		this->feature_indx = copy.feature_indx;
		this->label = copy.label;
		this->proba = copy.proba;
		this->is_leaf = copy.is_leaf;
		this->left_node_ptr = move(copy.left_node_ptr);
		this->right_node_ptr = move(copy.right_node_ptr);
	}

	return *this;
}

MyTree::MyTree() {}

MyTree::~MyTree() noexcept
{}

MyTree::MyTree(int num_class, float lambda, float lr, int max_depth, int verbose,
               bool bootstrap): splitter(num_class, lambda, lr), root_ptr(new MyNode(-1, -1, -1, true)) {
	this->max_depth = max_depth;
	this->verbose = verbose;
	this->bootstrap = bootstrap;
}

MyTree::MyTree(MyTree const &tree): splitter(tree.splitter) {
	MyNode new_root(*tree.root_ptr);
	this->root_ptr = move(make_unique<MyNode>(new_root));
	this->max_depth = max_depth;
	this->verbose = verbose;
	this->bootstrap = bootstrap;
}

MyTree::MyTree(MyTree &&tree) noexcept
	: splitter(move(tree.splitter)), root_ptr(move(tree.root_ptr)) {}

MyTree &MyTree::operator=(const MyTree &tree) {
	if ( this != &tree) {
		MyTree copy(tree);
		this->root_ptr = move(copy.root_ptr);
		this->splitter = copy.splitter;
		this->max_depth = max_depth;
		this->verbose = verbose;
		this->bootstrap = bootstrap;
	}

	return *this;
}

MyTree &MyTree::operator=(MyTree &&tree) noexcept {
	if ( this != &tree) {
		this->root_ptr = move(tree.root_ptr);
		this->splitter = move(tree.splitter);
		this->max_depth = max_depth;
		this->verbose = verbose;
		this->bootstrap = bootstrap;
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
		if (this->bootstrap)
			indx = random() % Data_vec.size();
		if (this->verbose >= 1 and i == 0)
			cout << "> FIRST INDX IN BAGGING: " << indx << endl;
		global_vec.push_back(Data_vec[indx]);
	}
	for (int i = 0; i < Data_vec.size(); ++i)
		ptr_vec.push_back( &(global_vec[i]) );
	vec1.push_back(NodeInfoContainer(*(this->root_ptr), ptr_vec));

	int I = 0, depth = 0, nodes = 0;

	while (info_ptr->size() != 0 and depth < max_depth) {
		nodes += info_ptr->size();
		depth++;
		for (auto info : *info_ptr) {
			std::vector<DataObject*> left_data, right_data;
			int l_unq_indx, r_unq_indx = 0;

			tie(info.node_ptr->feature_indx, l_unq_indx, r_unq_indx, info.node_ptr->is_leaf) =
			    this->splitter.get_best(info.data, unq_features, this->verbose, depth);

			if (depth == max_depth)
				info.node_ptr->is_leaf = true;

			if (info.node_ptr->is_leaf) {
				tie(info.node_ptr->label, info.node_ptr->proba) =
				    MyTree::splitter.get_label(info.data);
				continue;
			}

			vector<float> unq_vec = unq_features[info.node_ptr->feature_indx];

			info.node_ptr->threshold = (unq_vec[l_unq_indx] + unq_vec[r_unq_indx]) / 2.0;
			info.node_ptr->append_children();
			tie(left_data, right_data) =
			    this->splitter.split_data(info.data, l_unq_indx,
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
	if (this->verbose >= 1) {
		cout << "Глубина дерева: " << depth << endl;
		cout << "Число Звеньев: " << nodes << endl;
	}
}

void MyTree::fit(const std::vector< std::vector<float> > &X,
                 const std::vector<int> &labels,
                 const std::vector< vector<float> > &probs) {
	std::vector< DataObject > global_vec;
	std::vector< std::vector<float> > unq_features;

	std::tie(global_vec, unq_features) = this->splitter.transform_data(X, labels, probs);
	this->fit(global_vec, unq_features);
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

		MyNode *node_ptr = this->root_ptr.get();

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