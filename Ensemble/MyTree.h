#pragma once
#include <vector>
#include <memory>
#include "MySplitter.h"

using namespace std;


class MyNode {
  public:
	float threshold;
	int feature_indx;
	int label;
	vector<float> proba;
	bool is_leaf; // True => leaf
	std::unique_ptr< MyNode> left_node_ptr;
	std::unique_ptr< MyNode> right_node_ptr;

	MyNode();

	~MyNode();

	MyNode(float threshold, int feature_indx, int label, bool is_leaf);

	MyNode(MyNode const & node);

	void append_children();

	MyNode& operator=(const MyNode& node);
};


class NodeInfoContainer {
	/*
		Container contains node and data, which
		will be used for next node creation, or for tree walk
	*/
  public:
	MyNode *node_ptr;
	std::vector<DataObject*> data;

	NodeInfoContainer(MyNode &current_node, const std::vector<DataObject*> &data);
};


class MyTree {
  public:
	MySplitterClass splitter;
	unique_ptr<MyNode> root_ptr;
	int max_depth;
	int verbose;
	bool bootstrap;

	MyTree();

	~MyTree() noexcept;

	MyTree(const MyTree &tree);

	MyTree(MyTree &&tree) noexcept;

	MyTree(int num_class, float lambda, float lr, int max_depth, int verbose, bool bootstrap);

	MyTree& operator=(const MyTree &node);

	MyTree& operator=(MyTree &&node) noexcept;

	void fit(const std::vector< DataObject >  &Data_vec,
	         const std::vector< std::vector<float> > &unq_features);

	void fit(const std::vector< std::vector<float> > &X,
	         const std::vector<int> &labels,
	         const std::vector< vector<float> > &probs);

	std::tuple<std::vector<int>, std::vector< vector<float> > >
	predict(const std::vector< std::vector<float> > &X);
};