#pragma once
#include <limits>
#include <algorithm>
#include <vector>
#include <iostream>
#include <map>
#include <numeric>
#include <iterator>
#include <fstream>
#include <sstream>

using namespace std;

vector<float> getNextLineAndSplitIntoTokens(istream& str);

vector< vector<float> > getRaws(string name);

tuple< vector< vector<float> >, vector<int>> getData(string name);