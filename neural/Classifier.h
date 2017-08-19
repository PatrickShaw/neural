#pragma once
#include <vector>
#include <memory>
using namespace std;
namespace neural {
  class Classifier {
  public:
	/**
	* Outputs a set of values for a given set of inputs.
	*/
    virtual shared_ptr<vector<double>> classify(const vector<double>& inputs) = 0;
  };
}
