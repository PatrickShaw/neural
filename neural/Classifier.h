#pragma once
#include <vector>
#include <memory>
namespace neural {
  using namespace std;
  class Classifier {
  public:
	/**
	* Outputs a set of values for a given set of inputs.
	*/
    virtual shared_ptr<vector<double>> classify(const vector<double>& inputs) = 0;
  };
}
