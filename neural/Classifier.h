#pragma once
#include <vector>
#include <memory>
namespace neural {
  using namespace std;
  class Classifier {
  public:
    virtual shared_ptr<vector<double>> classify(const vector<double>& inputs) = 0;
  };
}
