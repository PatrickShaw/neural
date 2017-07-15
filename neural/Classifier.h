#pragma once
#include <memory>
namespace neural {
  using namespace std;
  class Classifier {
  public:
    shared_ptr<vector<double>> classify(const vector<double>& inputs);
  };
}
