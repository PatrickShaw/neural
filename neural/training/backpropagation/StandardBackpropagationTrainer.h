#pragma once
#include <memory>
#include <vector>
namespace neural {
  using namespace std;
  class Classifier {
    shared_ptr<vector<double>> classify(const vector<double>& inputs);
  };
}
