#pragma once
#include <vector>
#include <memory>
using namespace std;
namespace neural {
  class IReadonlyNeuron {
		virtual double weight(size_t weightIndex) = 0;
		virtual double neuron_weight(size_t neuronIndex) = 0;
		virtual double threshold() = 0;
		virtual double output(const vector<double>& inputs) = 0;
		virtual size_t weight_size() = 0;
		virtual shared_ptr<vector<double>> clone_weights() = 0;
  };
}
