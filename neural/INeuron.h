#pragma once
#include <vector>
#include <memory>
#include "./IReadonlyNeuron.h";
using namespace std;
namespace neural {
  class INeuron : IReadonlyNeuron {
		virtual void push_weight(double weight) = 0;
		virtual void remove_neuron_weight(size_t weightIndex) = 0;
		virtual void set_weights(shared_ptr<vector<double>> weights) = 0;
		virtual void set_weight(size_t weightIndex, double weight) = 0;
		virtual void set_threshold(double weight) = 0;
		virtual void set_neuron_weight(size_t neuronIndex, double weight) = 0;
  };
}
