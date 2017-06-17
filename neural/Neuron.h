#pragma once
#include <vector>
#include <memory>
namespace neural {
	using namespace std;
	class Neuron {
	private:
		shared_ptr<vector<double>> weights;
	public:
		Neuron(shared_ptr<vector<double>> weights);
		double weight(size_t weightIndex);
		void push_weight(double weight);
		void remove_weight(size_t neuronIndex);
		void set_weights(shared_ptr<vector<double>> weights);
		double neuron_weight(size_t neuronIndex);
		double threshold();
		void set_weight(size_t weightIndex, double weight);
		void set_threshold(double weight);
		void set_neuron_weight(size_t neuronIndex, double weight);
		double output(const vector<double>& inputs);
		size_t weight_size();
		shared_ptr<vector<double>> clone_weights();
	};
}