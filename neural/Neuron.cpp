#include "Neuron.h"
#include "MathHelper.h"
namespace neural {
	Neuron::Neuron(shared_ptr<vector<double>> weights) {
		this->weights = weights;
	}

	void Neuron::push_weight(double weight = 0) {
		this->weights->push_back(weight);
	}

	void Neuron::remove_weight(size_t neuronIndex) {
		size_t weightIndex = neuronIndex + 1;
		this->weights->erase(this->weights->begin() + weightIndex);
	}

	void Neuron::set_weights(shared_ptr<vector<double>> weights) {
		this->weights = weights;
	}

	double Neuron::weight(size_t weightIndex) {
		return this->weights->at(weightIndex);
	}

	size_t Neuron::weight_size() {
		return this->weights->size();
	}

	double Neuron::neuron_weight(size_t weightIndex) {
		return this->weights->at(weightIndex + 1);
	}

	double Neuron::threshold() {
		return this->weights->at(0);
	}

	void Neuron::set_weight(size_t weightIndex, double weight) {
		this->weights->at(weightIndex) = weight;
	}

	void Neuron::set_threshold(double weight) {
		this->weights->at(0) = weight;
	}

	void Neuron::set_neuron_weight(size_t neuronIndex, double weight) {
		this->weights->at(neuronIndex + 1) = weight;
	}

	double Neuron::output(const vector<double>& inputs) {
		double output = -this->weights->at(0);
		for (size_t i = 0; i < inputs.size(); i++) {
			output += this->weights->at(i + 1) * inputs[i];
		}
		return MathHelper::sigmoid(output);
	}

	shared_ptr<vector<double>> Neuron::clone_weights() {
		return make_shared<vector<double>>(*this->weights);
	}
}