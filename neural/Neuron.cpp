#include "Neuron.h"
#include "MathHelper.h"
namespace Neural {
	NeuronC::NeuronC(shared_ptr<vector<double>> weights) {
		this->weights = weights;
	}

	shared_ptr<vector<double>> NeuronC::GetWeights() {
		return this->weights;
	}

	void NeuronC::AddWeight(double weight = 0) {
		this->weights->push_back(weight);
	}

	void NeuronC::RemoveNeuronWeight(int neuronIndex) {
		size_t weightIndex = neuronIndex + 1;
		this->weights->erase(this->weights->begin() + weightIndex);
	}

	void NeuronC::SetWeights(shared_ptr<vector<double>> weights) {
		this->weights = weights;
	}

	double NeuronC::GetNeuronWeight(int neuronIndex) {
		return (*this->weights)[neuronIndex + 1];
	}

	double NeuronC::GetThreshold() {
		return (*this->weights)[0];
	}

	void NeuronC::SetWeight(int weightIndex, double weight) {
		(*this->weights)[weightIndex] = weight;
	}

	void NeuronC::SetThresholdWeight(double weight) {
		(*this->weights)[0] = weight;
	}

	void NeuronC::SetNeuronWeight(int neuronIndex, double weight) {
		(*this->weights)[neuronIndex + 1] = weight;
	}

	double NeuronC::GetOutput(vector<double>& inputs) {
		double output = -(*this->weights)[0];
		for (size_t i = 0; i < inputs.size(); i++) {
			output += (*this->weights)[i + 1] * inputs[i];
		}
		return MathHelper::Sigmoid(output);
	}
}