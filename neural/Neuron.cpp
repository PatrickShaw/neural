#include "Neuron.h"
#include "MathHelper.h"
namespace Neural {
	NeuronC::NeuronC(shared_ptr<vector<double>> weights) {
		this->weights = weights;
	}

	void NeuronC::AddWeight(double weight = 0) {
		this->weights->push_back(weight);
	}

	void NeuronC::RemoveNeuronWeight(size_t neuronIndex) {
		size_t weightIndex = neuronIndex + 1;
		this->weights->erase(this->weights->begin() + weightIndex);
	}

	void NeuronC::SetWeights(shared_ptr<vector<double>> weights) {
		this->weights = weights;
	}

	double NeuronC::GetWeight(size_t weightIndex) {
		return this->weights->at(weightIndex);
	}

	size_t NeuronC::GetWeightSize() {
		return this->weights->size();
	}

	double NeuronC::GetNeuronWeight(size_t weightIndex) {
		return this->weights->at(weightIndex + 1);
	}

	double NeuronC::GetThreshold() {
		return this->weights->at(0);
	}

	void NeuronC::SetWeight(size_t weightIndex, double weight) {
		this->weights->at(weightIndex) = weight;
	}

	void NeuronC::SetThresholdWeight(double weight) {
		this->weights->at(0) = weight;
	}

	void NeuronC::SetNeuronWeight(size_t neuronIndex, double weight) {
		this->weights->at(neuronIndex + 1) = weight;
	}

	double NeuronC::GetOutput(const vector<double>& inputs) {
		double output = -this->weights->at(0);
		for (size_t i = 0; i < inputs.size(); i++) {
			output += this->weights->at(i + 1) * inputs[i];
		}
		return MathHelper::Sigmoid(output);
	}

	shared_ptr<vector<double>> NeuronC::CloneWeights() {
		return make_shared<vector<double>>(*this->weights);
	}
}