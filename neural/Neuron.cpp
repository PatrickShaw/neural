#include "Neuron.h"
#include "MathHelper.h"
namespace Neural {
	NeuronC::NeuronC(vector<double>& weights) {
		this->weights = weights;
	}

	void NeuronC::AddWeight(double weight = 0) {
		this->weights.push_back(weight);
	}

	void NeuronC::RemoveNeuronWeight(size_t neuronIndex) {
		size_t weightIndex = neuronIndex + 1;
		this->weights.erase(this->weights.begin() + weightIndex);
	}

	void NeuronC::SetWeights(vector<double>& weights) {
		this->weights = weights;
	}

	double NeuronC::GetWeight(size_t weightIndex) {
		return this->weights[weightIndex];
	}

	size_t NeuronC::GetWeightSize() {
		return this->weights.size();
	}

	double NeuronC::GetNeuronWeight(size_t weightIndex) {
		return this->weights[weightIndex + 1];
	}

	double NeuronC::GetThreshold() {
		return this->weights[0];
	}

	void NeuronC::SetWeight(size_t weightIndex, double weight) {
		this->weights[weightIndex] = weight;
	}

	void NeuronC::SetThresholdWeight(double weight) {
		this->weights[0] = weight;
	}

	void NeuronC::SetNeuronWeight(size_t neuronIndex, double weight) {
		this->weights[neuronIndex + 1] = weight;
	}

	double NeuronC::GetOutput(vector<double>& inputs) {
		double output = -this->weights[0];
		for (size_t i = 0; i < inputs.size(); i++) {
			output += this->weights[i + 1] * inputs[i];
		}
		return MathHelper::Sigmoid(output);
	}

	vector<double>& NeuronC::CloneWeights() {
		vector<double> clonedWeights(this->GetWeightSize());
		for each(double weight in this->weights) {
			clonedWeights.push_back(weight);
		}
		return clonedWeights;
	}
}