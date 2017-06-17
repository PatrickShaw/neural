#pragma once
#include <vector>
#include <memory>
namespace Neural {
	using namespace std;
	class NeuronC {
	private:
		shared_ptr<vector<double>> weights;
	public:
		NeuronC(shared_ptr<vector<double>> weights);
		double GetWeight(size_t weightIndex);
		void AddWeight(double weight);
		void RemoveNeuronWeight(size_t neuronIndex);
		void SetWeights(shared_ptr<vector<double>> weights);
		double GetNeuronWeight(size_t neuronIndex);
		double GetThreshold();
		void SetWeight(size_t weightIndex, double weight);
		void SetThresholdWeight(double weight);
		void SetNeuronWeight(size_t neuronIndex, double weight);
		double GetOutput(const vector<double>& inputs);
		size_t GetWeightSize();
		shared_ptr<vector<double>> CloneWeights();
	};
}