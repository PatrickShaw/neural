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
		shared_ptr<vector<double>> GetWeights();
		void AddWeight(double weight);
		void RemoveNeuronWeight(int neuronIndex);
		void SetWeights(shared_ptr<vector<double>> weights);
		double GetNeuronWeight(int neuronIndex);
		double GetThreshold();
		void SetWeight(int weightIndex, double weight);
		void SetThresholdWeight(double weight);
		void SetNeuronWeight(int neuronIndex, double weight);
		double GetOutput(vector<double>& inputs);
	};
}