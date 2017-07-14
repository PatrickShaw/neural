#pragma once
#include <memory>
#include <vector>
namespace neural {
	namespace training {
		namespace backpropagation {
			using namespace std;
			class StandardBackpropagationTrainer : SupervisedTrainer<NeuralNetwork> {
			public:
				void backpropagation(NeuralNetwork& neuralNetwork, const vector<double>& inputs, const vector<double>& desiredOutputs, double learningRateFactor = 0.1);

				void train(NeuralNetwork& trainable, const vector<double>& trainingInputs, const vector<double>& trainingOutputs);
			}
		}
	}
