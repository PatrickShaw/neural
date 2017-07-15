#pragma once
#include <memory>
#include <vector>
#include "./StandardBackpropagationTrainer.h"
namespace neural {
	namespace training {
		namespace backpropagation {
			void StandardBackpropagationTrainer::backpropagation(
				NeuralNetwork& neuralNetwork, 
				const vector<double>& inputs, 
				const vector<double>& desiredOutputs, 
				double learningRateFactor
			) {

			}
			void StandardBackpropagationTrainer::train(
				NeuralNetwork& trainable, 
				const vector<double>& trainingInputs, 
				const vector<double>& trainingOutputs
			) {
				
			}
		}
	}
}