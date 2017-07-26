#pragma once
#include <memory>
#include <vector>
#include "../SupervisedTrainer.h"
#include "../../NeuralNetwork.h"
namespace neural {
	namespace training {
		namespace backpropagation {
			using namespace std;
			/**
			 * A supervised trainer that uses standard backpropgation to train a NeuralNetwork
			 */
			class StandardBackpropagationTrainer : SupervisedTrainer<NeuralNetwork> {
			public:
				/**
				 * Performs backpropagation supervised training on a NeuralNetwork.
				 * Note that this method does exactly the same thing as train but allows you to specify 
				 * the learning rate.
				 * TODO: Return partial differentials and changes in weightings.
				 */
				void backpropagation(
					NeuralNetwork& neuralNetwork, 
					const vector<double>& inputs, 
					const vector<double>& desiredOutputs, 
					double learningRateFactor = 0.1
				);
				/**
				 * Note: It is recommended that the backpropagation method is used instead of this one.
				 */
				void train(
					NeuralNetwork& trainable, 
					const vector<double>& trainingInputs,
					const vector<double>& trainingOutputs
				);
			};
		}
	}
}