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
				// We need to calculate the current outputs, given a set of inputs in order to do backpropagation.
				shared_ptr<vector<shared_ptr<vector<double>>>> outputs = neuralNetwork.all_outputs(inputs);
				// TODO: Revisit 'weirdDThing'.
				vector<vector<double>> weirdDThing = vector<vector<double>>(outputs->size());
				for (size_t l = 0; l < outputs->size(); l++) {
					weirdDThing[l] = vector<double>(outputs->at(l)->size());
				}
				for (size_t n = 0; n < outputs->at(outputs->size() - 1)->size(); n++) {
					double neuronOutput = outputs->at(outputs->size() - 1)->at(n);
					weirdDThing.at(weirdDThing.size() - 1).at(n) = (neuronOutput - desiredOutputs[n]) * neuronOutput * (1 - neuronOutput);
				}
				for (size_t l = outputs->size() - 2; l >= 0; l--) {
					size_t inputLength = neuralNetwork.neuron_size(l);
					for (size_t n = 0; n < inputLength; n++) {
						double neuronOutput = outputs->at(l)->at(n);
						double sumThing = 0;
						for (size_t n2 = 0; n2 < neuralNetwork.neuron_size(l + 1); n2++) {
							sumThing += weirdDThing.at(l + 1).at(n2) * neuralNetwork.weight(l, n2, n);
						}
						weirdDThing[l][n] = sumThing * neuronOutput * (1 - neuronOutput);
					}
				}
				// Now we actually modify the the weights of the neurons.
				// The first layer is a special case it doesn't have any previous layers to deal with.
				for (size_t n = 0; n < neuralNetwork.input_size(); n++) {
					// Modify the threshold's weight
					// The threshold/bias takes a -1 as input
					double newThresholdWeight = neuralNetwork.weight(0, n, 0);
					newThresholdWeight -= learningRateFactor * weirdDThing.at(0).at(n) * -1;
					neuralNetwork.set_weight(0, n, 0, newThresholdWeight);
					// Modify the neuron to input weights
					for (size_t n2 = 0; n2 < inputs.size(); n2++) {
						size_t weightIndex = n2 + 1;
						double newNeuronToInputWeight = neuralNetwork.weight(0, n, weightIndex);
						newNeuronToInputWeight -= learningRateFactor * weirdDThing.at(0).at(n) * inputs.at(n2);
						neuralNetwork.set_weight(0, n, weightIndex, newNeuronToInputWeight);
					}
				}
				// Now modify the weights for the other neural networks.
				for (size_t l = 1; l < neuralNetwork.layer_size(); l++) {
					for (size_t n = 0; n < neuralNetwork.neuron_size(l); n++) {
						double newThresholdWeight = neuralNetwork.weight(l, n, 0);
						newThresholdWeight -= learningRateFactor * weirdDThing.at(l).at(n) * -1;
						neuralNetwork.set_weight(l, n, 0, newThresholdWeight);
						for (size_t n2 = 0; n2 < neuralNetwork.neuron_size(l - 1); n2++) {
							size_t weightIndex = n2 + 1;
							double newNeuronToNeuronWeight = neuralNetwork.weight(l - 1, n2, weightIndex);
							newNeuronToNeuronWeight -= learningRateFactor * weirdDThing.at(l).at(n) * outputs->at(l - 1)->at(n2);
							neuralNetwork.set_weight(l-1, n2, weightIndex, newNeuronToNeuronWeight);
						}
					}
				}
			}

			void StandardBackpropagationTrainer::train(
				NeuralNetwork& trainable, 
				const vector<double>& trainingInputs, 
				const vector<double>& trainingOutputs
			) {
				this->backpropagation(trainable, trainingInputs, trainingOutputs);
			}
		}
	}
}