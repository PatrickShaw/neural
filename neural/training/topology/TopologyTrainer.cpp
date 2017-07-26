#include "TopologyTrainer.h"
namespace neural {
	namespace training {
		namespace topology {
			shared_ptr<NeuralNetwork> TopologyTrainer::configure_topology(
				SupervisedTrainer<NeuralNetwork>& trainer,
				const vector<vector<double>>& trainingInputs,
				const vector<vector<double>>& trainingDesiredOutputs,
				const vector<vector<double>>& testInputs,
				const vector<vector<double>>& testDesiredOutputs
			) {
				vector<size_t> initialTopology = { trainingInputs[0].size(), trainingDesiredOutputs[0].size() };
				shared_ptr<NeuralNetwork> untrainedStubNetwork = make_shared<NeuralNetwork>(trainingInputs[0].size(), &initialTopology);
				shared_ptr<NeuralNetwork> trainedStubNetwork = untrainedStubNetwork->produce_new_neural_network();
				TopologyTrainer::train<NeuralNetwork>(trainer, *trainedStubNetwork, trainingInputs, trainingDesiredOutputs);
				double stubNetworkError = TopologyTrainer::total_error(*trainedStubNetwork, trainingInputs, trainingDesiredOutputs) + TopologyTrainer::total_error(*trainedStubNetwork, testInputs, testDesiredOutputs);
				while (true) {
					shared_ptr<NeuralNetwork> untrainedLayerNetwork = untrainedStubNetwork->produce_new_neural_network();
					untrainedLayerNetwork->insert_after(untrainedLayerNetwork->layer_size() - 1);
					shared_ptr<NeuralNetwork> trainedLayerNetwork = untrainedStubNetwork->produce_new_neural_network();
					TopologyTrainer::train<NeuralNetwork>(trainer, *trainedLayerNetwork, trainingInputs, trainingDesiredOutputs);
					double layerNetworkError = TopologyTrainer::total_error(*trainedLayerNetwork, trainingInputs, trainingDesiredOutputs) + TopologyTrainer::total_error(*trainedLayerNetwork, testInputs, testDesiredOutputs);
					while (true) {
						shared_ptr<NeuralNetwork> untrainedNeuronNetwork = untrainedLayerNetwork->produce_new_neural_network();
						untrainedNeuronNetwork->add_neuron_non_destructive(untrainedNeuronNetwork->layer_size() - 2);
						shared_ptr<NeuralNetwork> trainedNeuronNetwork = untrainedNeuronNetwork->produce_new_neural_network();
						TopologyTrainer::train<NeuralNetwork>(trainer, *trainedNeuronNetwork, trainingInputs, trainingDesiredOutputs);
						double neuronNetworkError = TopologyTrainer::total_error(*trainedNeuronNetwork, trainingInputs, trainingDesiredOutputs);
						if (neuronNetworkError < layerNetworkError) {
							untrainedLayerNetwork = untrainedNeuronNetwork;
							layerNetworkError = neuronNetworkError;
						}
						else {
							break;
						}
					}
					if (layerNetworkError < stubNetworkError) {
						untrainedStubNetwork = untrainedLayerNetwork;
						stubNetworkError = layerNetworkError;
					}
					else {
						break;
					}
				}
				return untrainedStubNetwork;
			}
			double TopologyTrainer::total_error(
				Classifier& trainedNetwork,
				const vector<vector<double>>& inputs,
				const vector<vector<double>>& desiredOutputs
			) {
				double error = 0;
				for (size_t t = 0; t < inputs.size(); t++) {
					shared_ptr<vector<double>> actualOutputs = trainedNetwork.classify(inputs[t]);
					for (size_t o = 0; o < actualOutputs->size(); o++) {
						error += TopologyTrainer::error(desiredOutputs.at(t).at(o), actualOutputs->at(o));
					}
				}
				return error;
			}
			double TopologyTrainer::error(
				const double desiredOutput,
				const double actualOutput
			) {
				double difference = desiredOutput - actualOutput;
				return difference * difference;
			}
			template<typename T>
			void TopologyTrainer::train(
				SupervisedTrainer<T>& trainer,
				T& network,
				const vector<vector<double>>& inputs,
				const vector<vector<double>>& desiredOutputs
			) {
				for (size_t _ = 0; _ < 100; _++) {
					for (size_t t = 0; t < inputs.size(); t++) {
						for (size_t i = 0; i < 50; i++) {
							trainer.train(network, inputs[t], desiredOutputs[t]);
						}
					}
				}
			}
		}
	}
}
