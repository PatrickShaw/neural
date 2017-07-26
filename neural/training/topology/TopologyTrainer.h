#pragma once
#include <memory>
#include <vector>
#include "../../NeuralNetwork.h"
#include "../SupervisedTrainer.h"
#include "../../Classifier.h"
namespace neural {
	namespace training {
		namespace topology {
			using namespace std;
			/**
			 * The TopologyTrainer intends to create a basic method of configuring the topology of a NeuralNetwork with 
			 * the help of a SupervisedTrainer.
			 * TODO: Abstract this class, you could make multiple types of TopologyTrainer.
			 */
			class TopologyTrainer {
			public:
				/**
				 * First, it starts off with a stub network, trains it and then remembers its best errorRate/accuracy.
				 * It then adds a neuron and repeats the process.
				 * If the network starts degarding in performance with extra neurons it adds a layer instead.
				 * It repeats the process again until the network starts degrading in performance with added layers.
				 * @returns An untrained neural network who's topology performed the best during the configuration
				 */
				static shared_ptr<NeuralNetwork> configure_topology(
					SupervisedTrainer<NeuralNetwork>& trainer,
					const vector<vector<double>>& trainingInputs,
					const vector<vector<double>>& trainingDesiredOutputs,
					const vector<vector<double>>& testInputs,
					const vector<vector<double>>& testDesiredOutputs
				);
				static double total_error(
					Classifier& trainedNetwork,
					const vector<vector<double>>& inputs,
					const vector<vector<double>>& desiredOutputs
				);
				static double error(
					double desiredOutput,
					double actualOutput
				);
				template<typename T>
				static void train(
					SupervisedTrainer<T>& trainer,
					T& network,
					const vector<vector<double>>& inputs,
					const vector<vector<double>>& desiredOutputs
				);
			};
		}
	}
}
