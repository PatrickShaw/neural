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
			class TopologyTrainer {
			public:
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
