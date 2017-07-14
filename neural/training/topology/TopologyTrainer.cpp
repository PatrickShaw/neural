#pragma once
#include <memory>
#include <vector>
#include "./TopologyTrainer.h"
namespace neural {
	namespace training {
		namespace toplogy {
			using namespace std;
			class TopologyTrainer {
			public:
				static NeuralNetwork configure_topology(SupervisedTrainer<NeuralNetwork> trainer, double[][] trainingInputs, double[][] trainingDesiredOutputs, double[][] testInputs, double[][] testDesiredOutputs);
			};
		}
	}
}
