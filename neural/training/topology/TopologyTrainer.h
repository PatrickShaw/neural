#pragma once
#include <memory>
#include <vector>
namespace neural::training::topology {
  using namespace std;
  class TopologyTrainer {
  public:
    static shared_ptr<NeuralNetwork> ConfigureTopology(
      SupervisedTrainer<NeuralNetwork>& trainer,
      vector<vector<double>>& trainingInputs,
      vector<vector<double>>& trainingDesiredOutputs,
      vector<vector<double>>& testInputs,
      vector<vector<double>>& testDesiredOutputs
    );
    static double total_error(
      Classifier trainedNetwork,
      vector<vector<double>>& inputs,
      vector<vector<double>>& desiredOutputs
    );
    static double error(
      double desiredOutput,
      double actualOutput
    );
    public static void train<T>(
      SupervisedTrainer<T> trainer,
      T network,
      vector<vector<double>>& inputs,
      vector<vector<double>>& desiredOutputs
    );
  }

}
