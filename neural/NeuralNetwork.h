#pragma once
#include "Neuron.h"
#include "Classifier.h"
#include <memory>
#include <random>
namespace neural {
  using namespace std;
  class NeuralNetwork : Classifier {
  private:
    shared_ptr<vector<shared_ptr<vector<shared_ptr<Neuron>>>>> neurons;
  protected:
    NeuralNetwork(const NeuralNetwork& network);
    std::shared_ptr<std::vector<std::shared_ptr<Neuron>>> NeuralNetwork::layer(size_t layerIndex);
  public:
    NeuralNetwork(size_t inputCount, const vector<int>& neuralCounts);
    size_t layer_size();
    double threshold_to_result_in_zero();
    double inactive_neuron_weight();
    void randomize_weights(double min, double max);
    shared_ptr<vector<double>> create_inactive_neuron_weights(size_t weightCount);
    shared_ptr<vector<double>> raw_outputs(vector<double>& inputs);
    shared_ptr<vector<shared_ptr<vector<double>>>> all_outputs(vector<double>& inputs);
    size_t neuron_size(size_t layerIndex);
    /**
     * Inserts a layer into the given index with the same neuron count as the previous layer. 
     * Each neuron in the layer only acknowledges the input of the neuron from the previous layer with the same index.
     * Warning: This will affect the output values of the network.
     **/
    void insert_after(size_t layerIndex);
    void remove_neuron(size_t layerIndex, size_t neuronIndex);
    void remove_layer(size_t layerIndex);
    void insert_layer(size_t layerIndex, shared_ptr<vector<shared_ptr<Neuron>>> layer);
    void add_output_neuron(shared_ptr<Neuron> neuron);
    void add_non_output_neuron(size_t layerIndex, shared_ptr<Neuron> neuron, const vector<double>& outputWeights);
    /**
     * Splits a neuron into 2 that produce half the output of the original neuron.
     * This effectively adds a neuron without causing the network's behaviour/outputs to change.
     **/
    void split_neuron_non_destructive(size_t layerIndex, size_t neuronIndex);
    /**
     * Adds a new neuron that is not affected by any neurons from the previous layer and outputs 0 (i.e. always outputs 0).
     * This effectively adds a neuron without causing the network's behaviour/outputs to change.
     */
    void add_neuron_non_destructive(size_t layerIndex);
    shared_ptr<NeuralNetwork> produce_new_neural_network();
  };
}