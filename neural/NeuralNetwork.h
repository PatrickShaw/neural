#pragma once
#include "Neuron.h"
#include "Classifier.h"
#include <memory>
#include <random>
namespace neural {
  using namespace std;
  /**
   * A standard sigmoid feed-forward neural network.
   * TODO: Possible have a Layer class? 
   * TODO: Change NeuralNetwork to work with a plethora of activation functions.
   */
  class NeuralNetwork : public Classifier {
  private:
	/** 
	 * The network of neurons.
	 */
    shared_ptr<vector<shared_ptr<vector<shared_ptr<Neuron>>>>> neurons;
  protected:
	/**
	 * Selects a layer in the neural network.
  	 */
    shared_ptr<vector<shared_ptr<Neuron>>> layer(size_t layerIndex);
	/**
	 * Selects a neuron in the neural network.
	 */
	shared_ptr<Neuron> neuron(size_t layerIndex, size_t neuronIndex);
  public:
	/**
	 * Clones a the neural network.
	 * TODO: I don't think this should be public.
	 */
	NeuralNetwork(const NeuralNetwork& network);
	NeuralNetwork(size_t inputCount, const vector<size_t>& neuralCounts);
	/**
	 * The weight of a neuron.
	 * @param layerIndex
	 * The index of the layer that the weight resides within.
	 * @param neuronIndex
	 * The index of the neruon that the weight resides within.
	 * @param weightIndex
	 * The index of the weight for the given neuron (This includes the threshold weight).
	 */
	double weight(size_t layerIndex, size_t neuronIndex, size_t weightIndex);
	/**
	 * Sets the weight for a given neuron on a given layer of the neural network.
	 */
	void set_weight(size_t layerIndex, size_t neuronIndex, size_t weightIndex, double weight);
	/**
	 * @returns The number of layers in the NeuralNetwork
	 */
	size_t layer_size();
	/**
	 * The required value for the threshold of a neuron to output 0.
	 */
    double threshold_to_result_in_zero();
	/**
	 * The required value for a weight that links two neurons to produce an output of 0 regardless of the input value.
	 */
    double inactive_neuron_weight();
	/**
	 * Randomizes the weights for all neurons in the network
	 */
    void randomize_weights(double min, double max);
	/**
	 * Creates an array of weights that will cause a neuron to output zero regardless of the input.
	 */
    shared_ptr<vector<double>> create_inactive_neuron_weights(size_t weightCount);
	/**
	 * The output of the network without normalizing the output.
	 */
    shared_ptr<vector<double>> raw_outputs(const vector<double>& inputs);
	/**
	 * All the outputs from each layer of the network given an array of inputs.
	 */
    shared_ptr<vector<shared_ptr<vector<double>>>> all_outputs(const vector<double>& inputs);
	/**
	 * The number of neurons in a given layer.
	 */
    size_t neuron_size(size_t layerIndex);
	/**
	 * The number of neurons in the input layer.
	 */
	size_t input_size();
	/**
	 * The number of neurons in the output layer.
	 */
	size_t output_size();
	/**
     * Inserts a layer into the given index with the same neuron count as the previous layer. 
     * Each neuron in the layer only acknowledges the input of the neuron from the previous layer with the same index.
     * Warning: This will affect the output values of the network.
     **/
    void insert_after(size_t layerIndex);
    /**
	 * Removes a neuron from a layer in the network.
	 */
	void remove_neuron(size_t layerIndex, size_t neuronIndex);
	/**
	 * Removes a layer from the network.
	 */
    void remove_layer(size_t layerIndex);
	/**
	 * Inserts a layer into the network.
	 */
    void insert_layer(size_t layerIndex, shared_ptr<vector<shared_ptr<Neuron>>> layer);
	/**
	 * Adds a neuron to the output layer.
	 */
    void add_output_neuron(shared_ptr<Neuron> neuron);
	/**
	 * Adds a neuron to anything other than the output layer.
	 */
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
	/**
	 * Produces a new neural network.
	 */
    shared_ptr<NeuralNetwork> produce_new_neural_network();
	shared_ptr<vector<double>> classify(const vector<double>& inputs);
  };
}