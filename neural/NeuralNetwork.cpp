#include "NeuralNetwork.h"
namespace neural {
    NeuralNetwork::NeuralNetwork(const NeuralNetwork& network) {
      this->neurons = make_shared<vector<shared_ptr<vector<shared_ptr<Neuron>>>>>(network.neurons->size());
      for (size_t l = 0; l < this->neurons->size(); l++) {
        this->neurons->at(l) = make_shared<vector<shared_ptr<Neuron>>>(network.neurons->at(l)->size());
        for (size_t n = 0; n < this->neurons->at(l)->size(); n++) {
          this->neurons->at(l)->at(n) = make_shared<shared_ptr<Neuron>>(network.neurons->at(l)->at(n)->clone_weights());
        }
      }
    }

	size_t NeuralNetwork::input_size() {
		return this->neuron_size(0);
	}

	size_t NeuralNetwork::output_size() {
		return this->neuron_size(this->layer_size() - 1);
	}

    NeuralNetwork::NeuralNetwork(size_t inputCount, const vector<size_t>& neuralCounts) {
      if (neuralCounts.size() < 1) { throw; }
      this->neurons = make_shared<vector<shared_ptr<vector<shared_ptr<Neuron>>>>>(neuralCounts.size());
      this->neurons->at(0) = make_shared<vector<shared_ptr<Neuron>>>(neuralCounts.at(0));
      for (size_t n = 0; n < neurons->at(0)->size(); n++) {
        this->neurons->at(0)->at(n) = make_shared<Neuron>(this->create_inactive_neuron_weights(inputCount + 1));
      }
      for (size_t l = 1; l < neuralCounts.size(); l++) {
        this->neurons->at(l) = make_shared<vector<shared_ptr<Neuron>>>(neuralCounts.at(l));
        for (int n = 0; n < neurons->at(l)->size(); n++) {
          size_t weight_size = this->neurons->at(l - 1)->size();
          this->neurons->at(l)->at(n) = make_shared<Neuron>(this->create_inactive_neuron_weights(weight_size));
        }
      }
    }

	shared_ptr<Neuron> NeuralNetwork::neuron(size_t layerIndex, size_t neuronIndex) {
		return this->layer(layerIndex)->at(neuronIndex);
	}

	double NeuralNetwork::weight(size_t layerIndex, size_t neuronIndex, size_t weightIndex) {
		return this->neuron(neuronIndex, layerIndex)->weight(weightIndex);
	}

	void NeuralNetwork::set_weight(size_t layerIndex, size_t neuronIndex, size_t weightIndex, double weight) {
		this->neuron(neuronIndex, layerIndex)->set_weight(weightIndex, weight);
	}

    size_t NeuralNetwork::layer_size() {
      return this->neurons->size();
    }

    double NeuralNetwork::threshold_to_result_in_zero() {
      return 6;
    }

    double NeuralNetwork::inactive_neuron_weight() {
      return 0;
    }

    void NeuralNetwork::randomize_weights(double min, double max) {
      std::random_device rd;
      std::mt19937 gen(rd());
      uniform_real_distribution<double> dist(-6, 6);
      for (size_t l = 0; l < this->neurons->size(); l++) {
        for (size_t n = 0; n < neurons->at(l)->size(); n++) {
          for (size_t w = 0; w < neurons->at(l)->at(n)->weight_size(); w++) {
            this->neurons->at(l)->at(n)->set_weight(w, dist(gen));
          }
        }
      }
    }

    shared_ptr<vector<double>> NeuralNetwork::create_inactive_neuron_weights(size_t weightCount) {
      shared_ptr<vector<double>> weights = make_shared<vector<double>>(weightCount);
      weights->at(0) = this->threshold_to_result_in_zero();
      for (size_t w = 1; w < weights->size(); w++) {
        weights->at(w) = this->inactive_neuron_weight();
      }
      return weights;
    }

    shared_ptr<vector<double>> NeuralNetwork::raw_outputs(const vector<double>& inputs) {
      return this->all_outputs(inputs)->at(0);
    }

    shared_ptr<vector<shared_ptr<vector<double>>>> NeuralNetwork::all_outputs(const vector<double>& inputs) {
      shared_ptr<vector<shared_ptr<vector<double>>>> outputs = make_shared<vector<shared_ptr<vector<double>>>>(this->neurons->size());
      outputs->at(0) = make_shared<vector<double>>(this->neurons->at(0)->size());
      for (size_t n = 0; n < this->neurons->at(0)->size(); n++) {
        this->neurons->at(0)->at(n)->output(inputs);
      }
      for (size_t l = 1; l < this->neurons->size(); l++) {
        for (size_t n = 0; n < this->neurons->at(n)->size(); n++) {
          this->neurons->at(l)->at(n)->output(*outputs->at(l - 1));
        }
      }
      return outputs;
    }

    shared_ptr<vector<shared_ptr<Neuron>>> NeuralNetwork::layer(size_t layerIndex) {
      return this->neurons->at(layerIndex);
    }

    size_t NeuralNetwork::neuron_size(size_t layerIndex) {
      return this->layer(layerIndex)->size();
    }

    /**
    * Inserts a layer into the given index with the same neuron count as the previous layer.
    * Each neuron in the layer only acknowledges the input of the neuron from the previous layer with the same index.
    * Warning: This will affect the output values of the network.
    **/
    void NeuralNetwork::insert_after(size_t layerIndex) {
      shared_ptr<vector<shared_ptr<Neuron>>> layer = make_shared<vector<shared_ptr<Neuron>>>(this->neurons->at(layerIndex)->size());
      for (size_t n = 0; n < layer->size(); n++) {
        shared_ptr<vector<double>> inputWeights = make_shared<vector<double>>(this->neurons->at(layerIndex)->size() + 1);
        layer->at(n) = make_shared<Neuron>(inputWeights);
      }
      this->insert_layer(layerIndex + 1, layer);
    }

    void NeuralNetwork::remove_neuron(size_t layerIndex, size_t neuronIndex) {
      size_t originalLayerLength = neurons->at(layerIndex)->size();
      if (originalLayerLength <= 1) {
        this->remove_layer(layerIndex);
      } else {
        shared_ptr<vector<shared_ptr<Neuron>>> newLayer = make_shared<vector<shared_ptr<Neuron>>>(originalLayerLength - 1);
        for (size_t n = 0; n < neuronIndex; n++) {
          newLayer->at(n) = this->neurons->at(layerIndex)->at(n);
        }
        for (size_t n = neuronIndex + 1; n < this->neurons->at(layerIndex)->size(); n++) {
          newLayer->at(n - 1) = this->neurons->at(layerIndex)->at(n);
        }
        this->neurons->at(layerIndex) = newLayer;
        if (layerIndex != this->neurons->size() - 1) {
          for (size_t n2 = 0; n2 < this->neurons->at(layerIndex + 1)->size(); n2++) {
            this->neurons->at(layerIndex + 1)->at(n2)->remove_neuron_weight(neuronIndex);
          }
        }
      }
    }

    void NeuralNetwork::remove_layer(size_t layerIndex) {
      if (this->neurons->size() <= 1) { throw; }
      if (layerIndex == 0) {
        // First index
        for (size_t n2 = 0; n2 < this->neurons->at(layerIndex + 1)->size(); n2++) {
          neurons->at(layerIndex + 1)->at(n2)->set_weights(make_shared<vector<double>>(this->neurons->at(layerIndex)->at(0)->weight_size()));
        }
      } else if (layerIndex != this->neurons->size() - 1) {
        // Last index
        for (size_t n2 = 0; n2 < this->neurons->at(layerIndex + 1)->size(); n2++) {
          this->neurons->at(layerIndex + 1)->at(n2)->set_weights(make_shared<vector<double>>(this->neurons->at(layerIndex - 1)->size() + 1));
        }
      }
      this->neurons->erase(this->neurons->begin() + layerIndex);
    }

    void NeuralNetwork::insert_layer(size_t layerIndex, shared_ptr<vector<shared_ptr<Neuron>>> layer) {
      this->neurons->insert(this->neurons->begin() + layerIndex, layer);
    }

    void NeuralNetwork::add_output_neuron(shared_ptr<Neuron> neuron) {
      this->neurons->at(this->neurons->size() - 1)->push_back(neuron);
    }

    void NeuralNetwork::add_non_output_neuron(size_t layerIndex, shared_ptr<Neuron> neuron, const vector<double>& outputWeights) {
      this->neurons->at(layerIndex)->push_back(neuron);
      for (int n2 = 0; n2 < this->neurons->at(layerIndex + 1)->size(); n2++) {
        this->neurons->at(layerIndex + 1)->at(n2)->push_weight(outputWeights.at(n2));
      }
    }

    /**
    * Splits a neuron into 2 that produce half the output of the original neuron.
    * This effectively adds a neuron without causing the network's behaviour/outputs to change.
    **/
    void NeuralNetwork::split_neuron_non_destructive(size_t layerIndex, size_t neuronIndex) {
      shared_ptr<Neuron> duplicatedNeuron = make_shared<Neuron>(this->neurons->at(layerIndex)->at(neuronIndex)->clone_weights());
      if (layerIndex == this->neurons->size() - 1) {
        this->add_output_neuron(duplicatedNeuron);
      } else {
        vector<double> outputWeights(this->neurons->at(layerIndex + 1)->size());
        for (size_t n2 = 0; n2 < this->neurons->at(layerIndex + 1)->size(); n2++) {
          double halvedWeight = neurons->at(layerIndex + 1)->at(n2)->neuron_weight(neuronIndex);
          outputWeights.at(n2) = halvedWeight;
          this->neurons->at(layerIndex + 1)->at(n2)->set_neuron_weight(neuronIndex, halvedWeight);
        }
        this->add_non_output_neuron(layerIndex, duplicatedNeuron, outputWeights);
      }
    }

    /**
    * Adds a new neuron that is not affected by any neurons from the previous layer and outputs 0 (i.e. always outputs 0).
    * This effectively adds a neuron without causing the network's behaviour/outputs to change.
    */
    void NeuralNetwork::add_neuron_non_destructive(size_t layerIndex) {
      size_t weightsLength = layerIndex == 0 ? 2 : this->neurons->at(layerIndex - 1)->size() + 1;
      shared_ptr<vector<double>> neuronWeights = make_shared<vector<double>>(weightsLength);
      neuronWeights->at(0) = this->threshold_to_result_in_zero();
      shared_ptr<Neuron> addedNeuron = make_shared<Neuron>(neuronWeights);
      if (layerIndex == neurons->size() - 1) {
        this->add_output_neuron(addedNeuron);
      } else {
        vector<double> outputWeights(this->neurons->at(layerIndex + 1)->size());
        this->add_non_output_neuron(layerIndex, addedNeuron, outputWeights);
      }
    }

    shared_ptr<NeuralNetwork> NeuralNetwork::produce_new_neural_network() {
      return make_shared<NeuralNetwork>(this);
    }
}