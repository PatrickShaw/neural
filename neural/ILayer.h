#pragma once
#include <vector>
#include <memory>
#include "./Neuron.h"
#include "./IReadonlyLayer.h"
using namespace std;
namespace neural {
	class ILayer : IReadonlyLayer {
	public:
		/**
		* Removes a neuron from a layer in the network.
		*/
		virtual void remove(size_t neuronIndex) = 0;
		/**
		* Adds a neuron to anything other than the output layer.
		*/
		virtual void add_non_output_neuron(shared_ptr<Neuron> neuron, const vector<double>& outputWeights) = 0;
		/**
		* Adds a neuron to the output layer.
		*/
		virtual void add_output_neuron(shared_ptr<Neuron> neuron) = 0;
		/**
		* Splits a neuron into 2 that produce half the output of the original neuron.
		* This effectively adds a neuron without causing the network's behaviour/outputs to change.
		**/
		virtual void split_neuron_non_destructive(size_t neuronIndex) = 0;
		/**
		* Adds a new neuron that is not affected by any neurons from the previous layer and outputs 0 (i.e. always outputs 0).
		* This effectively adds a neuron without causing the network's behaviour/outputs to change.
		*/
		virtual void add_neuron_non_destructive() = 0;
	};
}
