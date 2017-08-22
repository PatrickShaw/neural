#pragma once
#include <vector>
#include <memory>
#include "./Neuron.h";
using namespace std;
namespace neural {
  class IReadonlyLayer {
  public:
		virtual shared_ptr<Neuron> neuron(size_t neuronIndex) = 0;
		/**
		* The number of neurons in a given layer.
		*/
		virtual size_t size() = 0;
  };
}
