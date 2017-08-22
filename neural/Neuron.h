#pragma once
#include <vector>
#include <memory>
#include "./INeuron.h"
using namespace std;
namespace neural {
	class Neuron : INeuron {
	private:
		shared_ptr<vector<double>> weights;
	public:
		Neuron(shared_ptr<vector<double>> weights);
	};
}