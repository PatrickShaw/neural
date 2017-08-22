#pragma once
#include <memory>
#include <vector>
using namespace std;
namespace neural {
	namespace training {
		/**
		 * A machine learning supervised trainer. Trains an object to output a set of value for a given set of inputs.
	     */
		template<typename T>
	  class SupervisedTrainer {
		public:
			/**
			* Trains a given trainable entity to output a certain set of outputs, given a certain set of inputs.
			* @param trainable
			* The object being trained.
			* @param trainingInputs
			* The inputs that the trainable will use to train.
			* @param trainingOutputs
			* The outputs that the trainable needs to reproduce given the training inputs.
			*/	
			virtual void train(
				T& trainable,
				const vector<double>& trainingInputs,
				const vector<double>& trainingOutputs
			) = 0;
		};
	}
}
