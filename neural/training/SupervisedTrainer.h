#pragma once
#include <memory>
namespace neural::training {
  template<T>
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
    void train(T& trainable, const vector<double>& trainingInputs, const vector<double>& trainingOutputs);
  };
}
