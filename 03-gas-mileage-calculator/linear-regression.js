const tf = require("@tensorflow/tfjs");
const _ = require("lodash");

class LinearRegression {
  constructor(features, labels, options) {
    this.features = tf.tensor(features);
    this.labels = tf.tensor(labels);
    // Add a column of 1s to enable matrix multiplication
    this.features = tf
      .ones([this.features.shape[0], 1])
      .concat(this.features, 1);

    this.options = Object.assign(
      { learningRate: 0.1, iterations: 1000 }, // fallback values if no options are provided
      options
    );
    // 1) Pick a value for m and b
    this.weights = tf.zeros([2, 1]); // [[0],[0]];
  }

  gradientDescent() {
    // 2) Calculate slope of MSE with respect to m and b
    // features * ((features * weights)) - labels) / n
    const currentGuesses = this.features.matMul(this.weights);
    const differences = currentGuesses.sub(this.labels);
    const slopes = this.features
      .transpose()
      .matMul(differences)
      .div(this.features.shape[0]);
    // .mul(2) // unnecessary since we don't care about the exact value, only +/- is useful to interpret slope of MSE;

    // 3) and 4) Multiply both slopes by learning rates and substract results from b and m
    this.weights = this.weights.sub(slopes.mul(this.options.learningRate));
  }

  train() {
    for (let i = 0; i < this.options.iterations; i++) {
      this.gradientDescent();
    }
  }
}

module.exports = LinearRegression;
