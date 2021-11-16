const tf = require("@tensorflow/tfjs");

class LogisticRegression {
  constructor(features, labels, options) {
    this.features = this.processFeatures(features);
    this.labels = tf.tensor(labels);
    this.mseHistory = [];

    this.options = Object.assign(
      { learningRate: 0.1, iterations: 1000, batchSize: 10 }, // fallback values if no options are provided
      options
    );
    // 1) Pick a value for Ms and B
    this.weights = tf.zeros([this.features.shape[1], 1]); // [[0],[0]];
  }

  gradientDescent(features, labels) {
    // 2) Calculate cross entropy with respect to m and b
    // sigmoid equation: 1 / (1 + e^-(m * x + b)
    const currentGuesses = features.matMul(this.weights).sigmoid();
    const differences = currentGuesses.sub(labels);
    const slopes = features
      .transpose()
      .matMul(differences)
      .div(features.shape[0]);

    // 3) and 4) Multiply both slopes by learning rates and substract results from b and m
    this.weights = this.weights.sub(slopes.mul(this.options.learningRate));
  }

  train() {
    // Handle batchs
    const { batchSize } = this.options;
    const batchQuantity = Math.floor(this.features.shape[0] / batchSize);
    for (let i = 0; i < this.options.iterations; i++) {
      for (let j = 0; j < batchQuantity; j++) {
        const startIndex = j * batchSize;
        const featureSlice = this.features.slice(
          [startIndex, 0],
          [batchSize, -1]
        );
        const labelSlice = this.labels.slice([startIndex, 0], [batchSize, -1]);
        this.gradientDescent(featureSlice, labelSlice);
      }
      this.recordMSE();
      this.updateLearningRate();
    }
  }

  predict(observations) {
    return this.processFeatures(observations).matMul(this.weights).sigmoid();
  }

  test(testFeatures, testLabels) {
    testFeatures = this.processFeatures(testFeatures);
    testLabels = tf.tensor(testLabels);

    const predictions = testFeatures.matMul(this.weights);

    // Sum of squares of residuals
    const res = testLabels.sub(predictions).pow(2).sum().arraySync();
    // Total sum of squares
    const tot = testLabels.sub(testLabels.mean()).pow(2).sum().arraySync();
    // Coefficient of determination
    return 1 - res / tot;
  }

  processFeatures(features) {
    features = tf.tensor(features);
    // Standardize values
    if (this.mean && this.variance) {
      features = features.sub(this.mean).div(this.variance.pow(0.5));
    } else {
      features = this.standardize(features);
    }
    // Add a column of 1s to enable matrix multiplication
    features = tf.ones([features.shape[0], 1]).concat(features, 1);
    return features;
  }

  standardize(features) {
    const { mean, variance } = tf.moments(features, 0);
    this.mean = mean;
    this.variance = variance;
    return features.sub(mean).div(variance.pow(0.5));
  }

  recordMSE() {
    // Calculate vectorized MSE
    const mse = this.features
      .matMul(this.weights)
      .sub(this.labels)
      .pow(2)
      .sum()
      .div(this.features.shape[0])
      .arraySync();
    this.mseHistory.unshift(mse); // rather than push() to make comparisons easier;
  }

  updateLearningRate() {
    if (this.mseHistory.length < 2) {
      return;
    }
    if (this.mseHistory[0] > this.mseHistory[1]) {
      this.options.learningRate /= 2; // = this.options.learningRate / 2;
    } else {
      this.options.learningRate *= 1.05;
    }
  }
}

module.exports = LogisticRegression;
