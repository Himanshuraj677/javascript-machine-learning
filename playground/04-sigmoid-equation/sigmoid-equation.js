// use https://stephengrider.github.io/JSPlaygrounds/
// sigmoid equation: 1 / (1 + e^-(m * x + b)

const weights = tf.tensor([[1], [1]]);

const features = tf.tensor([
  [1, 95],
  [1, 120],
  [1, 135],
  [1, 175],
]);

features.matMul(weights);
// mx + b
features.matMul(weights).sigmoid();
// sigmoid(mx + b)
