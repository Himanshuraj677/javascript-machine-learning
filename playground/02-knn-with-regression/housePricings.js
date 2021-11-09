const features = tf.tensor([
  [-121, 47],
  [-121.2, 46.5],
  [-122, 46.4],
  [-120.9, 46.7],
]);

const labels = tf.tensor([[200], [250], [215], [240]]);

const predictionPoint = tf.tensor([-121, 47]);

const k = 2;

// calculate distance between features and prediction point
features
  .sub(predictionPoint)
  .pow(2)
  .sum(1)
  .pow(0.5)
  .expandDims(1)
  .concat(labels, 1) // keep order relationships
  .unstack() // create a JS array of individual tensors

  // sort from lowest to greatest
  .sort((a, b) => (a.get(0) > b.get(0) ? 1 : -1))

  // pick the K nearest values
  .slice(0, k)

  // average the values
  .reduce((acc, pair) => acc + pair.get(1), 0) / // get house value label // 440 (200+240)
  k; // 220
