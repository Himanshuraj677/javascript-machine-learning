// Reference: https://www.tensorflow.org/js/guide/nodejs
// require("@tensorflow/tfjs-node"); // CPU will be used for calculations
// require("@tensorflow/tfjs-node-gpu"); // GPU will be used for calculations
const tf = require("@tensorflow/tfjs");
const loadCSV = require("./load-csv");

function knn(features, labels, predictionPoint, k) {
  // calculate distance between features and prediction point
  return (
    features
      .sub(predictionPoint)
      .pow(2)
      .sum(1)
      .pow(0.5)
      .expandDims(1)
      .concat(labels, 1) // keep order relationships
      .unstack() // create a JS array of individual tensors

      // sort from lowest to greatest
      .sort((a, b) => (a.arraySync()[0] > b.arraySync()[0] ? 1 : -1)) // https://github.com/tensorflow/tfjs/issues/2999

      // pick the K nearest values
      .slice(0, k)

      // average the values
      .reduce((acc, pair) => acc + pair.arraySync()[1], 0) / // get house value label
    k
  );
}

let { features, labels, testFeatures, testLabels } = loadCSV(
  "kc_house_data.csv",
  {
    shuffle: true,
    splitTest: 10,
    dataColumns: ["lat", "long"],
    labelColumns: ["price"],
  }
);

features = tf.tensor(features);
labels = tf.tensor(labels);

testFeatures.forEach((testPoint, i) => {
  const result = knn(features, labels, tf.tensor(testPoint), 10);
  // error percentage: ((expected value - predicted value) / expected value) * 100
  const err = (testLabels[i][0] - result) / testLabels[i][0];
  console.log("Guess", err, result, testLabels[i][0]);
});
