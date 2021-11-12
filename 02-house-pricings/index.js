// Reference: https://www.tensorflow.org/js/guide/nodejs
// require("@tensorflow/tfjs-node"); // CPU will be used for calculations
// require("@tensorflow/tfjs-node-gpu"); // GPU will be used for calculations
const tf = require("@tensorflow/tfjs");
const loadCSV = require("./load-csv");
// excel viewer extension: https://marketplace.visualstudio.com/items?itemName=GrapeCity.gc-excelviewer

function knn(features, labels, predictionPoint, k) {
  // standardization: value - average / standardDeviation
  // standardDeviation: sqrt(variance)
  const { mean, variance } = tf.moments(features, 0);
  const scaledPrediction = predictionPoint.sub(mean).div(variance.pow(0.5));

  // calculate distance between features and prediction point
  return (
    features
      .sub(mean)
      .div(variance.pow(0.5))
      // features standardization
      .sub(scaledPrediction)
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
    dataColumns: ["lat", "long", "sqft_lot", "sqft_living"],
    labelColumns: ["price"],
  }
);

features = tf.tensor(features);
labels = tf.tensor(labels);

testFeatures.forEach((testPoint, i) => {
  const result = knn(features, labels, tf.tensor(testPoint), 10);
  // error percentage: ((expected value - predicted value) / expected value) * 100
  const err = (testLabels[i][0] - result) / testLabels[i][0];
  console.log("Guess", err * 100, result, testLabels[i][0]);
});

// debugger: node --inspect-brk index.js
// in chrome: about:inspect
