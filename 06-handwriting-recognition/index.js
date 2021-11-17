// Reference: https://www.tensorflow.org/js/guide/nodejs
// require("@tensorflow/tfjs-node"); // CPU will be used for calculations
// require("@tensorflow/tfjs-node-gpu"); // GPU will be used for calculations
const tf = require("@tensorflow/tfjs");
const LogisticRegression = require("./multinominal-logistic-regression");
const _ = require("lodash");
const plot = require("nodeplotlib");
const mnist = require("mnist-data");

const mnistData = mnist.training(0, 5000);
const features = mnistData.images.values.map((image) => _.flatMap(image));
const encodedLabels = mnistData.labels.values.map((label) => {
  const row = new Array(10).fill(0);
  row[label] = 1; // row[7] = 1 if label value is 7
  return row; // [ 0, 0, 0, 0, 0, 0, 0, 1, 0, 0  ],
});

const regression = new LogisticRegression(features, encodedLabels, {
  learningRate: 1,
  iterations: 20,
  batchSize: 100, // batch quantity: 5000 / 100 = 50
});

regression.train();

const testMnistData = mnist.testing(0, 100);
const testFeatures = testMnistData.images.values.map((image) =>
  _.flatMap(image)
);
const testEncodedLabels = testMnistData.labels.values.map((label) => {
  const row = new Array(10).fill(0);
  row[label] = 1; // row[7] = 1 if label value is 7
  return row; // [ 0, 0, 0, 0, 0, 0, 0, 1, 0, 0  ],
});

const accuracy = regression.test(testFeatures, testEncodedLabels);
console.log("Accuracy is", accuracy); // 0.89
