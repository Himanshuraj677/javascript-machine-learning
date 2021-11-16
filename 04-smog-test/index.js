// Reference: https://www.tensorflow.org/js/guide/nodejs
// require("@tensorflow/tfjs-node"); // CPU will be used for calculations
// require("@tensorflow/tfjs-node-gpu"); // GPU will be used for calculations
const tf = require("@tensorflow/tfjs");
const loadCSV = require("./load-csv");
const LogisticRegression = require("./logistic-regression");

let { features, labels, testFeatures, testLabels } = loadCSV("./cars.csv", {
  dataColumns: ["horsepower", "displacement", "weight"],
  labelColumns: ["passedemissions"],
  shuffle: true,
  splitTest: 50,
  converters: { passedemissions: (value) => (value === "TRUE" ? 1 : 0) },
});

const regression = new LogisticRegression(features, labels, {
  learningRate: 0.5,
  iterations: 100,
  batchSize: 50,
});

regression.train();
console.log(regression.test(testFeatures, testLabels));
