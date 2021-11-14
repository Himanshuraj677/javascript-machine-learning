// Reference: https://www.tensorflow.org/js/guide/nodejs
// require("@tensorflow/tfjs-node"); // CPU will be used for calculations
// require("@tensorflow/tfjs-node-gpu"); // GPU will be used for calculations
const tf = require("@tensorflow/tfjs");
const loadCSV = require("./load-csv");
const LinearRegression = require("./linear-regression");

let { features, labels, testFeatures, testLabels } = loadCSV("./cars.csv", {
  shuffle: true,
  splitTest: 50,
  dataColumns: ["horsepower", "weight", "displacement"],
  labelColumns: ["mpg"],
});

const regression = new LinearRegression(features, labels, {
  learningRate: 10,
  iterations: 100,
});

regression.train();

// Use accuracy instead
// console.log(
//   "Updated M is:",
//   regression.weights.arraySync()[1][0],
//   "Updated B is:",
//   regression.weights.arraySync()[0][0]
// );

const r2 = regression.test(testFeatures, testLabels);
console.log("MSE History", regression.mseHistory);
console.log("R2 is", r2);
