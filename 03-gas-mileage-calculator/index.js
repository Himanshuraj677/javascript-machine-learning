// Reference: https://www.tensorflow.org/js/guide/nodejs
// require("@tensorflow/tfjs-node"); // CPU will be used for calculations
// require("@tensorflow/tfjs-node-gpu"); // GPU will be used for calculations
const tf = require("@tensorflow/tfjs");
const loadCSV = require("./load-csv");
const LinearRegression = require("./linear-regression");
const plot = require("nodeplotlib");

let { features, labels, testFeatures, testLabels } = loadCSV("./cars.csv", {
  shuffle: true,
  splitTest: 50,
  dataColumns: ["horsepower", "weight", "displacement"],
  labelColumns: ["mpg"],
});

const regression = new LinearRegression(features, labels, {
  learningRate: 0.1,
  iterations: 3,
  batchSize: 10, // batch gradient descent // 66%
  // batchSize: 1, // stochastic gradient descent // 57%
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
// console.log("MSE History", regression.mseHistory);
console.log("R2 is", r2);

const data = [
  {
    // x: regression.bHistory, // to see gradient descent visually
    y: regression.mseHistory.reverse(),
  },
];
const layout = {
  title: {
    text: "Plotting MSE Values",
    font: {
      family: "Courier New, monospace",
      size: 24,
    },
    xref: "paper",
  },
  xaxis: {
    title: {
      text: "Iteration #",
      font: {
        family: "Courier New, monospace",
        size: 18,
        color: "#7f7f7f",
      },
    },
  },
  yaxis: {
    title: {
      text: "Mean Squared Error",
      font: {
        family: "Courier New, monospace",
        size: 18,
        color: "#7f7f7f",
      },
    },
  },
};

plot.plot(data, layout);
