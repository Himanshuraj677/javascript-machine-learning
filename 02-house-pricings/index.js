// require("@tensorflow/tfjs-node"); // CPU will be used for calculations
const tf = require("@tensorflow/tfjs");
const loadCSV = require("./load-csv");

let { features, labels, testFeatures, testLabels } = loadCSV(
  "kc_house_data.csv",
  {
    shuffle: true,
    splitTest: 10,
    dataColumns: ["lat", "long"],
    labelColumns: ["price"],
  }
);

console.log(testFeatures);
console.log(testLabels);