// Reference: https://www.tensorflow.org/js/guide/nodejs
// require("@tensorflow/tfjs-node"); // CPU will be used for calculations
// require("@tensorflow/tfjs-node-gpu"); // GPU will be used for calculations
const tf = require("@tensorflow/tfjs");
const loadCSV = require("./load-csv");
const LogisticRegression = require("./multinominal-logistic-regression");
const _ = require("lodash");
const plot = require("nodeplotlib");
