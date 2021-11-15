const fs = require("fs");
const _ = require("lodash");
const shuffleSeed = require("shuffle-seed");

function extractColumns(data, columnNames) {
  const headers = _.first(data);
  const indexes = _.map(columnNames, (column) => headers.indexOf(column));
  const extracted = _.map(data, (row) => _.pullAt(row, indexes));
  return extracted;
}

function loadCSV(
  filename,
  {
    converters = {},
    dataColumns = [],
    labelColumns = [],
    shuffle = true,
    splitTest = false,
  }
) {
  let data = fs.readFileSync(filename, { encoding: "utf-8" });
  // if you need to remove '\r': https://stackoverflow.com/questions/21640902/remove-r-cr-from-csv
  data = data.split("\n").map((row) => row.split(","));
  // remove empty columns
  data = data.map((row) => _.dropRightWhile(row, (val) => val === ""));
  const headers = _.first(data);
  data = data.map((row, index) => {
    if (index === 0) {
      return row;
    }
    return row.map((element, index) => {
      // parse booleans
      if (converters[headers[index]]) {
        const converted = converters[headers[index]](element);
        return _.isNaN(converted) ? element : converted;
      }
      // parse number values
      const result = parseFloat(element);
      return _.isNaN(result) ? element : result;
    });
  });
  let labels = extractColumns(data, labelColumns);
  data = extractColumns(data, dataColumns);
  // remove headers
  data.shift();
  labels.shift();
  // shuffle
  if (shuffle) {
    data = shuffleSeed.shuffle(data, "UseSamePhraseToHaveSameShuffling");
    labels = shuffleSeed.shuffle(labels, "UseSamePhraseToHaveSameShuffling");
    // change phrase if you want to reshuffle between runs
  }
  // split test and training data
  if (splitTest) {
    const trainSize = _.isNumber(splitTest)
      ? splitTest
      : Math.floor(data.length / 2);
    return {
      features: data.slice(0, trainSize),
      labels: labels.slice(0, trainSize),
      testFeatures: data.slice(trainSize),
      testLabels: labels.slice(trainSize),
    };
  } else {
    return { features: data, labels };
  }
}

const { features, labels, testFeatures, testLabels } = loadCSV("data.csv", {
  dataColumns: ["height", "value"],
  labelColumns: ["passed"],
  shuffle: true,
  splitTest: 1,
  converters: { passed: (val) => (val === "TRUE" ? 1 : 0) },
});

console.log("Features", features);
console.log("Labels", labels);
console.log("Test Features", testFeatures);
console.log("Test Labels", testLabels);
