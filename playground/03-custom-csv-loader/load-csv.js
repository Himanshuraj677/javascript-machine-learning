const fs = require("fs");
const _ = require("lodash");

function extractColumns(data, columnNames) {
  const headers = _.first(data);
  const indexes = _.map(columnNames, (column) => headers.indexOf(column));
  const extracted = _.map(data, (row) => _.pullAt(row, indexes));
  return extracted;
}

function loadCSV(
  filename,
  { converters = {}, dataColumns = [], labelColumns = [] }
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
  console.log(data);
}
loadCSV("data.csv", {
  dataColumns: ["height", "value"],
  labelColumns: ["passed"],
  converters: { passed: (val) => (val === "TRUE" ? 1 : 0) },
});
