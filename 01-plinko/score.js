const outputs = [];
// [[10, 0.5, 16, 1],
// [200, 0.5, 16, 4],
// [350, 0.5, 16, 4],
// [600, 0.5, 16, 5],]
const predictionPoint = 300;
const k = 3;

// Ran every time a balls drops into a bucket
function onScoreUpdate(dropPosition, bounciness, size, bucketLabel) {
  outputs.push([dropPosition, bounciness, size, bucketLabel]);
  // console.log(outputs);
}

// Write code here to analyze stuff
function runAnalysis() {
  const bucket = _.chain(outputs)
    // 1) substract drop point from 300px
    .map((row) => [distance(row[0]), row[3]])
    // 2) sort by ascending order
    .sortBy((row) => row[0])
    // 3) keep the 'k' top records
    .slice(0, k)
    // 4) look at the most common bucket
    .countBy((row) => row[1]) // object {"1": 1, "4": 2}
    .toPairs() // array of arrays [["4",2], ["1",1]]
    .sortBy((row) => row[1]) // sort by ascending order [["1", 1],["4", 2]]
    .last() // most common array [4, 2]
    .first() // most common bucket "4"
    .parseInt() // 4
    .value();
  console.log("Your point will probably fall into", bucket);
}

function distance(point) {
  return Math.abs(point - predictionPoint);
}
