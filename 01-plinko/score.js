const outputs = [];
// const outputs = [
//   [10, 0.5, 16, 1],
//   [200, 0.5, 16, 4],
//   [350, 0.5, 16, 4],
//   [600, 0.5, 16, 5],
// ];
// const predictionPoint = 300;
const k = 3;

// Ran every time a balls drops into a bucket
function onScoreUpdate(dropPosition, bounciness, size, bucketLabel) {
  outputs.push([dropPosition, bounciness, size, bucketLabel]);
  // console.log(outputs);
}

// Write code here to analyze stuff
function runAnalysis() {
  const testSetSize = 10;
  const [testSet, trainingSet] = splitDataset(outputs, testSetSize);
  let numberCorrect = 0;

  // With a for loop
  // for (let i = 0; i < testSet.length; i++) {
  //   const bucket = knn(trainingSet, testSet[i][0]); // testSet: [10, 0.5, 16, 1] [point, bounciness, size, predictedBucket]
  //   // console.log(bucket, testSet[i][3]);
  //   if (bucket === testSet[i][3]) {
  //     numberCorrect++;
  //   }
  // }
  // console.log("Accuracy", numberCorrect / testSetSize);

  // With lodash chaining
  const accuracy = _.chain(testSet)
    .filter((testPoint) => knn(trainingSet, testPoint[0]) === testPoint[3])
    .size()
    .divide(testSetSize)
    .value();
  console.log("Accuracy is", accuracy);
}

function distance(pointA, pointB) {
  return Math.abs(pointA - pointB);
}

function knn(data, point) {
  return (
    _.chain(data)
      // 1) substract drop point from 300px
      .map((row) => [distance(row[0], point), row[3]])
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
      .value()
  );
}

// Randomize test data
function splitDataset(data, testCount) {
  const shuffled = _.shuffle(data);
  const testSet = _.slice(shuffled, 0, testCount);
  const trainingSet = _.slice(shuffled, testCount);
  return [testSet, trainingSet];
}
