const outputs = [];
// const outputs = [
//   [10, 0.5, 16, 1],
//   [200, 0.5, 16, 4],
//   [350, 0.5, 16, 4],
//   [600, 0.5, 16, 5],
// ];
// const predictionPoint = 300;
// const k = 3;

// Ran every time a balls drops into a bucket
function onScoreUpdate(dropPosition, bounciness, size, bucketLabel) {
  outputs.push([dropPosition, bounciness, size, bucketLabel]);
  // console.log(outputs);
}

// Write code here to analyze stuff
function runAnalysis() {
  const testSetSize = 100;
  const k = 10;
  // const normalizedData = minMax(outputs, 3);
  // const [testSet, trainingSet] = splitDataset(normalizedData, testSetSize);

  // With a for loop
  // let numberCorrect = 0;
  // for (let i = 0; i < testSet.length; i++) {
  //   const bucket = knn(trainingSet, testSet[i][0]); // testSet: [10, 0.5, 16, 1] [point, bounciness, size, predictedBucket]
  //   // console.log(bucket, testSet[i][3]);
  //   if (bucket === testSet[i][3]) {
  //     numberCorrect++;
  //   }
  // }
  // console.log("Accuracy", numberCorrect / testSetSize);

  // With lodash chaining
  // estimate the best k value
  // _.range(1, 20).forEach((k) => {
  //   const accuracy = _.chain(testSet)
  //     // .filter((testPoint) => knn(trainingSet, testPoint[0], k) === testPoint[3])
  //     .filter(
  //       (testPoint) =>
  //         knn(trainingSet, _.initial(testPoint), k) === testPoint[3]
  //     )
  //     .size()
  //     .divide(testSetSize)
  //     .value();
  //   console.log("For k of", k, "accuracy is", accuracy);
  // });

  // pick best feature
  _.range(0, 3).forEach((feature) => {
    // generate a dataset with only one feature and label
    const data = _.map(outputs, (row) => [row[feature], _.last(row)]);
    const normalizedData = minMax(data, 1);
    const [testSet, trainingSet] = splitDataset(normalizedData, testSetSize);
    const accuracy = _.chain(testSet)
      .filter(
        (testPoint) =>
          knn(trainingSet, _.initial(testPoint), k) === _.last(testPoint)
      )
      .size()
      .divide(testSetSize)
      .value();
    console.log("For feature of", feature, "accuracy is", accuracy);
  });
}

function distance(pointA, pointB) {
  // pointA = 300, pointB = 350
  // return Math.abs(pointA - pointB);

  // 3D Pythagorean theorem
  // pointA = [300, 0.5, 16], pointB = [350, 0.55, 16]
  return (
    _.chain(pointA)
      .zip(pointB)
      .map(([a, b]) => (a - b) ** 2)
      .sum()
      .value() ** 0.5
  );
}

function knn(data, point, k) {
  return (
    _.chain(data)
      // 1a) substract drop point from 300px
      // .map((row) => [distance(row[0], point), row[3]])
      // 1b) handle multiple features
      .map((row) => {
        return [distance(_.initial(row), point), _.last(row)];
      })
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

// Normalize witn MinMax
function minMax(data, featureCount) {
  const clonedData = _.cloneDeep(data);
  // iterate over columns
  for (let i = 0; i < featureCount; i++) {
    const column = clonedData.map((row) => row[i]);
    const min = _.min(column);
    const max = _.max(column);
    // iterate over rows to normalize data
    for (let j = 0; j < clonedData.length; j++) {
      clonedData[j][i] = (clonedData[j][i] - min) / (max - min);
    }
  }
  return clonedData;
}
