const outputs = [];

// Ran every time a balls drops into a bucket
function onScoreUpdate(dropPosition, bounciness, size, bucketLabel) {
  outputs.push([dropPosition, bounciness, size, bucketLabel]);
  console.log(outputs);
}

// Write code here to analyze stuff
function runAnalysis() {}
