const jumpData = tf.tensor([
  [70, 70, 70],
  [70, 70, 70],
  [70, 70, 70],
  [70, 70, 70],
]);

const playerData = tf.tensor([
  [1, 160],
  [2, 160],
  [3, 160],
  [4, 160],
]);

jumpData.sum(1); // horizontal axis
// [210, 210, 210, 210]
jumpData.sum(0); // vertical axis
// [280, 280, 280]

// jumpData.sum(1) // shape [4] reduce tensor to 1d
// jumpData.sum(1, true) // shape [4, 1] keep 2d
jumpData.sum(1, true).concat(playerData, 1);
// [[210, 1, 160], [210, 2, 160], [210, 3, 160], [210, 4, 160]]

// more robust method
jumpData.sum(1).expandDims(1).concat(playerData, 1);
// expand dimensions by 1 and convert it to 1 column with four rows, shape [4, 1]
// [[210], [210], [210], [210]]
// [[210, 1, 160], [210, 2, 160], [210, 3, 160], [210, 4, 160]]
