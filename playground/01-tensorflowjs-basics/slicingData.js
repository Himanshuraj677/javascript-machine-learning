const twoDData = tf.tensor([
  [20, 30, 40],
  [50, 60, 70],
  [20, 30, 40],
  [50, 60, 70],
  [20, 30, 40],
  [50, 60, 70],
  [20, 30, 40],
  [50, 60, 70],
]);

twoDData.slice([0, 1], [8, 1]);
// [start index: row, column] [size, not 0 indexed!]
// [[30], [60], [30], [60], [30], [60], [30], [60]]
// data.shape[0] or shortcut -1 to have the length
