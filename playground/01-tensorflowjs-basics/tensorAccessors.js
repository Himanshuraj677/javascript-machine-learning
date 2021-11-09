const data = tf.tensor([10, 20, 30]);

data.get(0); // 10

const twoDData = tf.tensor([
  [20, 30, 40],
  [50, 60, 70],
]);

twoDData.get(1, 0); // row, column // 50
