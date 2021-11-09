// use https://stephengrider.github.io/JSPlaygrounds/
// broadcasting operations are allowed if shapes are equal or one is '1'

// const data = tf.tensor([1, 2, 3]); // shape [3]
// const otherData = tf.tensor([4]); // shape [1]

// data.add(otherData);
// [5, 6, 7]

const data = tf.tensor([
  [1, 2, 3],
  [4, 5, 6],
]);
const otherData = tf.tensor([[1], [1]]);

data.add(otherData);
// [[2, 3, 4], [5, 6, 7]]
