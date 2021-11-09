// https://stephengrider.github.io/JSPlaygrounds/

// const data = tf.tensor([1, 2, 3])
// data.shape
// [3]

// const data = tf.tensor([
//  [1, 2, 3]
// ])
// data.shape
// [1,3]

// const data = tf.tensor([1, 2, 3])
// const otherData = tf.tensor([4, 5, 6])

// data.add(otherData)
// [5, 7, 9]

// data.sub(otherData)
// [-3, -3, -3]

// data.mul(otherData)
// [4, 10, 18]

// data.div(otherData)
// [0.25, 0.4, 0.5]

const twoDdata = tf.tensor([
  [1, 2, 3],
  [4, 5, 6],
]);
const otherTwoDData = tf.tensor([
  [4, 5, 6],
  [1, 2, 3],
]);

twoDdata.add(otherTwoDData);
// [[5, 7, 9], [5, 7, 9]]
