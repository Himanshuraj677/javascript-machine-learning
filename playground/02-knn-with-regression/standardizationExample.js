// use https://stephengrider.github.io/JSPlaygrounds/

// standardization: value - average / standardDeviation
// standardDeviation: sqrt(variance)

const numbers = tf.tensor([
  [1, 2],
  [3, 4],
  [5, 6],
]);

const { mean, variance } = tf.moments(numbers, 0); // 0 for column axis

console.log(mean); // [3, 4]
console.log(variance); // [2.6666667, 2.6666667]

numbers.sub(mean).div(variance.pow(0.5));
