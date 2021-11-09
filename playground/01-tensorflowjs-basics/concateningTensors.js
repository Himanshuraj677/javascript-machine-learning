const tensorA = tf.tensor([
  [1, 2, 3],
  [4, 5, 6],
]);

const tensorB = tf.tensor([
  [7, 8, 9],
  [10, 11, 12],
]);

// concatening (imagine glue at the bottom)
// axis 0 by default for rows
tensorA.concat(tensorB);
// [[1 , 2 , 3 ], [4 , 5 , 6 ], [7 , 8 , 9 ], [10, 11, 12]]

// joining rows (imagine horizontal arrows)
// axis 1 for columns
tensorA.concat(tensorB, 1);
// [[1, 2, 3, 7 , 8 , 9 ], [4, 5, 6, 10, 11, 12]]
