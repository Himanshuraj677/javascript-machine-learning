// node --inspect-brk memory.js
// use memory tab in Chrome Debugger to take snapshot

const _ = require("lodash");

const loadData = () => {
  const randoms = _.range(0, 999999);
  //   return randoms;
};

const data = loadData();

debugger;

console.log(data);
