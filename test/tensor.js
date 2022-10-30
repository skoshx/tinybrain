import { Tensor } from '../dist/exports.mjs';
import { deepEqual } from 'assert';

// matmul
const mx = Tensor.arange(4).reshape(2, 2);
const my = Tensor.arange(4).reshape(2, 2);
const mz = mx.mul(my);

console.log('matmul result: ');
console.log(mz.toString());
// deepEqual(mz.data, [0, 1, 4, 9], 'invalid matmul');

// add

const ax = Tensor.arange(4).reshape(2, 2);
const ay = Tensor.arange(4).reshape(2, 2);
const az = ax.add(ay);

console.log('add result: ');
console.log(az.toString());
deepEqual(az.data, [0, 2, 4, 6], 'invalid add');

// sub
const sx = Tensor.arange(4).reshape(2, 2);
const sy = Tensor.arange(4).reshape(2, 2);
const sz = sx.sub(sy);

console.log('sub result: ');
console.log(sz.toString());
deepEqual(sz.data, [0, 0, 0, 0], 'invalid sub');

console.log('✅ all tests passed');
