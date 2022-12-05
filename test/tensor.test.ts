import { expect, test } from 'vitest';
import { prod, Tensor } from '../src/exports';

function test_util(xShape: number[], yShape: number[], op: string, truth: any) {
	const x = Tensor.arange(prod(xShape)).reshape(...xShape);
	const y = Tensor.arange(prod(yShape)).reshape(...yShape);
	// @ts-ignore
	expect(x[op](y).data).toStrictEqual(truth);
}

test('tensor > add', () => {
	test_util([2, 2], [2, 2], 'add', [0, 2, 4, 6]);
	test_util([3, 3], [3, 3], 'add', [0, 2, 4, 6, 8, 10, 12, 14, 16]);
});

test('tensor > sub', () => {
	test_util([2, 2], [2, 2], 'sub', [0, 0, 0, 0]);
	test_util([3, 3], [3, 3], 'sub', [0, 0, 0, 0, 0, 0, 0, 0, 0]);
});

const multiplyMatrices = (a: any, b: any) => {
	if (!Array.isArray(a) || !Array.isArray(b) || !a.length || !b.length) {
		throw new Error('arguments should be in 2-dimensional array format');
	}
	let x = a.length,
		z = a[0].length,
		y = b[0].length;
	if (b.length !== z) {
		// XxZ & ZxY => XxY
		throw new Error(
			'number of columns in the first matrix should be the same as the number of rows in the second'
		);
	}
	let productRow = Array.apply(null, new Array(y)).map(Number.prototype.valueOf, 0);
	let product = new Array(x);
	for (let p = 0; p < x; p++) {
		product[p] = productRow.slice();
	}
	for (let i = 0; i < x; i++) {
		for (let j = 0; j < y; j++) {
			for (let k = 0; k < z; k++) {
				product[i][j] += a[i][k] * b[k][j];
			}
		}
	}
	return product;
};
// 5 x 4
let a = [
	[1, 2, 3, 1],
	[4, 5, 6, 1],
	[7, 8, 9, 1],
	[1, 1, 1, 1],
	[5, 7, 2, 6]
];
// 4 x 6
let b = [
	[1, 4, 7, 3, 4, 6],
	[2, 5, 8, 7, 3, 2],
	[3, 6, 9, 6, 7, 8],
	[1, 1, 1, 2, 3, 6]
];
// should result in a 5 x 6 matrix
// console.log(multiplyMatrices(a, b));

test.skip('tensor > expand', () => {
	const a = new Tensor([0, 1, 2], [1, 3]);
	a.expand(2, 3);
	expect(a.strides).toStrictEqual([0, 1]);
});

test('tensor > get', () => {
	const a = Tensor.arange(4).reshape(2, 2);
	expect(a.get(0, 0)).toBe(0);
	expect(a.get(0, 1)).toBe(1);
	expect(a.get(1, 0)).toBe(2);
	expect(a.get(1, 1)).toBe(3);

	const b = Tensor.arange(12).reshape(3, 2, 2);
	expect(b.get(0, 0, 1)).toBe(1);
	expect(b.get(1, 1, 1)).toBe(7);
	expect(b.get(0, 1, 1)).toBe(3);
});

test('tensor > strides are calculated correctly', () => {
	const a = Tensor.arange(12).reshape(3, 2, 2);
	expect(a.strides).toStrictEqual([4, 2, 1]);

	const b = Tensor.arange(12).reshape(3, 4);
	expect(b.strides).toStrictEqual([4, 1]);

	const c = Tensor.arange(12);
	expect(c.strides).toStrictEqual([1]);
});

test.skip('tensor > slice', () => {
	expect(false).toBe(true);
});

test('tensor > toString', () => {
	const a = Tensor.arange(4);
	const b = Tensor.arange(4).reshape(1, 4);
	const c = Tensor.arange(4).reshape(2, 2);
	const d = Tensor.arange(4).reshape(2, 2).transpose();
	const e = Tensor.arange(12).reshape(3, 2, 2);

	// console.log(d.toString());
	console.log('Stridses', e.strides);
	console.log(e.toString());
	console.log(c.toString());

	// expect(a.toString()).toBe('tensor([0, 1, 2, 3])');
	// expect(b.toString()).toBe('tensor([[0, 1, 2, 3]])');
	expect(c.toString()).toBe('tensor([[0, 1], \n	[2, 3]])');
	expect(d.toString()).toBe('tensor([[0, 2], \n	[1, 3]])');

	// shape 2, 2
	// iterate shape[0] times
	// if shape[i + 1] -> go deeper
	// - when going deeper, construct a tensor from slice
	// - How do we slice? slice start 2 * depth?
	// if not, just print values

	// Read from axis down -> axis 1, then axis 0 (for 3dimensional, axis 2, axis 1, axis 0)
	// step a.strides[axis] until we have stepped a.shape[axis] steps
	// move minus backstride for axis -> (a.shape[axis] - 1) * a.strides[axis]
	// advance a.strides[next axis] -> next axis = 0
	// Repeat steps 1-4 a total of 3 times to read the remaining rows (b.shape[0]) HOW??

	// const stringified = a.transpose().toString();
	// console.log(stringified);
	// expect(stringified).toBe(false);
	// should be
	// array([[0, 2],
	// 				[1, 3]])
});

test('tensor > transpose > strides are correct', () => {
	const a = Tensor.arange(12).reshape(3, 2, 2);
	expect(a.transpose(1, 0, 2).strides).toStrictEqual([2, 4, 1]);

	const b = Tensor.arange(12).reshape(3, 4);
	expect(b.transpose().strides).toStrictEqual([1, 4]);

	const c = Tensor.arange(24).reshape(3, 2, 2, 2);
	expect(c.transpose().strides).toStrictEqual([1, 2, 4, 8]);
	// >>> x.transpose().strides
	// (8, 16, 32, 64)

	const d = Tensor.arange(24).reshape(3, 2, 2, 2);
	expect(d.transpose(1, 0, 2, 3).strides).toStrictEqual([4, 8, 2, 1]);
	// >>> x.transpose(1, 0, 2, 3).strides
	//(32, 64, 16, 8)
});

test.skip('tensor > mul', () => {
	const a = [1, 2, 3, 1, 4, 5, 6, 1, 7, 8, 9, 1, 1, 1, 1, 1, 5, 7, 2, 6];
	const b = [1, 4, 7, 3, 4, 6, 2, 5, 8, 7, 3, 2, 3, 6, 9, 6, 7, 8, 1, 1, 1, 2, 3, 6];

	// const a =
	console.log(Tensor.arange(12).reshape(3, 4).toString());

	const aTensor = new Tensor(a, [5, 4]);
	const bTensor = new Tensor(b, [4, 6]);
	// test_util([2, 2], [2, 2], 'mul', [2, 3, 6, 11]);
	// test_util([3, 3], [3, 3], 'mul', [15, 18, 21, 42, 54, 66, 69, 90, 111]);
	// const a = [[1,2], [3, 4]];
	// const b = [[1,2], [3, 4]];
	console.log(multiplyMatrices(a, b));
	expect(false).toBe(true);
	/*
	[ 15,  18,  21],
  [ 42,  54,  66],
  [ 69,  90, 111]

	/*
	A= [0, 1]
		 [2, 3]

	B= [0, 1]
		 [2, 3]

	A * B = [2, 3]
					[6, 11] */
});

test.todo('tensor > relu');
test.todo('tensor > softmax');
test.todo('tensor > log softmax');
test.todo('tensor > mean');
test.todo('tensor > min');
test.todo('tensor > max');
