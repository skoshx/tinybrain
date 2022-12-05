import invariant from 'tiny-invariant';
import { calculateStrides, dump, matchingShape, prod } from './util';

export class Tensor {
	public grad: Tensor | null = null;
	public data: number[] = [];
	public requiresGrad: boolean = true;
	public shape: number[];

	public strides: number[] = [];
	public offset: number = 0;

	constructor(
		data: number[],
		shape: number[],
		requiresGrad = true,
		strides = calculateStrides(shape),
		offset = 0
	) {
		this.data = data;
		this.shape = shape;
		this.requiresGrad = requiresGrad;
		this.strides = strides;
		this.offset = offset;
	}

	public transpose(...order: number[]) {
		if (order.length === 0) order = [...Array(this.strides.length).keys()].reverse();
		// TODO checks that transpose is "legal"
		this.strides = order.map((orderVal) => this.strides[orderVal]);
		return this;
	}

	// TODO write slicing, not sure how we should go about this, since
	// no similar slicing syntax as Python
	public slice(...args: number[]) {
		return this;
	}

	// ChatGPT generated solution :D
	public sliceGPT(start: number[], end: number[]): Tensor {
		// TODO: assert start.length === this.shape.length maybe ?
		// Calculate the new shape and strides for the sliced tensor
		const newShape = [];
		const newStrides = [];
		for (let i = 0; i < this.shape.length; i++) {
			newShape[i] = end[i] - start[i];
			newStrides[i] = this.strides[i];
		}

		// Calculate the new offset for the sliced tensor
		let newOffset = this.offset;
		for (let i = 0; i < this.shape.length; i++) {
			newOffset += start[i] * this.strides[i];
		}

		// Create and return the sliced tensor
		return new Tensor(this.data, newShape, this.requiresGrad, newStrides, newOffset);
	}

	// TODO expand?
	public expand(...newShape: number[]) {
		// TODO assert only expansion of dimensions of size 1.
		// TODO set stride of dimension of size 1 -> 0
		return this;
	}

	// TODO coordinate is a bad name
	public get(...coordinate: number[]) {
		const index = coordinate
			.map((coord, i) => coord * this.strides[i])
			.reduce((prev, curr) => prev + curr, 0);
		return this.data[this.offset + index];
	}

	// TODO write this with strides & offset
	// TODO cap off decimals to 3-4 places
	public toString() {
		return `tensor(${dump(this)})`;
	}

	public static ones(...shape: number[]) {
		return new Tensor(
			Array.from({ length: prod(shape) }, () => 1),
			shape
		);
	}
	public static zeros(...shape: number[]) {
		return new Tensor(
			Array.from({ length: prod(shape) }, () => 0),
			shape
		);
	}
	public static arange(stop: number, start = 0, step = 1) {
		return new Tensor(
			Array.from({ length: stop - start }, (_, n) => n * step),
			[stop - start]
		);
	}
	// uniform distribution
	public static uniform(...shape: number[]) {
		return new Tensor(
			Array.from({ length: prod(shape) }, () => Math.random()),
			shape
		);
	}
	// normal distribution
	public static randn(...shape: number[]) {
		const randn = () => {
			const u = 1 - Math.random();
			const v = Math.random();
			return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
		};
		return new Tensor(
			Array.from({ length: prod(shape) }, () => randn()),
			shape
		);
	}

	public reshape(...newShape: number[]) {
		invariant(
			this.data.length === prod(newShape),
			`reshape() error: cannot reshape tensor with ${this.data.length} items to shape ${newShape}`
		);
		this.shape = newShape;
		this.strides = calculateStrides(this.shape);
		return this;
	}

	public assign(y: Tensor) {
		invariant(
			matchingShape(y.shape, this.shape),
			`assign() error: mismatching shapes ${y.shape} vs. ${this.shape}`
		);
		this.data = y.data;
		return y;
	}

	public sub(y: Tensor) {
		invariant(
			matchingShape(y.shape, this.shape),
			`sub() error: mismatching shapes ${y.shape} vs. ${this.shape}`
		);
		// this is fine just like this, since not going to be implementing broadcasting
		for (let i = 0; i < this.data.length; i++) {
			this.data[i] -= y.data[i];
		}
		return this;
	}

	// A (k * m) + B (n * k) -> C (m * n)
	public mul(y: Tensor) {
		// TODO this assertion is wrong, we should check that axis match
		/* invariant(
			matchingShape(y.shape, this.shape),
			`mul() error: mismatching shapes ${y.shape} vs. ${this.shape}`
		); */
		// todo assert shapes
		// invariant(this.shape[0] === this.shape[1], `mul() error: invalid shapes for matmul shapes (${y.shape}) and (${this.shape}) don't share a similar length axis.`);
		// cnugteren.github.io/tutorial/pages/page2.html
		const M = this.shape[0];
		const K = this.shape[1];
		const N = y.shape[1];

		// const out = Tensor.zeros(N, M);
		const out = Tensor.zeros(M, N);

		console.log('THIs shape');
		console.log(this.shape);
		console.log(y.shape);

		console.log('Outshape', N, M);

		for (let m = 0; m < M; m++) {
			for (let n = 0; n < N; n++) {
				let acc = 0;
				for (let k = 0; k < K; k++) {
					acc += this.data[k * M + m] * y.data[n * K + k];
				}

				//this.data[n * M + m] = acc;
				out.data[n * M + n] = acc;
			}
		}
		return out;
		// return this;
	}

	public add(y: Tensor) {
		invariant(
			matchingShape(y.shape, this.shape),
			`add() error: mismatching shapes ${y.shape} vs. ${this.shape}`
		);
		// same here, no broadcasting
		for (let i = 0; i < this.data.length; i++) {
			this.data[i] += y.data[i];
		}
		return this;
	}

	// activations
	public relu() {
		for (let i = 0; i < this.data.length; i++) {
			if (this.data[i] < 0) this.data[i] = 0;
		}
		return this;
	}
	public logsoftmax() {
		return this;
	}

	// autograd logic

	public backward() {
		// todo: fix below assertion
		invariant(matchingShape(this.shape, [1]), 'backward() error: invalid shape');

		// implicit gradients
		this.grad = Tensor.ones(...this.shape);
	}
}
