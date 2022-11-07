import invariant from 'tiny-invariant';
import { calculateStrides, matchingShape, prod } from './util';

export class Tensor {
	public grad: Tensor | null = null;
	public data: number[] = [];
	public requiresGrad: boolean = true;
	public shape: number[];

	public strides: number[] = [];
	public offset: number = 0;

	constructor(data: number[], shape: number[], offset = 0, requiresGrad = true) {
		this.data = data;
		this.shape = shape;
		this.requiresGrad = requiresGrad;

		// Calculate strides
		this.strides = calculateStrides(shape);
		this.offset = offset;
	}

	public transpose(...order: number[]) {
		if (order.length === 0) order = [...Array(this.strides.length).keys()].reverse();
		// TODO checks that transpose is "legal"
		this.strides = order.map((orderVal) => this.strides[orderVal]);
		return this;
	}

	// TODO write this with strides & offset
	public toString() {
		const lines = [];
		// for (let i = 0; i < this.shape.length - 1; i++) {
		if (this.shape.length - 1 === 0) lines.push(this.data.join(', '));
		for (let i = 0; i < this.shape.length - 1; i++) {
			const ndim = this.shape[i]; // 2
			if (ndim === 1) continue; // todo think this never happens
			for (let j = 0; j < ndim; j++) {
				const stepSize = this.shape[i + 1];
				lines.push(`[${this.data.slice(j * stepSize, j * stepSize + stepSize).join(', ')}]`);
			}
		}

		return `<Tensor data=[${lines.join('\n'.padEnd(15))}] with grad=${this.grad}>`;
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
		const data = [];
		for (let i = 0; i < prod(shape); i++) {
			data.push(Math.random());
		}
		return new Tensor(data, shape);
	}
	// normal distribution
	public static randn(shape: number[]) {
		const normalDist = (x: number, mean = 0, variance = 1) => {
			const firstPart = 1 / (Math.sqrt(variance) * Math.sqrt(2 * Math.PI));
			const secondPart = Math.exp((-1 / 2) * ((x - mean) / Math.sqrt(variance)) ** 2);
			return firstPart * secondPart;
		};
		const data: number[] = [];
		for (let i = 0; i < 3; i++) {
			data.push(normalDist(Math.random() * 5));
		}
		return new Tensor(data, shape);
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

		console.log("THIs shape");
		console.log(this.shape);
		console.log(y.shape);

		console.log("Outshape", N, M);

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
		/* for (let yi = 0; yi < y.shape[0]; yi++) {
			for (let xi = 0; xi < y.shape[0]; xi++) {
				let acc = 0;
				for (let k = 0; k < y.shape[0]; k++) {
					acc += this.data[k + y.shape[0] * yi] * y.data[xi + y.shape[0] * k];
					// acc += A[yi][k] * B[k][xi];
					// acc += 
				}
				this.data[xi + this.shape[0] * yi] = acc;
				// C[yi][xi] = acc;
			}
		} */
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
