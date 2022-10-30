// Simple MNIST

import { Tensor } from './tensor';
import { SGD } from './optimizer';

class TinyMNIST {
	public l1: Tensor;
	public l2: Tensor;
	constructor() {
		this.l1 = Tensor.uniform(784, 128);
		this.l2 = Tensor.uniform(128, 10);
	}

	forward(x: Tensor) {
		return x.mul(this.l1).relu().mul(this.l2).logsoftmax();
	}
}

const model = new TinyMNIST();
const optimizer = new SGD([model.l1, model.l2], 0.001);

// todo mnist training data
const X_train: Tensor[] = [];
const Y_train: Tensor[] = [];

const out = model.forward(X_train);
const loss = out.mul(Y_train).mean();
optimizer.zeroGrad();
loss.backward();
optimizer.step();
