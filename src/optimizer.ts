import { Tensor } from './tensor';

export class Optimizer {
	public params: Tensor[];
	constructor(params: Tensor[]) {
		// TODO: this is kind of wrong, doesn't really make sense to
		// enable grad for all params, should be a distinction between
		// default requiresGrad and just flat out requiresGrad = false;
		for (let i = 0; i < params.length; i++) {
			params[i].requiresGrad = true;
		}
		this.params = params.filter((params) => params.requiresGrad);
	}

	zeroGrad() {
		this.params.forEach((param) => {
			param.grad = null;
		});
	}
}

export class SGD extends Optimizer {
	public learningRate: number;
	constructor(params: Tensor[], learningRate = 0.001) {
		super(params);
		this.learningRate = learningRate;
	}

	step() {
		for (let i = 0; i < this.params.length; i++) {
			const currentParam = this.params[i];
			// TODO: might need broadcasting for this, sigh
			const updatedParam = currentParam.sub(currentParam.grad.mul(this.learningRate));
			this.params[i].assign(updatedParam);
		}
	}
}
