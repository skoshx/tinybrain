import { Tensor } from './tensor';

export const prod = (shape: number[]) => {
	return shape.reduce((prev, curr) => prev * curr);
};

export const matchingShape = (leftShape: number[], rightShape: number[]) => {
	if (leftShape.length !== rightShape.length) return false;
	for (let i = 0; i < leftShape.length; i++) {
		if (leftShape[i] !== rightShape[i]) return false;
	}
	return true;
};

export const calculateStrides = (shape: number[]) => {
	const strides = [1];
	for (let i = 1; i < shape.length; i++) {
		strides[i] = strides[i - 1] * shape[i];
	}
	return strides.reverse();
};

export const dump = (tensor: Tensor, ...dimens: number[]) => {
	// if (dimens.length === tensor.shape.length - dimens?.[0] ?? 0) {
	if (dimens.length === tensor.shape.length - 1) {
		let p = '[';
		console.log('trying to add inner stuff', tensor.shape[0]);
		for (let i = 0; i < tensor.shape[0]; i++) {
			console.log('accessing dimens', [...dimens, i]);
			p += tensor.get(...dimens, i) + (i === tensor.shape[0] - 1 ? '' : ', ');
		}
		p += ']';
		return p;
	}
	let s = '[';
	for (let i = 0; i < tensor.shape[0]; i++) {
		if (tensor.shape.length === 1) continue;
		// console.log("Linindex for i ", i);
		const newTensor = new Tensor(
			tensor.data,
			tensor.shape.slice(1),
			false,
			tensor.strides.slice(1)
		);
		console.log('dumping inner, index', i);
		console.log('newtensor ', newTensor);
		s += dump(newTensor, i);
		if (i < tensor.shape[0] - 1) s += ', \n	';
	}
	return (s += ']');
};
