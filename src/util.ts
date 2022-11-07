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
