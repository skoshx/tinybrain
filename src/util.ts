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
