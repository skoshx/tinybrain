<p align="center">
<img src="docs/tinybrain-logo.png" />
</p>

> A tiny autograd tensor library written from scratch in TypeScript

[![license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/skoshx/tinybrain/blob/main/LICENSE.md)
[![CI](https://github.com/skoshx/tinybrain/actions/workflows/ci.yml/badge.svg)](https://github.com/skoshx/tinybrain/actions/workflows/ci.yml)
[![prettier](https://img.shields.io/badge/code_style-prettier-ff69b4.svg)](https://github.com/prettier/prettier)
[![jest](https://jestjs.io/img/jest-badge.svg)](https://github.com/facebook/jest)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/skoshx/tinybrain/blob/main/README.md)

This library is a tiny autograd tensor library written from scratch in TypeScript, to learn in depth how everything works and to build intuition. My goal with this project is just to be able to train & run inference on a simple naive MNIST model. I might add convolutions later, if I have time.

## Features

- Inference & training
- Optimizers (SGD, more coming…)
- ~~Runs simple MNIST~~

## Usage

```typescript
import { Tensor } from 'tinybrain';

const x = Tensor.arange(6).reshape(2, 3);
const y = Tensor.uniform(2, 3);
const z = y.mul(x).sum();
z.backward();

console.log(x.grad); // dz/dx
console.log(y.grad); // dz/dy
```

## License

`tinybrain` is released under the [MIT License](https://opensource.org/licenses/MIT).

## TODO

- Basic "DataLoader"
- Training code
- Adam, RMSprop optimizers?
