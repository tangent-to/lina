# tangent/lina

Linear algebra for JavaScript (ESM). Browser-first, zero dependencies,
runs in Node.js and Deno. The **lin**ear **a**lgebra leaf of the
[tangent suite](https://github.com/tangent-to) — MIT-licensed
infrastructure consumed by tangent/ds and tangent/sem.

Matrices are plain nested row-major arrays (`number[][]`) at the API
boundary; computation runs on flat `Float64Array` storage internally.

## What's in it

- **Factorizations**: `lu` (partial pivoting), `qr` (Householder,
  reduced/full), `cholesky`, `svd` (one-sided Jacobi — high relative
  accuracy), `eigSym` (cyclic Jacobi, symmetric matrices)
- **Solvers**: `solve` (vector or multi-RHS), `choleskySolve`, `lstsq`
  (QR-based), `pinvSolve` (minimum-norm, any rank)
- **SVD-derived**: `pinv`, `rank`, `cond`
- **Utilities**: `matmul`, `transpose`, `identity`, `diag`, `norm`
  (fro/1/inf), `trace`, `det`, `inv`, `isPositiveDefinite`

## Install

```bash
npm install @tangent.to/lina     # npm
deno add jsr:@tangent/lina       # Deno / JSR
```

## Usage

```javascript
import { cholesky, eigSym, lstsq, solve, svd } from '@tangent.to/lina';

solve([[2, 1], [1, 3]], [3, 5]);            // [0.8, 1.4]

const { values, vectors } = eigSym(covarianceMatrix);  // PCA in two lines
const { U, s, V } = svd(dataMatrix);

const { x } = lstsq(designMatrix, y);        // OLS coefficients
const L = cholesky(spdMatrix);               // throws if not positive definite
```

## Validation against numpy/scipy

`tests_compare-to-scipy/` checks every operation against
`numpy.linalg`/`scipy.linalg` on seeded random matrices — solve/det/inv,
Cholesky (entrywise vs numpy), QR and SVD invariants, singular values and
symmetric eigenvalues vs numpy, lstsq, pinv (including rank-deficient),
rank and cond. Agreement is at machine precision (~1e-15). Requires
[uv](https://docs.astral.sh/uv/) and Node:

```bash
npm run test:scipy
```

## Scope

Dense, double-precision, textbook-modern algorithms sized for the suite's
workloads (covariance algebra, regression, ordination — n up to a few
hundred). Deliberately out of scope: sparse matrices, complex numbers,
general nonsymmetric eigenproblems (until a consumer needs them), and
BLAS-style micro-optimization — wasm kernels belong in tangent/nd when it
lands.

## Performance

lina favors simple, unconditionally-stable algorithms (one-sided Jacobi
SVD, cyclic Jacobi eigensolver) that hit machine precision on the suite's
target sizes. On `solve`/`inv` it is competitive with the mature
`ml-matrix`; on `matmul` and `svd` it is roughly 1.5–2× slower, and on
symmetric eigendecomposition about 5× slower, because those Jacobi methods
trade speed for simplicity. If a consumer ever profiles past this, the fix
is targeted, not a rewrite: swap the symmetric eigensolver for tridiagonal
QL and the SVD for Golub–Reinsch (the JAMA algorithms), and improve
`matmul` cache locality. Until then, correctness and a zero-dependency
footprint win. lina is not a general-purpose matrix library — for the rich
`Matrix` API (elementwise arithmetic, broadcasting, views, nonsymmetric
eigen), use `ml-matrix`; lina is the focused linear-algebra leaf the
tangent suite's own packages build on.

## License

MIT.
