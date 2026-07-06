// ---
// title: Linear algebra on plain arrays
// id: lina-linear-algebra
// ---

// %% [markdown]
/*
# Linear algebra on plain arrays

`@tangent.to/lina` provides the workhorse decompositions of numerical linear
algebra (LU, QR, Cholesky, SVD, symmetric eigendecomposition, least squares)
over ordinary JavaScript arrays. A matrix is a `number[][]` in row-major order
and a vector is a `number[]`; there is no custom matrix class to learn and no
build step to wire up. Every routine is validated against numpy and
scipy.linalg.
*/

// %% [javascript]

import { solve, svd, eigSym, lstsq, matmul, rank, cond } from 'https://esm.sh/@tangent.to/lina';

// A matrix is a nested array of rows; a vector is a flat array. matmul is the
// standard row-by-column product, useful for checking any result by hand.
matmul([[1, 2], [3, 4]], [[5, 6], [7, 8]]); // [[19, 22], [43, 50]]

// %% [markdown]
/*
## Solving a linear system

`solve(A, b)` returns the `x` satisfying `A x = b` using LU decomposition with
partial pivoting, the same method a hand computation would use but numerically
stable. For the system below the answer is exactly `[0.8, 1.4]`, and multiplying
`A` by that vector reproduces `b`.
*/

// %% [javascript]

const A_sys = [[2, 1], [1, 3]];
const b_sys = [3, 5];
const x_sys = solve(A_sys, b_sys); // [0.8, 1.4]

// Multiply back to confirm A x recovers b. matmul wants a column, so wrap x.
const check_sys = matmul(A_sys, x_sys.map((v) => [v])).map((row) => row[0]);

({ solution: x_sys, A_times_x: check_sys, target_b: b_sys });

// %% [markdown]
/*
## Singular values and conditioning

`svd(A)` returns the thin decomposition `{ U, s, V }` with `A = U diag(s) V^T`.
The singular values `s` are non-negative and descending; they measure how much
the matrix stretches space along each principal axis. Their count of nonzero
entries is the rank, and their ratio (largest over smallest) is the condition
number, which `rank` and `cond` report directly.
*/

// %% [javascript]

const M_svd = [[2, 0, 0], [0, 1, 0]];
const { s: sing } = svd(M_svd);

({
  singular_values: sing, // [2, 1]
  rank: rank(M_svd), // 2, both singular values are nonzero
  cond: cond(M_svd), // 2 = largest / smallest singular value
});

// %% [markdown]
/*
## PCA in two lines

`eigSym(A)` computes the eigendecomposition of a symmetric matrix, returning
`{ values, vectors }` with eigenvalues in descending order and the matching
eigenvectors as the columns of `vectors`. Applied to a covariance matrix this
is principal component analysis: the top eigenvalue is the largest variance and
its eigenvector is the first principal component. The matrix below has
eigenvalues 3 and 1, and the leading component points along `[1, 1] / sqrt(2)`.
*/

// %% [javascript]

const cov = [[2, 1], [1, 2]];
const { values: eigvals, vectors: eigvecs } = eigSym(cov);

// Eigenvectors are columns, so the first principal component is column 0.
const firstPC = eigvecs.map((row) => row[0]);

({
  eigenvalues: eigvals, // [3, 1]
  first_principal_component: firstPC, // ~[0.707, 0.707]
});

// %% [markdown]
/*
## Least squares line fit

For an overdetermined system (more equations than unknowns) there is generally
no exact solution, so `lstsq(A, b)` returns the `x` minimizing `||A x - b||`
along with that `residualNorm`. To fit a line `y = a + b*x`, each row of `A` is
`[1, x_i]` and the solution is `[a, b]`. The points below lie exactly on
`y = 1 + 2x`, so the fit is exact and the residual is zero.
*/

// %% [javascript]

// Points (0,1), (1,3), (2,5), (3,7): exactly y = 1 + 2x.
const A_fit = [[1, 0], [1, 1], [1, 2], [1, 3]];
const y_fit = [1, 3, 5, 7];
const fit = lstsq(A_fit, y_fit);

// Perturb one point and refit: the line shifts slightly and the residual grows.
const y_noisy = [1, 3, 4, 7];
const fitNoisy = lstsq(A_fit, y_noisy);

({
  intercept_slope: fit.x, // [1, 2]
  residual: fit.residualNorm, // 0
  noisy_intercept_slope: fitNoisy.x, // ~[0.9, 1.9]
  noisy_residual: fitNoisy.residualNorm, // ~0.837
});
