// ---
// title: Linear algebra on plain arrays
// id: lina-linear-algebra
// ---

// %% [markdown]
/*
`@tangent.to/lina` provides the workhorse decompositions of numerical linear
algebra (LU, QR, Cholesky, SVD, symmetric eigendecomposition, least squares)
over ordinary JavaScript arrays. A matrix is a `number[][]` in row-major order
and a vector is a `number[]`; there is no custom matrix class to learn and no
build step to wire up. Every routine is validated against numpy and
scipy.linalg.

The heatmap below shows a `matmul` product as a labelled grid of cells.
*/

// %% [javascript]

import * as __lib from 'https://esm.sh/@tangent.to/lina';
const solve = __lib.solve;
const svd = __lib.svd;
const eigSym = __lib.eigSym;
const lstsq = __lib.lstsq;
const matmul = __lib.matmul;
const rank = __lib.rank;
const cond = __lib.cond;

// A matrix is a nested array of rows; a vector is a flat array. matmul is the
// standard row-by-column product, useful for checking any result by hand.
const P_intro = matmul([[1, 2], [3, 4]], [[5, 6], [7, 8]]); // [[19, 22], [43, 50]]

// Heatmap of the product: each entry (row, col) is a cell coloured and labelled
// by its value, so the shape of the matrix is readable at a glance.
const P_cells = P_intro.flatMap((row, i) => row.map((value, j) => ({ i, j, value })));
Plot.plot({
  marginLeft: 40,
  color: { scheme: 'blues', legend: true, label: 'value' },
  x: { label: 'column', axis: 'top', tickFormat: (d) => d },
  y: { label: 'row' },
  marks: [
    Plot.cell(P_cells, { x: 'j', y: 'i', fill: 'value', inset: 0.5 }),
    Plot.text(P_cells, { x: 'j', y: 'i', text: (d) => d.value, fill: 'black' }),
  ],
});

// %% [markdown]
/*
## Solving a linear system

`solve(A, b)` returns the `x` satisfying `A x = b` using LU decomposition with
partial pivoting, the same method a hand computation would use but numerically
stable. For the system below the answer is exactly `[0.8, 1.4]`, and multiplying
`A` by that vector reproduces `b`.

The grouped bars below place the reconstructed `A·x` beside the target `b`.
*/

// %% [javascript]

const A_sys = [[2, 1], [1, 3]];
const b_sys = [3, 5];
const x_sys = solve(A_sys, b_sys); // [0.8, 1.4]

// Multiply back to confirm A x recovers b. matmul wants a column, so wrap x.
const check_sys = matmul(A_sys, x_sys.map((v) => [v])).map((row) => row[0]);
// solution [0.8, 1.4]; A·x = [3, 5] = b.

// Grouped bars: the reconstructed A·x sits exactly on the target b per equation.
const solveBars = [
  ...check_sys.map((v, i) => ({ eq: `eq ${i}`, series: 'A·x', value: v })),
  ...b_sys.map((v, i) => ({ eq: `eq ${i}`, series: 'b', value: v })),
];
Plot.plot({
  x: { label: null },
  y: { label: 'value', grid: true },
  color: { legend: true },
  fx: { label: 'equation' },
  marks: [
    Plot.barY(solveBars, { fx: 'eq', x: 'series', y: 'value', fill: 'series' }),
    Plot.ruleY([0]),
  ],
});

// %% [markdown]
/*
## Singular values and conditioning

`svd(A)` returns the thin decomposition `{ U, s, V }` with `A = U diag(s) V^T`.
The singular values `s` are non-negative and descending; they measure how much
the matrix stretches space along each principal axis. Their count of nonzero
entries is the rank, and their ratio (largest over smallest) is the condition
number, which `rank` and `cond` report directly.

The bar chart below is a scree plot of the singular-value spectrum.
*/

// %% [javascript]

const M_svd = [[2, 0, 0], [0, 1, 0]];
const { s: sing } = svd(M_svd);
const rank_svd = rank(M_svd); // 2, both singular values are nonzero
const cond_svd = cond(M_svd); // 2 = largest / smallest singular value

// Scree/spectrum plot: one bar per singular value, descending. Their spread is
// the conditioning; the count of nonzero bars is the rank.
Plot.plot({
  x: { label: 'component' },
  y: { label: 'singular value', grid: true },
  caption: `singular values [${sing.map((v) => v.toFixed(2)).join(', ')}] · rank ${rank_svd} · cond ${cond_svd.toFixed(2)}`,
  marks: [
    Plot.barY(sing.map((v, i) => ({ i: `σ${i + 1}`, value: v })), { x: 'i', y: 'value', fill: 'steelblue' }),
    Plot.ruleY([0]),
  ],
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

The bar chart below is the eigenvalue spectrum (variance per component).
*/

// %% [javascript]

const cov = [[2, 1], [1, 2]];
const { values: eigvals, vectors: eigvecs } = eigSym(cov);

// Eigenvectors are columns, so the first principal component is column 0.
const firstPC = eigvecs.map((row) => row[0]); // ~[0.707, 0.707]

// Eigenvalue spectrum: each bar is the variance captured by one principal
// component. The tallest bar is the leading component.
Plot.plot({
  x: { label: 'component' },
  y: { label: 'eigenvalue (variance)', grid: true },
  caption: `eigenvalues [${eigvals.map((v) => v.toFixed(2)).join(', ')}] · first PC ≈ [${firstPC.map((v) => v.toFixed(3)).join(', ')}]`,
  marks: [
    Plot.barY(eigvals.map((v, i) => ({ i: `λ${i + 1}`, value: v })), { x: 'i', y: 'value', fill: 'seagreen' }),
    Plot.ruleY([0]),
  ],
});

// %% [markdown]
/*
## Least squares line fit

For an overdetermined system (more equations than unknowns) there is generally
no exact solution, so `lstsq(A, b)` returns the `x` minimizing `||A x - b||`
along with that `residualNorm`. To fit a line `y = a + b*x`, each row of `A` is
`[1, x_i]` and the solution is `[a, b]`. The points below lie exactly on
`y = 1 + 2x`, so the fit is exact and the residual is zero.

The scatter below overlays the data points with the least-squares fit line.
*/

// %% [javascript]

// Points (0,1), (1,3), (2,5), (3,7): exactly y = 1 + 2x.
const A_fit = [[1, 0], [1, 1], [1, 2], [1, 3]];
const y_fit = [1, 3, 5, 7];
const fit = lstsq(A_fit, y_fit); // x = [1, 2], residual 0

// Perturb one point and refit: the line shifts slightly and the residual grows.
const y_noisy = [1, 3, 4, 7];
const fitNoisy = lstsq(A_fit, y_noisy); // x ~ [0.9, 1.9], residual ~0.837

// Scatter of the (perturbed) data overlaid with the least-squares fit line.
const xs_fit = A_fit.map((row) => row[1]);
const pts_noisy = xs_fit.map((x, i) => ({ x, y: y_noisy[i] }));
const fitLine = xs_fit.map((x) => ({ x, y: fitNoisy.x[0] + fitNoisy.x[1] * x }));
Plot.plot({
  x: { label: 'x', grid: true },
  y: { label: 'y', grid: true },
  caption: `fit y ≈ ${fitNoisy.x[0].toFixed(2)} + ${fitNoisy.x[1].toFixed(2)}·x · residual ≈ ${fitNoisy.residualNorm.toFixed(3)} (exact-data residual ${fit.residualNorm.toFixed(0)})`,
  marks: [
    Plot.line(fitLine, { x: 'x', y: 'y', stroke: 'steelblue' }),
    Plot.dot(pts_noisy, { x: 'x', y: 'y', r: 5, fill: 'orange', stroke: 'black' }),
  ],
});
